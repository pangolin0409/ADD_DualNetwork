import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm, trange
from transformers import Wav2Vec2FeatureExtractor
import onnxruntime as ort
import os, random, numpy as np
from models.Detector import LayerSelectorMoE, Classifier
from utils.eval_metrics import compute_eer
from src.data.load_datasets import load_datasets
from config import init
import gc

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def compute_total_loss(logits, labels, selector_scores, lambda_limp=0.1):
    ce = F.cross_entropy(logits, labels)
    probs = F.softmax(selector_scores, dim=1)
    limp = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()
    return ce + lambda_limp * limp, ce.item(), limp.item()

def safe_release(*objs):
    for obj in objs:
        if obj is not None:
            if isinstance(obj, nn.Module):
                obj.cpu()
            del obj
    gc.collect()
    torch.cuda.empty_cache()

def train_layer_selector(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Init wav2vec processor & ONNX session
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    onnx_session = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"])

    try:
        # Init model
        selector = LayerSelectorMoE(topk=args.top_k, hidden_dim=args.encoder_dim, proj_dim=args.expert_dim, processor=processor, onnx_session=onnx_session).to(device)
        classifier = Classifier(input_dim=args.expert_dim, num_classes=args.num_classes).to(device)

        optimizer = torch.optim.AdamW(list(selector.parameters()) + list(classifier.parameters()), lr=args.lr)

        # Load data
        train_loader, val_loader = load_datasets(
            sample_rate=args.nb_samp, batch_size=args.batch_size,
            dataset_names=args.datasets, worker_size=args.nb_worker,
            target_fake_ratio=1, test=False, is_downsample=False)

        best_eer = float('inf')
        os.makedirs(os.path.join(args.save_path, args.model_name), exist_ok=True)

        for epoch in trange(args.num_epochs, desc="Training Epochs"):
            selector.train(), classifier.train()
            total_loss, total_ce, total_limp = 0.0, 0.0, 0.0
            correct, total = 0, 0

            for wave, label in tqdm(train_loader, desc=f"Epoch {epoch}"):
                wave, label = wave.to(device), label.to(device)
                optimizer.zero_grad()

                weighted, scores, _, _ = selector(wave)
                logits = classifier(weighted)
                loss, ce_val, limp_val = compute_total_loss(logits, label, scores, lambda_limp=args.lambda_limp)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * label.size(0)
                total_ce += ce_val * label.size(0)
                total_limp += limp_val * label.size(0)
                pred = torch.argmax(logits, dim=1)
                correct += (pred == label).sum().item()
                total += label.size(0)

            print(f"[Epoch {epoch}] Loss: {total_loss/total:.4f}, CE: {total_ce/total:.4f}, Limp: {total_limp/total:.4f}, Acc: {correct/total:.4f}")

            # Eval
            eer, val_loss = validate(selector, classifier, val_loader, device)
            print(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}")
            torch.save({
                    'selector': selector.state_dict(),
                    'classifier': classifier.state_dict()
                }, os.path.join(args.save_path, args.model_name, "checkpt_last.pth"))
            if eer < best_eer:
                best_eer = eer
                torch.save({
                    'selector': selector.state_dict(),
                    'classifier': classifier.state_dict()
                }, os.path.join(args.save_path, args.model_name, "best_model.pth"))
                print("=> Saved best model")

        print(f"Training done. Best EER = {best_eer:.4f}")
    finally:
        safe_release(selector, classifier, onnx_session)
        print("Cleaned up models and sessions.")

def validate(selector, classifier, val_loader, device):
    selector.eval(), classifier.eval()
    score_list, label_list = [], []
    total_loss = 0.0
    ce_loss = nn.CrossEntropyLoss()

    with torch.no_grad():
        for wave, label in tqdm(val_loader, desc="Validation"):
            wave, label = wave.to(device), label.to(device)
            weighted, scores, _, _ = selector(wave)
            logits = classifier(weighted)
            loss = ce_loss(logits, label)
            total_loss += loss.item() * label.size(0)

            p_spoof = F.softmax(logits, dim=1)[:, 1]  # spoof prob
            score_list.append(p_spoof)
            label_list.append(label)

    scores = torch.cat(score_list).cpu().numpy()
    labels = torch.cat(label_list).cpu().numpy()
    eer, frr, far, threshold = compute_eer(scores[labels == 1], scores[labels == 0])
    return eer, total_loss / len(labels)

def main():
    args = init()
    set_seed(args.seed)
    train_layer_selector(args)

if __name__ == "__main__":
    main()
