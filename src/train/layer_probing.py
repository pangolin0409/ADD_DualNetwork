import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_curve
import numpy as np
import matplotlib.pyplot as plt
import onnxruntime as ort
from transformers import Wav2Vec2FeatureExtractor
from src.data.load_datasets import load_datasets
from config.config import init
from tqdm import trange, tqdm

def compute_eer(pred_scores, labels):
    fpr, tpr, thresholds = roc_curve(labels.cpu().numpy(), pred_scores.cpu().numpy())
    fnr = 1 - tpr
    eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
    eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]
    return eer * 100

class ProbeHead(nn.Module):
    def __init__(self, input_dim=1024, num_classes=2):
        super().__init__()
        self.proj = nn.Linear(input_dim, num_classes)

    def forward(self, x):  # x: (B, D)
        return self.proj(x)

def extract_features_from_onnx(waveform, processor, session):
    inputs = processor(waveform, sampling_rate=16000, return_tensors="np")
    input_values = inputs["input_values"]
    attention_mask = inputs["attention_mask"]

    if input_values.ndim == 3 and input_values.shape[0] == 1:
        input_values = np.squeeze(input_values, axis=0)
    if attention_mask.ndim == 3 and attention_mask.shape[0] == 1:
        attention_mask = np.squeeze(attention_mask, axis=0)

    input_values = input_values.astype(np.float16)
    attention_mask = attention_mask.astype(np.int64)

    outputs = session.run(None, {
        "input_values": input_values,
        "attention_mask": attention_mask
    })

    all_hidden_states = np.array(outputs[1])
    all_hidden_states = np.transpose(all_hidden_states, (1, 0, 2, 3))  # (B, 25, T, D)
    return torch.tensor(all_hidden_states, dtype=torch.float32).to(
        torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )

def run_layer_probe_from_loader(train_dataloader, test_loaders:dict, processor, onnx_path, input_dim=1024, num_classes=2, epochs=3, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probe_heads = nn.ModuleList([ProbeHead(input_dim, num_classes).to(device) for _ in range(24)]).to(device)
    optimizer = torch.optim.Adam(probe_heads.parameters(), lr=lr)
    session = ort.InferenceSession(onnx_path, providers=["CUDAExecutionProvider"])
    for epoch in trange(epochs):
        total_loss = 0.0
        for batch in tqdm(train_dataloader):
            waveform, labels = batch
            waveform = waveform.to(device)
            labels = labels.to(device)
            hidden_states = extract_features_from_onnx(waveform, processor, session)  # (B, 25, T, D)
            hidden_states = hidden_states[:, 1:]  # Drop conv layer, get 24 transformer layers
            for i in range(24):
                pooled = hidden_states[:, i].mean(dim=1)  # (B, D)
                logits = probe_heads[i](pooled)
                loss = F.cross_entropy(logits, labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                avg_loss = total_loss / (len(train_dataloader) * 24)
            print(f"[Epoch {epoch+1}] Avg loss: {avg_loss / 24:.4f}")

    for test_name, test_loader in test_loaders.items():
        all_probs = [[] for _ in range(24)]
        all_labels = []
        for batch in tqdm(test_loader):
            waveform, labels = batch
            waveform = waveform.to(device)
            labels = labels.to(device)
            hidden_states = extract_features_from_onnx(waveform, processor, session)[:, 1:]

            for i in range(24):
                pooled = hidden_states[:, i].mean(dim=1)
                with torch.no_grad():
                    logits = probe_heads[i](pooled)
                probs = F.softmax(logits, dim=-1)[:, 1]
                all_probs[i].append(probs.cpu())
            all_labels.append(labels.cpu())

        # concat everything
        all_labels = torch.cat(all_labels)

        eer_list = []
        for i in range(24):
            probs_i = torch.cat(all_probs[i])
            eer = compute_eer(probs_i, all_labels)
            eer_list.append(eer)

        plt.figure(figsize=(10, 5))
        plt.plot(range(1, 25), eer_list, marker='o')
        plt.xlabel("XLS-R Layer")
        plt.ylabel("EER (%)")
        plt.title(f"Layer-wise Probing EER on {test_name}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f"layer_wise_probing_eer_on_{test_name}.png")

if __name__ == "__main__":
    args = init()
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    onnx_session = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"]) 
    train_loader = load_datasets(sample_rate=args.nb_samp, batch_size=args.batch_size, dataset_names=args.datasets
    , worker_size=args.nb_worker, target_fake_ratio=1, part='train', is_downsample=False)

    tests = ["in_the_wild", "SOTA", "Asvspoof2019_LA", "ASVspoof2021_DF"]
    test_loaders = {}
    for test in tests:
        test_loader = load_datasets(sample_rate=args.nb_samp, batch_size=args.batch_size, dataset_names=[test]
        , worker_size=args.nb_worker, target_fake_ratio=1, part='test', is_downsample=False)
        test_loaders[test] = test_loader
    run_layer_probe_from_loader(train_loader, test_loaders, processor, args.onnx_path, input_dim=1024, num_classes=2, epochs=3, lr=args.lr)
    
