import gc
import os
import torch
import random
import numpy as np
from tqdm import trange, tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
from transformers import Wav2Vec2FeatureExtractor
import onnxruntime as ort
from src.models.Detector import Detector, WaveformAugmentor
from src.data.load_datasets import load_datasets
from src.utils.eval_metrics import compute_eer
import wandb
from src.utils.common_utils import get_git_branch, send_discord
from src.utils.DatasetUtils import manipulations
# 把所有「隨機」都固定下來，讓每次訓練結果都一樣
# 可重現實驗結果
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def safe_release(*objs):
    for obj in objs:
        if obj is not None:
            if isinstance(obj, nn.Module):
                obj.cpu()
            del obj
    gc.collect()
    torch.cuda.empty_cache()

###########################################
# Contrastive Loss 計算
###########################################
def prototype_to_prototype_loss(prototypes, temperature=0.1):
    L = prototypes.size(0)
    p0 = F.normalize(prototypes[:, 0, :], dim=-1)  # bona fide: [L, D]
    p1 = F.normalize(prototypes[:, 1, :], dim=-1)  # spoof:     [L, D]

    sim = (p0 * p1).sum(dim=-1) / temperature  # [L] cosine similarity

    # 我們希望它們「越不像越好」→ 目標是 0 → 使用 MSE loss against 0
    loss = F.mse_loss(sim, torch.zeros_like(sim))

    return loss

def sample_to_prototype_contrastive_loss(pooled, labels, prototypes, routing_weights, temperature=0.07):
    B, L, D = pooled.shape

    pooled_flat = pooled.view(B * L, D)
    labels_flat = labels.unsqueeze(1).expand(B, L).reshape(-1)
    expert_ids = torch.arange(L, device=pooled.device).repeat(B)
    routing_flat = routing_weights.reshape(-1)

    features = F.normalize(pooled_flat, dim=-1)
    proto = F.normalize(prototypes[expert_ids, labels_flat], dim=-1)
    sim = (features * proto).sum(dim=-1)  # [B * L]

    eps = 1e-6
    log_sim = torch.log((sim / temperature).clamp(min=eps))  # avoid log(0)
    layer_losses = -log_sim

    denom = routing_flat.sum().clamp(min=eps)
    weighted_loss = (layer_losses * routing_flat).sum() / denom

    return weighted_loss


###########################################
# 訓練程式碼
###########################################

def train_loop(args, model, device):
    webhook = args.discord_webhook

    # Optimizer 與 Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    scheduler_warmup = LambdaLR(optimizer, lambda epoch: min(epoch / warmup_epochs, 1.0))
    scheduler_step = StepLR(optimizer, step_size=5, gamma=0.7)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_step], milestones=[warmup_epochs])
    
    train_loader = load_datasets(sample_rate=args.nb_samp, batch_size=args.batch_size, dataset_names=args.datasets
    , worker_size=args.nb_worker, target_fake_ratio=1, part='train', is_downsample=False)
    val_loader = load_datasets(sample_rate=args.nb_samp, batch_size=args.batch_size, dataset_names=args.datasets
    , worker_size=args.nb_worker, target_fake_ratio=1, part='validation', is_downsample=False)
    quick_val_loader = load_datasets(sample_rate=args.nb_samp, batch_size=args.batch_size, dataset_names=args.datasets
    , worker_size=args.nb_worker, target_fake_ratio=1, part='validation', is_downsample=True)

    selected_keys = args.selected_augs if hasattr(args, "selected_augs") else list(manipulations.keys())
    selected_manipulations = {k: manipulations[k] for k in selected_keys if k in manipulations}
    print(f"👉 Selected augmentations: {selected_keys}")
    augmentor = WaveformAugmentor(
        manipulation_pool=selected_manipulations,
        activation_prob=args.aug_prob,
        min_aug=1,
        max_aug=2,
        verbose=True,
        target_len = args.nb_samp,
    )
    
    # Loss 計算：主要包括 CrossEntropy、Contrastive Loss、Consistency Loss、MoE 正則化（load balance, limp）
    ce_loss_fn = nn.CrossEntropyLoss()

    best_eer = 999.0
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in trange(args.num_epochs, desc="Epochs"):
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        augmentor.update(epoch)  # 更新 waveform augmentation 的參數
        for batch_idx, (wave, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            label = label.to(device)
            wave = wave.to(device)
            fake_wave = wave[label == 1]  # 假人聲
            real_wave = wave[label == 0]  # 真人聲
            fake_wave = augmentor(fake_wave)  # 只對假人聲進行增強
            wave = torch.cat([real_wave, fake_wave], dim=0)  # 合併增強後的假人聲與真人聲
            label = torch.cat([label[label == 0], label[label == 1]], dim=0)

            optimizer.zero_grad()

            logits, routing, fused_output, time_pooled_feat = model(wave=wave, epoch=epoch)

            # CrossEntropy Loss (分類 loss)
            loss_ce = ce_loss_fn(logits, label)

            # MoE 正則化：load balance 與 limp loss
            loss_limp = model.compute_limp_loss(routing)
            loss_load = model.compute_load_balance_loss(routing)
            loss_entropy = model.compute_entropy_loss(routing)
            loss_router_supervised_loss = model.compute_router_supervised_loss(routing, logits, label)

            # 組合總 loss
            total_batch_loss = (args.lambda_ce * loss_ce + 
                                args.lambda_limp * loss_limp +
                                args.lambda_load * loss_load +
                                args.lambda_entropy * loss_entropy+
                                args.lambda_router_supervised * loss_router_supervised_loss)

            if torch.isnan(total_batch_loss):
                send_discord("⚠️ NaN detected in total loss", webhook)
                raise RuntimeError("Stop training due to NaN.")

            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item() * label.size(0)
            total_samples += label.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()

        avg_loss = total_loss / total_samples
        train_acc = correct / total_samples

        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
        
        scheduler.step()
        print(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.5f}")
        
        # 驗證
        eer, val_loss = validate(model, quick_val_loader, device, epoch, args.router_temperature, args.router_alpha)
        # if epoch != 0 and epoch % 5 == 0:
        #     eer, val_loss = validate(model, val_loader, device, epoch, args.router_temperature, args.router_alpha)
        # else:
        #     eer, val_loss = validate(model, quick_val_loader, device, epoch, args.router_temperature, args.router_alpha)
        print(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}")

        with open(os.path.join(args.log_path, args.model_name, "log.txt"), "a") as f:
            f.write(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}\n")
            f.write(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.5f}\n")
            f.write(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}\n")

        # Save checkpoint
        model_path = os.path.join(args.save_path, args.model_name, f"checkpt_epoch_{epoch}.pth")
        best_model_path = os.path.join(args.save_path, args.model_name, "best_model.pth")
        torch.save({
                'model_state_dict': model.state_dict(),
                'best_temp': args.router_temperature,
                'best_alpha': args.router_alpha,
                'epoch': epoch,
            }, model_path)
        if eer < best_eer:
            best_eer = eer
            torch.save({
                'model_state_dict': model.state_dict(),
                'best_temp': args.router_temperature,
                'best_alpha': args.router_alpha,
                'epoch': epoch,
            }, best_model_path)
            send_discord(f"✨ 新最佳模型：{args.model_name} | EER: {eer:.4f}", webhook)
            print("=> Best model updated.")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                send_discord(f"❌ 提前停止訓練：{args.model_name} | EPOCH: {epoch+1} | EER: {best_eer:.4f}", webhook)
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

        wandb.log({
            "EER": eer,
            "accuracy": train_acc,
            "loss/total": float(total_loss),
            "epoch_loss": float(avg_loss),
            "epoch_val_loss": float(val_loss),
            "loss/ce": loss_ce.item(),
            "loss/loss_limp": loss_limp.item(),     
            "loss/loss_load": loss_load.item(),
            "loss/loss_entropy": loss_entropy.item(),
            "loss/loss_router_supervised": loss_router_supervised_loss.item(),
        })

    return best_eer

def train_model(args):
    wandb.init(
        project=args.project_name,  #專案名稱
        name=f"{get_git_branch()}_{args.model_name}",  # 實驗名稱
        config=vars(args),
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    
    os.makedirs(os.path.join(args.save_path, args.model_name), exist_ok=True)
    os.makedirs(os.path.join(args.log_path, args.model_name), exist_ok=True)

    model = None
    onnx_session = None
    try:
        # 初始化模型架構
        onnx_session = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"]) 
        model = Detector(encoder_dim=args.encoder_dim, num_experts=args.num_experts, num_classes=args.num_classes
                         , max_temp=args.max_temp, min_temp=args.min_temp, warmup_epochs=args.warmup_epochs , processor=processor, onnx_session=onnx_session).to(device)
        best_eer = train_loop(args, model, device)
    finally:
        safe_release(model, onnx_session)
        print("Cleaned up models and sessions.")

    print(f"Training done. Best EER = {best_eer:.4f}")

###########################################
# 驗證函數（包含 EER 計算）
###########################################
def validate(model, val_loader, device, epoch, router_temperature, router_alpha):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    ce_loss_fn = nn.CrossEntropyLoss()
    score_list = []
    label_list = []
    
    with torch.no_grad():
        for (wave, label) in tqdm(val_loader, desc="Validation"):
            label = label.to(device)
            wave = wave.to(device)
            logits, routing, fused_output, time_pooled_feat = model(wave=wave, epoch=epoch)
            loss = ce_loss_fn(logits, label)
            total_loss += loss.item() * label.size(0)
            total_samples += label.size(0)
            scores = F.softmax(logits, dim=1)[:, 1]  # spoof 機率

            score_list.append(scores)
            label_list.append(label)
    
    avg_loss = total_loss / total_samples
    scores = torch.cat(score_list, 0).cpu().numpy()
    labels = torch.cat(label_list, 0).cpu().numpy()
    # compute_eer() 是你已有的函數
    eer, frr, far, threshold = compute_eer(scores[labels == 1], scores[labels == 0])
    return eer, avg_loss

def main(args):
    train_model(args)

###########################################
# 主程式入口
###########################################
if __name__ == "__main__":
    from config.config import init
    args = init()
    # 這裡假設 init() 會返回一個包含所有參數的 args 物件
    main()
