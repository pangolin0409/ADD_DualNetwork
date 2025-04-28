import gc
import os
import json
import torch
import random
import numpy as np
from tqdm import trange, tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
from transformers import Wav2Vec2FeatureExtractor
import onnxruntime as ort
from src.models.Detector import Detector, MemoryBank, UnknownMemoryBank, ContrastiveLoss
from src.data.load_datasets import load_datasets
from src.utils.eval_metrics import compute_eer
import wandb
from src.utils.common_utils import get_git_branch, send_discord

# 把所有「隨機」都固定下來，讓每次訓練結果都一樣
# 可重現實驗結果
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Dual-space pseudo label: 根據 cosine similarity 將 feature 與 prototype 進行比對 ---
def compute_pseudo_label(features, prototypes, mode='hard'):
    """
    features: (B, D)
    prototypes: (num_classes, D)
    mode: 'hard' 直接 argmax, 'soft' 返回 softmax 分佈
    """
    # 需要正規化 features 與 prototypes，否則受到長度影響
    features = F.normalize(features, dim=1)
    prototypes = F.normalize(prototypes, dim=1)
    sim = torch.matmul(features, prototypes.T)  # (B, num_classes)
    
    # 根據 cosine similarity 計算 pseudo label
    # 這裡我們可以選擇 hard 或 soft 方式
    # hard: 直接 argmax，soft: 使用 softmax 計算概率分佈
    if mode == 'hard':
        return torch.argmax(sim, dim=1)
    else:
        return F.softmax(sim, dim=1)

"""
    模型一個「低信心懲罰」，來幫助模型辨識 unknown spoof
    pesudo label 給予較高的信心，routing 跟 MoE 都猜一樣的，模型假的機率要更有信心（機率高）
    反之，如果 routing 跟 MoE 猜的不同，則給予較低的信心（機率低）

    意思是 routing, moe 行為不應該硬分，而是保持中立太度，為了接下來 unknown proto 做準備
    為接下來的「unknown proto clustering」或「reject decision」留空間
"""
def compute_consistency_loss(feature_labels, routing_labels, logits, soft=False):
    conf = torch.max(F.softmax(logits, dim=1), dim=1)[0]  # [B] → 最大類別機率
    target = (feature_labels == routing_labels).float() # 一樣為 1，不一樣為 0
    # 也可以用 soft 方式，若不一致給 0.1 的信心
    if soft:
        target = target * 1.0 + (1 - target) * 0.1
    loss = F.binary_cross_entropy(conf, target)
    return loss

# === 推離 unknown prototype 的 loss ===
def push_away_from_unknown(features, unknown_prototypes, temperature=0.07):
    if unknown_prototypes is None or features.size(0) == 0:
        return torch.tensor(0.0, device=features.device)
    features = F.normalize(features, dim=1)
    unknown_prototypes = F.normalize(unknown_prototypes, dim=1)
    logits = torch.matmul(features, unknown_prototypes.T)  # (B, K)
    logits = logits / temperature
    loss = -torch.logsumexp(logits, dim=1).mean()
    return loss

def safe_release(*objs):
    for obj in objs:
        if obj is not None:
            if isinstance(obj, nn.Module):
                obj.cpu()
            del obj
    gc.collect()
    torch.cuda.empty_cache()

###########################################
# 訓練程式碼
###########################################
def train_model(args):
    webhook = args.discord_webhook

    wandb.init(
        project="audio-deepfake-detection",  #專案名稱
        name=f"{get_git_branch()}_{args.model_name}",  # 實驗名稱
        config=vars(args),
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    # ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
    model = None
    onnx_session = None
    try:
        # 初始化模型架構
        onnx_session = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"]) 
        model = Detector(encoder_dim=args.encoder_dim, num_experts=args.num_experts, num_classes=args.num_classes
                         , router_temperature=args.router_temperature, processor=processor, onnx_session=onnx_session).to(device)

        # Optimizer 與 Scheduler（此處沿用你原本的設定）
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        warmup_epochs = args.warmup_epochs
        scheduler_warmup = LambdaLR(optimizer, lambda epoch: min(epoch / warmup_epochs, 1.0))
        scheduler_step = StepLR(optimizer, step_size=5, gamma=0.7)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_step], milestones=[warmup_epochs])
        
        # 建立 DataLoader (假設 load_datasets 已實現)
        train_loader, val_loader = load_datasets(sample_rate=args.nb_samp, batch_size=args.batch_size, dataset_names=args.datasets
        , worker_size=args.nb_worker, target_fake_ratio=1, test=False, is_downsample=False)

        # Loss 計算：主要包括 CrossEntropy、Contrastive Loss、Consistency Loss、MoE 正則化（load balance, limp）
        ce_loss_fn = nn.CrossEntropyLoss()

        best_eer = 999.0
        best_val_loss = float("inf")
        patience_counter = 0

        os.makedirs(os.path.join(args.save_path, args.model_name), exist_ok=True)
        os.makedirs(os.path.join(args.log_path, args.model_name), exist_ok=True)
        best_model_path = os.path.join(args.save_path, args.model_name, "best_model.pth")

        for epoch in trange(args.num_epochs, desc="Epochs"):
            model.train()
            total_loss = 0.0
            total_samples = 0
            correct = 0
            for batch_idx, (wave, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
                label = label.to(device)
                wave = wave.to(device)
                optimizer.zero_grad()
                # forward pass，這裡我們根據一定機率啟用 router augmentation
                # random.random() < args.router_aug_prob, 則啟用 routing augmentation
                router_aug_flag = (random.random() < args.router_aug_prob)
                # spoof 樣本才做 mask
                aug_mask = (label == 1)
                logits, routing, fused_output = model(wave=wave, router_aug=router_aug_flag, aug_mask=aug_mask)
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
                
                print(f"loss_ce: {loss_ce.item():.4f}, loss_limp: {loss_limp.item():.4f}, loss_load: {loss_load.item():.4f}, loss_entropy: {loss_entropy.item():.4f}, loss_router_supervised: {loss_router_supervised_loss.item():.4f}")

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
            eer, val_loss = validate(model, val_loader, device, args)
            print(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}")

            with open(os.path.join(args.log_path, args.model_name, "log.txt"), "a") as f:
                f.write(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}\n")
                f.write(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.5f}\n")
                f.write(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}\n")

            # Save checkpoint
            model_path = os.path.join(args.save_path, args.model_name, f"checkpt_last.pth")
            torch.save(model.state_dict(), model_path)
            if eer < best_eer:
                best_eer = eer
                best_model_path = os.path.join(args.save_path, args.model_name, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
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

    finally:
        safe_release(model, onnx_session)
        print("Cleaned up models and sessions.")

    print(f"Training done. Best EER = {best_eer:.4f}")

###########################################
# 驗證函數（包含 EER 計算）
###########################################
def validate(model, val_loader, device, args):
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
            logits, routing, fused_output = model(wave=wave)
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
