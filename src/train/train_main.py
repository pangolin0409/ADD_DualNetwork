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
from src.utils.common_utils import get_git_branch

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
    wandb.init(
        project="audio-deepfake-detection",  #專案名稱
        name=f"{get_git_branch()}_{args.model_name}",  # 實驗名稱
        config=vars(args),
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True)

    try:
        # 初始化模型架構
        onnx_session = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"]) 
        model = Detector(encoder_dim=args.encoder_dim, expert_dim=args.expert_dim, 
        top_k=args.top_k, num_classes=args.num_classes
        , processor=processor, onnx_session=onnx_session).to(device)

        # 建立 Memory Bank 用來儲存 prototype（對比學習 anchor）
        # 這裡我們有 2 類: 0: bonafide, 1: spoof；並且用相同維度更新 feature 和 routing prototype
        feature_mem = MemoryBank(feature_dim=args.encoder_dim, num_classes=args.num_classes, momentum=0.99).to(device)
        routing_mem = MemoryBank(feature_dim=24, num_classes=args.num_classes, momentum=0.99).to(device)
        unknown_mem = UnknownMemoryBank(feature_dim=args.expert_dim, num_clusters=3).to(device)
        
        # 建立 Contrastive Loss（NT-Xent Loss）
        contrastive_loss_fn = ContrastiveLoss(temperature=0.07)

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
            total_inconsistent = 0
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
                
                # 初始化 prototype（只執行一次）
                if epoch == 0 and batch_idx == 0:
                    with torch.no_grad():
                        for cls in [0, 1]:
                            cls_mask = label == cls
                            if cls_mask.any():
                                avg_feat = fused_output[cls_mask].mean(dim=0)
                                feature_mem.prototypes[cls] = F.normalize(avg_feat, dim=0)
                                routing_mem.prototypes[cls] = F.normalize(routing[cls_mask].mean(dim=0), dim=0)

                # CrossEntropy Loss (分類 loss)
                loss_ce = ce_loss_fn(logits, label)

                # 從 logits 計算 spoof probability (index 1 為 spoof)
                spoof_prob = F.softmax(logits, dim=1)[:, 1]

                # MoE 正則化：load balance 與 limp loss
                loss_limp = model.compute_limp_loss(routing)
                loss_load = model.compute_load_balance_loss(routing)
                # loss_entropy = model.moe.compute_entropy_loss(routing)
                
                # Dual-space pseudo label：利用 moe_output (feature space) 與 routing (routing space)
                # 這裡我們用 MemoryBank 中的 prototypes作為 anchor
                pseudo_label_feat = compute_pseudo_label(fused_output, feature_mem.prototypes, mode='hard')
                pseudo_label_rout = compute_pseudo_label(routing, routing_mem.prototypes, mode='hard')

                # Consistency loss
                if epoch >= args.consistency_warmup:
                    loss_consistency = compute_consistency_loss(pseudo_label_feat, pseudo_label_rout, logits, soft=True)
                    # 這些樣本是 known，我們希望它們遠離 unknown prototype
                    known_mask = (pseudo_label_feat == pseudo_label_rout)
                    known_feats = fused_output[known_mask]
                    loss_pushaway = push_away_from_unknown(known_feats, unknown_mem.get())
                else:
                    loss_consistency = torch.tensor(0.0, device=device)
                    loss_pushaway = torch.tensor(0.0, device=device)

                # Contrastive loss（用 moe_output 來對比）
                loss_contrastive = contrastive_loss_fn(fused_output, label)

                # 組合總 loss（權重可根據 args 調整）
                total_batch_loss = (args.lambda_ce * loss_ce + 
                                    args.lambda_contrastive * loss_contrastive + 
                                    args.lambda_consistency * loss_consistency + 
                                    args.lambda_limp * (loss_limp + loss_load) + 
                                    args.lambda_unknown * loss_pushaway)
                print(f'loss_ce: {loss_ce:.4f}, loss_contrastive: {loss_contrastive:.4f}, loss_consistency: {loss_consistency:.4f}, loss_limp: {loss_limp:.4f}, loss_load: {loss_load:.4f}, loss_pushaway: {loss_pushaway:.4f}')
                total_batch_loss.backward()
                optimizer.step()

                # 更新 MemoryBank (prototype) for feature and routing spaces
                with torch.no_grad():
                    feature_mem.update(fused_output, label)
                    routing_mem.update(routing, label)

                total_loss += total_batch_loss.item() * label.size(0)
                total_samples += label.size(0)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == label).sum().item()

                inconsistent = (pseudo_label_feat != pseudo_label_rout)  # (B,)
                batch_inconsistent_count = inconsistent.sum().item()
                total_inconsistent += batch_inconsistent_count

            if epoch % args.unknown_cluster_interval == 0:
                inconsistent_mask = (pseudo_label_feat != pseudo_label_rout)
                if inconsistent_mask.sum() >= unknown_mem.get().size(0):
                    cluster_input = fused_output[inconsistent_mask].detach().cpu().numpy()
                    kmeans = KMeans(n_clusters=unknown_mem.get().size(0), random_state=42)
                    kmeans.fit(cluster_input)
                    centroids = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32, device=device)
                    unknown_mem.update_from_kmeans(centroids)

            avg_loss = total_loss / total_samples
            train_acc = correct / total_samples
            epoch_inconsistency_ratio = total_inconsistent / total_samples
            print(f"[Epoch {epoch}] Inconsistent Samples: {total_inconsistent}/{total_samples} ({epoch_inconsistency_ratio:.3f})")

            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")
            
            scheduler.step()
            print(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.5f}")
            
            # 驗證
            eer, val_loss = validate(model, val_loader, device, args, unknown_mem)
            print(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}")

            with open(os.path.join(args.log_path, args.model_name, "log.txt"), "a") as f:
                f.write(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}\n")
                f.write(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.5f}\n")
                f.write(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}\n")
                f.write(f"[Epoch {epoch}] Inconsistent Samples: {total_inconsistent}/{total_samples} ({epoch_inconsistency_ratio:.3f})\n")

            # Save checkpoint
            model_path = os.path.join(args.save_path, args.model_name, f"checkpt_last.pth")
            torch.save(model.state_dict(), model_path)
            if eer < best_eer:
                best_eer = eer
                best_model_path = os.path.join(args.save_path, args.model_name, "best_model.pth")
                torch.save(model.state_dict(), best_model_path)
                print("=> Best model updated.")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= args.patience:
                    print(f"Early stopping triggered after {epoch+1} epochs.")
                    break

            wandb.log({
                "EER": eer,
                "accuracy": train_acc,
                "loss/total": float(total_loss),
                "epoch_loss": float(avg_loss),
                "epoch_val_loss": float(val_loss),
                "loss/ce": loss_ce.item(),
                "loss/contrastive": loss_contrastive.item(),
                "loss/consistency": loss_consistency.item(),
                "loss/loss_limp": loss_limp.item(),     
                "loss/loss_load": loss_load.item(),
                "loss/pushaway": loss_pushaway.item(),
                "loss/unknown_cluster": epoch_inconsistency_ratio,
            })

    finally:
        safe_release(onnx_session)
        print("Cleaned up models and sessions.")

    print(f"Training done. Best EER = {best_eer:.4f}")

###########################################
# 驗證函數（包含 EER 計算）
###########################################
def validate(model, val_loader, device, args, unknown_mem):
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
            
            p_spoof = F.softmax(logits, dim=1)[:, 1]  # spoof 機率
            # === 新增信心修正 ===
            unk_proto = unknown_mem.get()     # (K, D)
            features = F.normalize(fused_output, dim=1)
            unk_proto = F.normalize(unk_proto, dim=1)

            # 計算每個樣本與 unknown prototype 的最大相似度
            sim = torch.matmul(features, unk_proto.T)  # (B, K)
            sim_max, _ = torch.max(sim, dim=1)         # (B,)

            # 調整 spoof probability（加權上升）
            lambda_adj = 0.2 
            scores = torch.clamp(p_spoof + lambda_adj * sim_max, 0, 1)
            
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
