import os
import json
import torch
import random
from models.classifier.ASSIST import AasistEncoder
from models.Detector_2 import Detector
from load_datasets import load_datasets
from config import init
from utils.eval_metrics import compute_eer
from torch.nn.functional import softmax
from tqdm import trange, tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

##############################################
# 1) 損失函式 (範例：CE + EER, 也可擴充 contrastive)
##############################################
def compute_loss(logits, labels):
    criterion = nn.CrossEntropyLoss()
    return criterion(logits, labels)

def load_balancing_loss(gating, eps=1e-8):
    """
    gating: [B, num_experts]，每個 sample 對各 expert 的 gating 機率
    目標：每個 expert 在整個 batch 中的平均被選中概率應接近 1/num_experts
    """
    B, num_experts = gating.shape
    avg_prob = gating.mean(dim=0)  # [num_experts]
    target = torch.ones_like(avg_prob) / num_experts
    loss = F.mse_loss(avg_prob, target)
    return loss

##############################################
# 2) 訓練流程
##############################################
def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 2.1 讀取 AASIST config
    # with open(args.aasist_config_path, "r") as f_json:
    #     aasist_config = json.loads(f_json.read())
    # aasist_model_config = aasist_config["model_config"]

    # 2.2 建立 AASIST Encoder
    # aasist_encoder = AasistEncoder(aasist_model_config).to(device)

    # 2.4 建立整合模型: FusionMoE (CrossAttn + MoE + AASIST)
    # 其中 AASIST Encoder 可能只處理 formants; 你可以依實際需求調整
    model = Detector().to(device)

    # 2.5 構造 Dataset & DataLoader
    # 你的 RawAudio 只需回傳 (filename, label) or waveform; 其餘特徵在 __getitem__ 內計算
    train_loader, val_loader = load_datasets(sample_rate=args.nb_samp, batch_size=args.batch_size, dataset_names=args.datasets, worker_size=args.nb_worker, target_fake_ratio=2, test=False)
    # 2.6 Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 設置 warmup 學習率調度器
    warmup_epochs = 5
    scheduler_warmup = LambdaLR(optimizer, lambda epoch: min(epoch / warmup_epochs, 1.0))

    # 設置 StepLR 學習率調度器
    scheduler_step = StepLR(optimizer, step_size=3, gamma=0.5)

    # 使用 SequentialLR 自動切換學習率策略
    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer,
                                                    schedulers=[scheduler_warmup, scheduler_step],
                                                    milestones=[warmup_epochs])
    
    os.makedirs(os.path.join(args.save_path, args.model_name), exist_ok=True)
    best_model_path = os.path.join(args.save_path, args.model_name, "best_model.pth")

    # Early stopping parameters
    patience = 5
    patience_counter = 0
    best_val_loss = float("inf")  # 確保第一個 epoch 會被記錄
    best_eer = 999.0
    # 2.7 進行 epoch 訓練
    for epoch in trange(args.num_epochs, desc="Epochs"):
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        for (wave, label, wav2vec_ft) in tqdm(train_loader, desc=f"Epoch {epoch}"):
            label = label.to(device)
            wave = wave.to(device)
            wav2vec_ft = wav2vec_ft.to(device)
            
            # 前向
            logits, local_gating, global_gating = model(wave, wav2vec_ft)
            # logits= model(wave, wav2vec_ft)
            local_moe_loss = load_balancing_loss(local_gating)
            global_moe_loss = load_balancing_loss(global_gating)
            # 計算損失
            loss = compute_loss(logits, label)
            print(f"Loss: {loss.item():.4f}, Local MoE Loss: {local_moe_loss.item():.4f}, Global MoE Loss: {global_moe_loss.item():.4f}")
            loss = loss + (local_moe_loss + global_moe_loss) * args.alpha

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * label.size(0)
            total_samples += label.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()

        avg_loss = total_loss / total_samples
        train_acc = correct / total_samples
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}")

        scheduler.step()
        print(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.5f}")
        with open(os.path.join(args.save_path, args.model_name, "log.txt"), "a") as f:
            f.write(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.5f}")
            
        # 驗證
        eer, val_loss = validate(model, val_loader, device)
        print(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}")
        #Write to txt
        with open(os.path.join(args.save_path, args.model_name, "log.txt"), "a") as f:
            f.write(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.4f}\n")
            f.write(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}\n")  

        # Save checkpoint
        torch.save(model.state_dict(), os.path.join(args.save_path, args.model_name, f"checkpt_{epoch}.pth"))

        if eer < best_eer:
            best_eer = eer
            torch.save(model.state_dict(), best_model_path)
            print("=> Best model updated.")

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs.")
                break

    print(f"Training done. Best EER = {best_eer:.4f}")
    return

##############################################
# 3) 驗證: compute EER
##############################################
def validate(model, val_loader, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    score_list = []
    label_list = []
    with torch.no_grad():
        for (wave, label, wav2vec_ft) in tqdm(val_loader, desc="Validation"):
            label = label.to(device)
            wave = wave.to(device)
            wav2vec_ft = wav2vec_ft.to(device)
            
            logits, local_gating, global_gating = model(wave, wav2vec_ft)
            local_moe_loss = load_balancing_loss(local_gating)
            global_moe_loss = load_balancing_loss(global_gating)
            loss = criterion(logits, label)

            total_loss += loss.item() * label.size(0)
            total_loss += (local_moe_loss + global_moe_loss) * args.alpha
            total_samples += label.size(0)

            scores = softmax(logits, dim=1)[:, 1]  # "spoof" score
            score_list.append(scores)
            label_list.append(label)

    avg_loss = total_loss / total_samples

    # EER 計算
    scores = torch.cat(score_list, 0).data.cpu().numpy()
    labels = torch.cat(label_list, 0).data.cpu().numpy()
    eer, frr, far, threshold = compute_eer(scores[labels == 1], scores[labels == 0])
    return eer, avg_loss

##############################################
# main
##############################################
if __name__ == "__main__":
    args = init()
    train_model(args)
