import os
import json
import torch
import random
import numpy as np
from tqdm import trange, tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
from src.models.ASSIST import AasistEncoder
from src.data.load_datasets import load_datasets
from config.config import init
from utils.eval_metrics import compute_eer

# 把所有「隨機」都固定下來，讓每次訓練結果都一樣
# 可重現實驗結果
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class LayerWiseSENet(nn.Module):
    def __init__(self, num_layers=3, feature_dim=1024, reduction=16):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.reduction = reduction

        self.fc1 = nn.Linear(num_layers * feature_dim, (num_layers * feature_dim) // reduction)
        self.fc2 = nn.Linear((num_layers * feature_dim) // reduction, num_layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, layer_feats):
        """
        layer_feats: Tensor of shape [B, L, T, D]
        B: batch size, L: number of layers (e.g., 3), T: time steps, D: feature dim (e.g., 1024)
        """
        B, L, T, D = layer_feats.shape

        # Step 1: mean pooling over time: [B, L, D]
        pooled = layer_feats.mean(dim=2)

        # Step 2: flatten features: [B, L * D]
        pooled_flat = pooled.reshape(B, L * D)

        # Step 3: pass through FC to get weights
        x = F.relu(self.fc1(pooled_flat))             # [B, hidden]
        weights = self.softmax(self.fc2(x))           # [B, L]

        # Step 4: apply weights: [B, L, 1, 1] * [B, L, T, D] → [B, T, D]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        fused = (weights * layer_feats).sum(dim=1)     # sum over L

        return fused, weights  # fused: [B, T, D], weights: [B, L]

class DownStreamLinearClassifier(nn.Module):
    def __init__(self, encoder, input_depth=1024):
        super(DownStreamLinearClassifier, self).__init__()
        self.senet = LayerWiseSENet(feature_dim=input_depth, num_layers=3, reduction=16)
        self.input_depth = input_depth
        self.proj = nn.Linear(input_depth, 128)
        self.encoder = encoder  # this should be able to encoder the input(batch_size, 64600) into feature vectors(batch_size, input_depth)
        self.fc = nn.Linear(160, 2)  

    def forward(self, x):
        # x, fused_weights = self.senet(x) # (batch, T, encoder_dim)
        x = self.proj(x)  # (batch, 128)
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.encoder(x)
        x = self.fc(x)
        x = x.squeeze(1)
        return x
    
###########################################
# 訓練程式碼
###########################################
def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立模型：假設你使用我們之前定義的 Detector 模組
    # 讀取 AASIST config
    with open(args.aasist_config_path, "r") as f_json:
        aasist_config = json.loads(f_json.read())
    aasist_model_config = aasist_config["model_config"]
    aasist_encoder = AasistEncoder(aasist_model_config).to(device)
    model = DownStreamLinearClassifier(aasist_encoder, input_depth=1024).to(device)
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
        for batch_idx, (wave, label, wav2vec_fts) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            optimizer.zero_grad()
            label = label.to(device)
            wav2vec_ft = wav2vec_fts[:,2,:,:].to(device)
            logits = model(wav2vec_ft)

            # CrossEntropy Loss (分類 loss)
            loss_ce = ce_loss_fn(logits, label)

            # 組合總 loss（權重可根據 args 調整）
            total_batch_loss = (args.lambda_ce * loss_ce)
            total_batch_loss.backward()
            optimizer.step()

            total_loss += total_batch_loss.item() * label.size(0)
            total_samples += label.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == label).sum().item()

        avg_loss = total_loss / total_samples
        train_acc = correct / total_samples
        epoch_inconsistency_ratio = total_inconsistent / total_samples
        print(f"[Epoch {epoch}] Inconsistent Samples: {total_inconsistent}/{total_samples} ({epoch_inconsistency_ratio:.3f})")

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
            f.write(f"[Epoch {epoch}] Inconsistent Samples: {total_inconsistent}/{total_samples} ({epoch_inconsistency_ratio:.3f})\n")

        # Save checkpoint
        model_path = os.path.join(args.save_path, args.model_name, f"checkpt_{epoch}.pth")
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
        for (wave, label, wav2vec_fts) in tqdm(val_loader, desc="Validation"):
            label = label.to(device)
            wav2vec_ft = wav2vec_fts[:,2,:,:].to(device)
            logits = model(wav2vec_ft)
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

###########################################
# 主程式入口
###########################################
if __name__ == "__main__":
    # 這裡假設 init() 會返回一個包含所有參數的 args 物件
    args = init()
    train_model(args)
