import os
import json
import torch
import random
import numpy as np
from tqdm import trange, tqdm
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
from models.classifier.ASSIST import AasistEncoder
from models.Detector_2 import Detector, MemoryBank, UnknownMemoryBank, ContrastiveLoss
from load_datasets import load_datasets
from config import init
from utils.eval_metrics import compute_eer
from torch.nn.functional import softmax
from sklearn.cluster import KMeans
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor

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

###########################################
# 訓練程式碼
###########################################
def train_model(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 建立模型：假設你使用我們之前定義的 Detector 模組
    # 讀取 AASIST config
    # with open(args.aasist_config_path, "r") as f_json:
    #     aasist_config = json.loads(f_json.read())
    # aasist_model_config = aasist_config["model_config"]
    # aasist_encoder = AasistEncoder(aasist_model_config).to(device)
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    wav2vec_model = Wav2Vec2Model.from_pretrained(args.wav2vec_path, output_hidden_states=True).to(device)
    wav2vec_model.eval()
    
    model = Detector(encoder_dim=args.encoder_dim, expert_dim=args.expert_dim, num_classes=args.num_classes, 
                 layer_ids=[9, 10, 11, 12], fusion_mode="avg").to(device)

    # 建立 Memory Bank 用來儲存 prototype（對比學習 anchor）
    # 這裡我們有 2 類: 0: bonafide, 1: spoof；並且用相同維度更新 feature 和 routing prototype
    
    # 建立 Contrastive Loss（NT-Xent Loss）
    contrastive_loss_fn = ContrastiveLoss(temperature=0.07)

    # Optimizer 與 Scheduler（此處沿用你原本的設定）
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = args.warmup_epochs
    scheduler_warmup = LambdaLR(optimizer, lambda epoch: min(epoch / warmup_epochs, 1.0))
    scheduler_step = StepLR(optimizer, step_size=3, gamma=0.5)
    scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_step], milestones=[warmup_epochs])
    
    # 建立 DataLoader (假設 load_datasets 已實現)
    train_loader, val_loader = load_datasets(sample_rate=args.nb_samp, batch_size=args.batch_size, dataset_names=args.datasets
    , worker_size=args.nb_worker, target_fake_ratio=1, test=False, is_downsample=False)

    # Loss 計算：主要包括 CrossEntropy、Contrastive Loss、Consistency Loss、MoE 正則化（load balance, limp）
    ce_loss_fn = nn.CrossEntropyLoss()

    best_eer = 999.0
    best_val_loss = float("inf")
    patience = args.patience
    patience_counter = 0


    os.makedirs(os.path.join(args.save_path, args.model_name), exist_ok=True)
    best_model_path = os.path.join(args.save_path, args.model_name, "best_model.pth")

    for epoch in trange(args.num_epochs, desc="Epochs"):
        model.train()
        total_loss = 0.0
        total_samples = 0
        correct = 0
        total_inconsistent = 0
        for batch_idx, (wave, label, wav2vec_ft) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            label = label.to(device)
            # wav2vec_ft = wav2vec_ft.to(device)
            waveforms_list = [w.cpu().numpy() for w in wave] 
            inputs = processor(waveforms_list, sampling_rate=16000, return_tensors="pt", padding=True)
            print(inputs['input_values'].shape)  # 應該是 [batch_size, seq_len]
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            wav2vec_hidden_state = wav2vec_model(input_values, attention_mask=attention_mask)
            optimizer.zero_grad()
            logits, moe_output = model(wav2vec_hidden_state.hidden_states)

            # CrossEntropy Loss (分類 loss)
            loss_ce = ce_loss_fn(logits, label)

            # Contrastive loss（用 moe_output 來對比）
            loss_contrastive = contrastive_loss_fn(moe_output, label)

            # 組合總 loss（權重可根據 args 調整）
            total_batch_loss = (args.lambda_ce * loss_ce + 
                                args.lambda_contrastive * loss_contrastive)
            # print(f'loss_ce: {loss_ce:.4f}, loss_contrastive: {loss_contrastive:.4f}, loss_consistency: {loss_consistency:.4f}, loss_limp: {loss_limp:.4f}, loss_load: {loss_load:.4f}, loss_entropy: {loss_entropy:.4f}, loss_pushaway: {loss_pushaway:.4f}')
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
        eer, val_loss = validate(model, val_loader, device, args, processor, wav2vec_model)
        print(f"[Epoch {epoch}] Val EER: {eer:.4f}, Val Loss: {val_loss:.4f}")

        with open(os.path.join(args.save_path, args.model_name, "log.txt"), "a") as f:
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
def validate(model, val_loader, device, args, processor, wav2vec_model):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    ce_loss_fn = nn.CrossEntropyLoss()
    score_list = []
    label_list = []
    
    with torch.no_grad():
        for (wave, label, wav2vec_ft) in tqdm(val_loader, desc="Validation"):
            label = label.to(device)
            wave = wave.to(device)
            inputs = processor(wave, sampling_rate=16000, return_tensors="pt", padding=True)
            input_values = inputs.input_values.to(device)
            attention_mask = inputs.attention_mask.to(device)
            wav2vec_hidden_state = wav2vec_model(input_values, attention_mask=attention_mask)      
            logits, _ = model(wav2vec_hidden_state.hidden_states)
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
