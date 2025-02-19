from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import utils.eval_metrics as em
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from models.rawnet3.RawNet3 import RawNet3
from models.rawnet3.RawNetBasicBlock import Bottle2neck
from models.speechsplit.model import Generator_3
from models.speechsplit.hparams import hparams
from models.classifier.AudioClassifier import SimpleMLPClassifier
from torch import nn, optim
from load_datasets import load_datasets
from config import init
from utils.AudioUtils import extract_mel_spectrogram, extract_pitch
import numpy as np
def get_components(args):
    device = args.device

    # -------------------
    # 1) Load encoders
    # -------------------
    encoders = {
        "RawNet3": RawNet3(
            Bottle2neck,
            model_scale=8,
            context=True,
            summed=True,
            encoder_type="ECA",
            nOut=256,
            out_bn=False,
            sinc_stride=10,
            log_sinc=True,
            norm_sinc="mean",
            grad_mult=1.0,
        ).to(device),
        "Wav2Vec2": Wav2Vec2Model.from_pretrained(args.wav2vec_path).to(device),
        "SpeechSplit_timbre_content": Generator_3(hparams).to(device),
        "SpeechSplit_pitch": Generator_3(hparams).to(device),
        "SpeechSplit_rhythm": Generator_3(hparams).to(device),
    }

    # Load RawNet3 weights
    encoders['RawNet3'].load_state_dict(
        torch.load(args.rawnet3_path, map_location=device, weights_only=True)["model"]
    )

    # Load SpeechSplit weights
    checkpoint = torch.load(args.speechsplit_path, map_location=device, weights_only=True)
    encoders['SpeechSplit_timbre_content'].load_state_dict(checkpoint['model'])
    encoders['SpeechSplit_pitch'].load_state_dict(checkpoint['model'])
    encoders['SpeechSplit_rhythm'].load_state_dict(checkpoint['model'])

    for encoder in encoders.values():
        encoder.requires_grad_(False)  # Freeze all encoders by default

    return encoders


# ---------------------------
# 2) 單一 encoder baseline
# ---------------------------
def single_encoder_baseline(args, encoder, encoder_name, input_dim):
    """
    執行 audio->Encoder->MLP->二元分類 的基準實驗:
      - encoder: 已經載入好的 encoder (並且會凍結)
      - input_dim: encoder 輸出 feature 的維度
    """

    device = args.device
    # 載入資料集 (train & val)
    train_dl, val_dl = load_datasets(args, target_fake_ratio=2, test=True)

    # 1) Freeze encoder
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # 2) 初始化 MLP
    mlp_classifier = SimpleMLPClassifier(
        input_dim=input_dim,
        hidden_dim=128,
        num_classes=2
    ).to(device)

    
    # Load Wav2Vec2 feature extractor
    wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)

    # 3) 設定 optimizer & loss
    optimizer = optim.Adam(mlp_classifier.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # ---------------------------
    # 4) 開始訓練迴圈
    # ---------------------------
    for epoch in range(args.epochs):
        mlp_classifier.train()
        total_loss = 0.0
        total_samples = 0
        features_list = []
        labels_list = []
        for batch in tqdm(train_dl, desc=f"[{encoder_name}] Train Epoch {epoch+1}"):
            audio, labels = batch
            audio, labels = audio.to(device), labels.to(device)

            # === forward ===
            with torch.no_grad():
                # 只算 encoder
                features = get_modality_embedding(modality=encoder_name, encoder=encoder, audio=audio, wav2vec_extractor=wav2vec_extractor, device=device)

            # 只 train MLP
            if features.dim()==3:
                features = features.mean(dim=1)
            features_list.append(features.cpu())  # 確保 feature 在 CPU 上
            labels_list.append(labels.cpu())
        #     logits = mlp_classifier(features)
        #     loss = criterion(logits, labels)

        #     # === backward ===
        #     optimizer.zero_grad()
        #     loss.backward()
        #     optimizer.step()
        #     print(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")

        #     # 統計
        #     bs = labels.size(0)
        #     total_loss += loss.item() * bs
        #     total_samples += bs

        # avg_loss = total_loss / total_samples
        # print(f"[{encoder_name}] Epoch {epoch+1}/{args.epochs} - Train Loss: {avg_loss:.4f}")
        # # write to log
        # with open("test_encoder_log.csv", "a") as f:
        #     f.write(f"{encoder_name},{epoch},{avg_loss:.4f},,,\n")

        # ---------------------------
        # Validation
        # ---------------------------
        # val_loss, val_eer = validate_single_encoder(val_dl, encoder, mlp_classifier, encoder_name, args)
        # print(f"[{encoder_name}] Epoch {epoch+1} - Val Loss: {val_loss:.4f}, EER: {val_eer:.4f}")
        # # write to log
        # with open("test_encoder_log.csv", "a") as f:
        #     f.write(f"{encoder_name},{epoch},{avg_loss:.4f},{val_loss:.4f},{val_eer:.4f}\n")
        
        # if epoch == args.epochs - 1:  # 只在最後一個 epoch 可視化
        visualize_features(features_list, labels_list, title=f"Encoder: {encoder_name}")

        
        scheduler.step()


def validate_single_encoder(val_dl, encoder, mlp_classifier, encoder_name, args):
    """
    Validation 階段: audio->encoder->MLP->(output)
    計算 CE Loss & EER (或其它metrics)
    """
    device = args.device
    
    # Load Wav2Vec2 feature extractor
    wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)

    encoder.eval()
    mlp_classifier.eval()

    total_loss = 0.0
    total_samples = 0
    criterion = nn.CrossEntropyLoss()

    # 收集預測分數與標籤 (計算 EER 用)
    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dl:
            audio, labels = batch
            audio, labels = audio.to(device), labels.to(device)

            # encoder
            features = get_modality_embedding(modality=encoder_name, encoder=encoder, audio=audio, wav2vec_extractor=wav2vec_extractor, device=device)
            # MLP
            if features.dim()==3:
                features = features.mean(dim=1)
            logits = mlp_classifier(features)
            loss = criterion(logits, labels)

            # Cross Entropy
            bs = labels.size(0)
            total_loss += loss.item() * bs
            total_samples += bs

            # 從 logits 取出「假聲機率」或 logit分數 來計算 EER
            # 假設 logits[:,1] = 假聲的分數 (二分類第2類)
            fake_scores = torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy()
            all_scores.extend(fake_scores)
            all_labels.extend(labels.cpu().numpy())

      # **轉成 NumPy 陣列**
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    print("Fake scores distribution:",
      f"min={all_scores.min():.4f},",
      f"mean={all_scores.mean():.4f},",
      f"max={all_scores.max():.4f}")
    # label=0 => target, label=1 => non-target
    target_scores = all_scores[all_labels == 1]
    nontarget_scores = all_scores[all_labels == 0]
    avg_loss = total_loss / total_samples
    eer, frr, far, thresholds = em.compute_eer(target_scores, nontarget_scores)  # 你實作的 compute_eer

    return avg_loss, eer


"""
    這個函數用來取得模態的 embedding。
    如果是 SpeechSplit 模態，會將 32 個 mel spectrogram 丟進 encoder。
    如果是 Wav2Vec2 模態，會將 waveform 丟進 wav2vec2 encoder。
"""
def get_modality_embedding(modality, audio, encoder, wav2vec_extractor, device):
    if "SpeechSplit" in modality:
        mel_list = []
        for i in range(audio.size(0)):  # 0~31
            single_waveform = audio[i]
            mel = extract_mel_spectrogram(single_waveform, sr=16000, n_mels=80, max_len_pad=192)
            mel_list.append(mel)
        # 最終把 32 個 mel spectrogram cat 起來 => [32, n_mels, T]
        mel_spectrogram = torch.cat(mel_list, dim=0).to(device)
        if modality == "SpeechSplit_timbre_content":
            output = encoder.encoder_2(mel_spectrogram, None)
        elif modality == "SpeechSplit_pitch":
            pitch_list  =[]
            for i in range(audio.size(0)):
                single_waveform = audio[i]
                pitch = extract_pitch(single_waveform)
                pitch_list.append(pitch)
            # 最終把 32 個 pitch cat 起來 => [32, 64, T]
            f0_trg = torch.cat(pitch_list, dim=0).to(device)
            f0_trg = torch.cat((mel_spectrogram, f0_trg), dim=1)
            _, output = encoder.encoder_1(f0_trg)
        elif modality == "SpeechSplit_rhythm":
            output = encoder.rhythm(mel_spectrogram.transpose(1, 2))
    elif "Wav2Vec2" in modality:
        input_values = wav2vec_extractor(
            audio, 
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(device)
        input_values = input_values.squeeze(0)

        # 不需要再 squeeze(1)，因為它現在是 2D: [batch, length]
        # 直接丟給 encoder (Wav2Vec2Model)
        wav2vec_out = encoder(input_values)
        output = wav2vec_out.last_hidden_state
        # 通常 => [batch, time, hidden_dim]
    else:
        output = encoder(audio)

    return output

def visualize_features(features, labels, title="Feature Visualization"):
    print("Visualizing features...")
    
    if isinstance(features, list):
        features = torch.cat(features, dim=0)  # list of tensors -> single tensor
    
    if isinstance(labels, list):
        labels = torch.cat(labels, dim=0)  # list of tensors -> single tensor
    
    features = features.cpu().numpy()
    labels = labels.cpu().numpy()

    n_samples = features.shape[0]
    perplexity_value = min(30, n_samples - 1)  # 避免 TSNE 的 perplexity 超過樣本數

    tsne = TSNE(n_components=2, perplexity=perplexity_value, random_state=42)
    reduced_features = tsne.fit_transform(features)

    # 根據 labels 將 0 和 1 的索引分開
    idx_0 = labels == 0
    idx_1 = labels == 1

    plt.figure(figsize=(8, 6))
    plt.scatter(reduced_features[idx_0, 0], reduced_features[idx_0, 1], c='blue', label="Class 0", alpha=0.7)
    plt.scatter(reduced_features[idx_1, 0], reduced_features[idx_1, 1], c='red', label="Class 1", alpha=0.7)
    
    plt.colorbar()
    plt.title(f"{title}, perplexity={perplexity_value}")
    plt.legend()  # 加上圖例，標示 class 0 / 1
    plt.show()
    plt.close()



# ---------------------------
# 3) 簡易 main
# ---------------------------
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    encoders = get_components(args)
  
    encoder_configs = [
        ("SpeechSplit_pitch", encoders['SpeechSplit_pitch'], 64),
        ("SpeechSplit_timbre_content", encoders['SpeechSplit_timbre_content'], 2), 
        ("RawNet3", encoders['RawNet3'], 256),
        ("Wav2Vec2", encoders['Wav2Vec2'], 1024),
        ("SpeechSplit_rhythm", encoders['SpeechSplit_rhythm'], 2)
    ]

    # 3) 逐一檢驗每個 encoder 的 baseline 表現
    for encoder_name, encoder_obj, feat_dim in encoder_configs:
        print(f"\n===== Baseline test for {encoder_name} =====")
        single_encoder_baseline(args, encoder_obj, encoder_name, feat_dim)


if __name__ == '__main__':
    args = init()
    main(args)