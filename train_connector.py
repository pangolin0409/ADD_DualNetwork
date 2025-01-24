import os 
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
import random
import argparse
from data.dataloader import RawAudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from models.rawnet3.RawNet3 import RawNet3
from models.rawnet3.RawNetBasicBlock import Bottle2neck
from models.speechsplit.model import Generator_3
from models.speechsplit.hparams import hparams
from utils.AudioUtils import extract_mel_spectrogram, extract_pitch
from models.experts.ExpertMLP import ExpertMLP
from models.connectors.Qformer import QFormerConnector
from models.classifier.AudioClassifier import PromptedLLMClassifier
from utils.Projection import Preprocessor
import pandas as pd
import utils.eval_metrics as em
import torch.nn.functional as F
import numpy as np
from utils.loss import info_nce_loss

def init():
    parser = argparse.ArgumentParser(description="Train Q-Former Connector for Audio Deepfake Detection")
    # 模型參數
    parser.add_argument("--input_dim", type=int, default=1024, help="Input feature dimension from RawNet2/Wav2Vec2")
    parser.add_argument("--query_dim", type=int, default=768, help="Projected feature dimension")
    parser.add_argument("--num_queries", type=int, default=4, help="Number of learnable queries")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers")

    # 訓練參數
    parser.add_argument('-model_name', type=str, default='DAC')
    parser.add_argument('-nb_samp', type=int, default=64600)
    parser.add_argument('-alpha', type=float, default=1.0)
    parser.add_argument('-beta', type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument('-nb_worker', type=int, default=8)

    # 預訓練模型路徑
    parser.add_argument("--rawnet3_path", type=str, default='./pretrained_models/rawnet3/rawnet3_weights.pt', help="Path to the RawNet3 model")
    parser.add_argument("--speechsplit_path", type=str, default='./pretrained_models/speechsplit/speechsplit_weights.ckpt', help="Path to the SpeechSplit model")
    parser.add_argument("--wav2vec_path", type=str, default='./pretrained_models/wav2vec2', help="Path to the wav2vec model")
    parser.add_argument("--llm_model_name", type=str, default="gpt2", help="LLM model name for ExpertMLP and Classifier")
    parser.add_argument("--llm_model_dir", type=str, default="./pretrained_models/gpt2_local", help="Path to the LLM model directory")
    args = parser.parse_args()
    return args


def initialize_connectors(modalities, target_query_dim, args):
    connectors = {}
    for modality in modalities:
        connectors[modality] = QFormerConnector(
            input_dim=target_query_dim
        ).to(args.device)
    return connectors


def multi_modal_alignment_loss(embeddings_dict, temperature=0.07):
    """
    多模態對比學習：
    embeddings_dict: { 
        "modality1": Tensor(shape [batch, dim]),
        "modality2": Tensor(shape [batch, dim]),
        ...
    }
    做兩兩 InfoNCE 後平均
    """
    modalities = list(embeddings_dict.keys())
    total_loss = 0.0
    count = 0

    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            emb_i = embeddings_dict[modalities[i]]
            emb_j = embeddings_dict[modalities[j]]

            # 單對 InfoNCE
            pair_loss = info_nce_loss(emb_i, emb_j, temperature)
            total_loss += pair_loss
            count += 1

    if count > 0:
        total_loss /= count
    return total_loss

def train_phase_1(
    train_dataloader,
    encoders,
    wav2vec_extractor,
    preprocessors,
    connectors,
    experts,
    classifier,
    optimizer,
    criterion,
    device,
    alpha_align=1.0
):
    """
    只更新 Connector 的權重，其餘 (Encoder, Preprocessor, Expert, Classifier) 都凍結。
    並在同一個 batch 內，同時計算各模態的分類損失和多模態對齊 (MSE) 損失。
    alpha_align 用於調整對齊損失的權重。
    """

    # -----------------------------
    # 1) 凍結 Encoder & Preprocessor & Expert & Classifier
    # -----------------------------
    for encoder in encoders.values():
        encoder.eval()
        for param in encoder.parameters():
            param.requires_grad = False

    for preprocessor in preprocessors.values():
        preprocessor.eval()
        for param in preprocessor.parameters():
            param.requires_grad = False

    for expert in experts.values():
        expert.eval()
        for param in expert.parameters():
            param.requires_grad = False

    classifier.eval()
    for param in classifier.parameters():
        param.requires_grad = False

    # -----------------------------
    # 2) 設定 Connector 可訓練
    # -----------------------------
    for connector in connectors.values():
        connector.train()
        for param in connector.parameters():
            param.requires_grad = True

    total_loss_epoch = 0.0
    total_samples_epoch = 0
    modality_losses = {modality: 0.0 for modality in encoders.keys()}

    # -----------------------------
    # 3) 開始訓練迴圈
    # -----------------------------
    for batch in tqdm(train_dataloader, desc="第一階段訓練"):
        audio, labels = batch
        audio, labels = audio.to(device), labels.to(device)

        # 用來累積各模態的分類損失，與 Connector 輸出（以便後面計算對齊損失）
        classification_losses = []
        connector_outputs = {}

        # -----------------------------
        # (A) 收集所有模態的輸出、分類損失
        # -----------------------------
        for modality in encoders.keys():
            with torch.no_grad():
                # 凍結的 Encoder & Preprocessor
                if "SpeechSplit" in modality:
                    mel_list = []
                    for i in range(audio.size(0)):  # 0~31
                        single_waveform = audio[i]
                        mel = extract_mel_spectrogram(single_waveform, sr=16000, n_mels=80, max_len_pad=192)
                        mel_list.append(mel)
                    # 最終把 32 個 mel spectrogram cat 起來 => [32, n_mels, T]
                    mel_spectrogram = torch.cat(mel_list, dim=0).to(device)
                    if modality == "SpeechSplit_timbre_content":
                        output = encoders[modality].encoder_2(mel_spectrogram, None)
                    elif modality == "SpeechSplit_pitch":
                        pitch_list  =[]
                        for i in range(audio.size(0)):
                            single_waveform = audio[i]
                            pitch = extract_pitch(single_waveform)
                            pitch_list.append(pitch)
                        # 最終把 32 個 pitch cat 起來 => [32, 64, T]
                        f0_trg = torch.cat(pitch_list, dim=0).to(device)
                        f0_trg = torch.cat((mel_spectrogram, f0_trg), dim=1)
                        _, output = encoders[modality].encoder_1(f0_trg)
                    elif modality == "SpeechSplit_rhythm":
                        output = encoders[modality].rhythm(mel_spectrogram.transpose(1, 2))
                elif "Wav2Vec2" in modality:
                    input_values = wav2vec_extractor(
                        audio, 
                        sampling_rate=16000,
                        return_tensors="pt"
                    ).input_values.to(device)
                    input_values = input_values.squeeze(0)

                    # 不需要再 squeeze(1)，因為它現在是 2D: [batch, length]
                    # 直接丟給 encoders[modality] (Wav2Vec2Model)
                    wav2vec_out = encoders[modality](input_values)
                    output = wav2vec_out.last_hidden_state
                    # 通常 => [batch, time, hidden_dim]
                else:
                    output = encoders[modality](audio)
                preprocessed_output = preprocessors[modality](output)
            # Connector (可訓練)
            connector_output = connectors[modality](preprocessed_output)
            connector_outputs[modality] = connector_output  # 留待計算對齊損失
            connector_output = connector_output.unsqueeze(1)  # -> [batch_size, 1, 768]
            expert_output = experts[modality](connector_output)
            classifier_output = classifier(expert_output)
            # 計算該模態的CE分類損失
            loss_ce = criterion(classifier_output, labels)
            classification_losses.append(loss_ce)

            # 累加該模態的 (CE x batch_size) 到 modality_losses，用於後面 epoch-end 的統計
            modality_losses[modality] += loss_ce.item() * len(labels)

        # -----------------------------
        # (B) 計算多模態對齊損失 + 分類損失，一次 backprop
        # -----------------------------
        # 先把各模態 CE Loss 做平均 (或可直接 sum)
        total_ce = sum(classification_losses) / len(classification_losses)

        # 計算對齊損失
        alignment_loss = multi_modal_alignment_loss(connector_outputs)
        # 透過 alpha_align 去調整對齊損失的權重
        total_loss = total_ce + alpha_align * alignment_loss

        # -----------------------------
        # (C) 反向傳播 & 更新
        # -----------------------------
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # -----------------------------
        # (D) 統計到 epoch 累加
        # -----------------------------
        total_samples_epoch += len(labels)
        total_loss_epoch += total_loss.item() * len(labels)

    avg_loss_epoch = total_loss_epoch / total_samples_epoch
    print(f"平均 Loss: {avg_loss_epoch:.4f}")

    return avg_loss_epoch

@torch.no_grad()
def validation_phase(
    valid_dataloader,
    encoders,
    wav2vec_extractor,
    preprocessors,
    connectors,
    experts,
    classifier,
    criterion,
    device
):
    """
    在驗證資料集上做前向推理，計算平均 Loss 及 EER。
    不會執行梯度更新。
    """

    # 切換到 eval 模式
    for encoder in encoders.values():
        encoder.eval()
    for preprocessor in preprocessors.values():
        preprocessor.eval()
    for connector in connectors.values():
        connector.eval()
    for expert in experts.values():
        expert.eval()
    classifier.eval()

    total_loss = 0.0
    total_samples = 0

    # 為了計算 EER，需要蒐集 scores 和 labels
    score_loader = {modality: [] for modality in encoders.keys()}
    label_loader = {modality: [] for modality in encoders.keys()}

    for batch in tqdm(valid_dataloader, desc="驗證階段"):
        audio, labels = batch
        audio, labels = audio.to(device), labels.to(device)

        for modality in encoders.keys():
            # 前向傳遞（encoders, preprocessors, connectors, experts, classifier）
            if "SpeechSplit" in modality:
                mel_list = []
                for i in range(audio.size(0)):  # 0~31
                    single_waveform = audio[i]
                    mel = extract_mel_spectrogram(single_waveform, sr=16000, n_mels=80, max_len_pad=192)
                    mel_list.append(mel)
                # 最終把 32 個 mel spectrogram cat 起來 => [32, n_mels, T]
                mel_spectrogram = torch.cat(mel_list, dim=0).to(device)
                if modality == "SpeechSplit_timbre_content":
                    output = encoders[modality].encoder_2(mel_spectrogram, None)
                elif modality == "SpeechSplit_pitch":
                    pitch_list  =[]
                    for i in range(audio.size(0)):
                        single_waveform = audio[i]
                        pitch = extract_pitch(single_waveform)
                        pitch_list.append(pitch)
                    # 最終把 32 個 pitch cat 起來 => [32, 64, T]
                    f0_trg = torch.cat(pitch_list, dim=0).to(device)
                    f0_trg = torch.cat((mel_spectrogram, f0_trg), dim=1)
                    _, output = encoders[modality].encoder_1(f0_trg)
                elif modality == "SpeechSplit_rhythm":
                    output = encoders[modality].rhythm(mel_spectrogram.transpose(1, 2))
            elif "Wav2Vec2" in modality:
                input_values = wav2vec_extractor(
                    audio, 
                    sampling_rate=16000,
                    return_tensors="pt"
                ).input_values.to(device)
                input_values = input_values.squeeze(0)

                # 不需要再 squeeze(1)，因為它現在是 2D: [batch, length]
                # 直接丟給 encoders[modality] (Wav2Vec2Model)
                wav2vec_out = encoders[modality](input_values)
                output = wav2vec_out.last_hidden_state
                # 通常 => [batch, time, hidden_dim]
            else:
                output = encoders[modality](audio)
            preprocessed_output = preprocessors[modality](output)
            connector_output = connectors[modality](preprocessed_output)
            expert_output = experts[modality](connector_output)
            classifier_output = classifier(expert_output)

            # 計算Loss
            loss = criterion(classifier_output, labels)
            total_loss += loss.item() * len(labels)
            total_samples += len(labels)

            # 記錄 EER 所需
            # 例如我們假設：label=0 => bonafide (真), label=1 => spoof (假)
            scores = F.softmax(classifier_output, dim=1)[:, 0]  # 取 class=0 的機率
            score_loader[modality].extend(scores.detach().cpu().numpy())
            label_loader[modality].extend(labels.detach().cpu().numpy())

    # 統計整體平均 Loss
    avg_loss = total_loss / total_samples if total_samples > 0 else 0.0

    # 計算 EER
    # 這裡示範「針對每個模態」都計算 EER，最後可視需求選擇平均、或只回傳其中一個
    eer_dict = {}
    for modality in encoders.keys():
        scores = np.array(score_loader[modality])
        labels_np = np.array(label_loader[modality])

        # label=0 => target, label=1 => non-target
        target_scores = scores[labels_np == 0]
        nontarget_scores = scores[labels_np == 1]

        eer, _, _, threshold = em.compute_eer(target_scores, nontarget_scores)
        eer_dict[modality] = eer

    # 這裡可自行決定要返回哪個 EER
    # 範例：回傳五個模態的平均 EER
    if len(eer_dict) > 0:
        avg_eer = np.mean(list(eer_dict.values()))
    else:
        avg_eer = 0.0

    print("=== Validation Results ===")
    for modality, e in eer_dict.items():
        print(f"{modality} EER: {e:.4f}")
    print(f"Average EER: {avg_eer:.4f}, Avg Loss: {avg_loss:.4f}")

    # 回傳(平均Loss, 平均EER)，看你要怎麼接在外部
    return avg_loss, avg_eer

"""
    下採樣數據集，保留所有真樣本，並將假樣本的數量控制為真樣本的 target_fake_ratio 倍。
"""
def downsample_data(meta_path: str, dataset_name: str, target_fake_ratio: int = 2) -> tuple:
    print(f"Processing dataset: {dataset_name}")
    meta = pd.read_csv(meta_path)
    
    # 提取真實和假樣本索引
    real_indices = meta[meta['label'] == 'bonafide'].index.tolist()
    spoof_indices = meta[meta['label'] == 'spoof'].index.tolist()
    
    # 設定下採樣目標
    target_fake_count = len(real_indices) * target_fake_ratio
    if len(spoof_indices) > target_fake_count:
        spoof_indices = random.sample(spoof_indices, target_fake_count)
    
    print(f'Real samples: {len(real_indices)}, Spoof samples: {len(spoof_indices)}')
    return real_indices, spoof_indices


def main(args):
    device = args.device

    # 定義每個模態的輸入維度
    modality_input_dims = {
        "RawNet3": 256,
        "Wav2Vec2": 1024,
        "SpeechSplit_timbre_content": 2,
        "SpeechSplit_pitch": 64,
        "SpeechSplit_rhythm": 2
    }

    # 定義目標維度（連接器的輸入維度）
    target_query_dim = args.query_dim  # 例如 512

    # 初始化 datasets 和 dataloaders
    training_sets = {
        "DFADD": RawAudio(
            path_to_database='../datasets/DFADD',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='train'
        ),
        "CodecFake": RawAudio(
            path_to_database='../datasets/CodecFake',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='train'
        ),
        "ASVspoof2021_DF": RawAudio(
            path_to_database='../datasets/ASVspoof2021_DF',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='train'
        )
    }

    training_set_list = []
    for name, training_set in training_sets.items():
        real_indices, spoof_indices = downsample_data(meta_path=f'../datasets/{name}/train/meta.csv', dataset_name=name, target_fake_ratio=2)
        real_subset = Subset(training_set, real_indices)
        spoof_subset = Subset(training_set, spoof_indices)
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        training_set_list.append(adjusted_set)

    final_training_set = ConcatDataset(training_set_list)
    train_dataloader = DataLoader(final_training_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.nb_worker)

    # 初始化 datasets 和 dataloaders
    validation_sets = {
        "DFADD": RawAudio(
            path_to_database='../datasets/DFADD',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='validation'
        ),
        "CodecFake": RawAudio(
            path_to_database='../datasets/CodecFake',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='validation'
        ),
        "ASVspoof2021_DF": RawAudio(
            path_to_database='../datasets/ASVspoof2021_DF',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='validation'
        )
    }

    validation_set_list = []
    for name, training_set in validation_sets.items():
        real_indices, spoof_indices = downsample_data(meta_path=f'../datasets/{name}/train/meta.csv', dataset_name=name, target_fake_ratio=2)
        real_subset = Subset(training_set, real_indices)
        spoof_subset = Subset(training_set, spoof_indices)
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        validation_set_list.append(adjusted_set)

    final_validation_set = ConcatDataset(validation_set_list)
    validation_dataloader = DataLoader(final_validation_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.nb_worker)

    # Load encoders
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
        "Wav2Vec2": Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",
            cache_dir=args.wav2vec_path
        ).to(device),
        "SpeechSplit_timbre_content": Generator_3(hparams).to(device),
        "SpeechSplit_pitch": Generator_3(hparams).to(device),
        "SpeechSplit_rhythm": Generator_3(hparams).to(device),
    }

    # Load feature extractor
    wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-xls-r-300m", cache_dir=args.wav2vec_path
    )
    
    # Load RawNet3 weights
    encoders['RawNet3'].load_state_dict(
        torch.load(
            args.rawnet3_path,
            map_location=device
        )["model"]
    )

    # Load SpeechSplit weights
    checkpoint = torch.load(args.speechsplit_path, map_location=device)
    encoders['SpeechSplit_timbre_content'].load_state_dict(checkpoint['model'])
    encoders['SpeechSplit_pitch'].load_state_dict(checkpoint['model'])
    encoders['SpeechSplit_rhythm'].load_state_dict(checkpoint['model'])

    for encoder in encoders.values():
        encoder.requires_grad_(False)  # Freeze all encoders by default

    # 初始化預處理層
    from utils.Projection import Preprocessor
    preprocessors = {
        modality: Preprocessor(modality, modality_input_dims[modality], target_query_dim)
        for modality in encoders.keys()
    }
    preprocessors = {k: v.to(device) for k, v in preprocessors.items()}

    # 初始化連接器
    connectors = initialize_connectors(encoders.keys(), target_query_dim, args)

    # 初始化專家
    experts = {
        modality: ExpertMLP(
            model_name=args.llm_model_name,
            model_dir=args.llm_model_dir
        ).to(device)
        for modality in encoders.keys()
    }

    # 初始化單一分類器
    classifier = PromptedLLMClassifier(
        llm_name=args.llm_model_name,
        model_dir=args.llm_model_dir,
        prompt  = 'You are an audio deepfake detector. You are given an audio embedding to determine if it is real or fake.',
    ).to(device)

    # 僅訓練連接器
    optimizer = optim.Adam(
        [param for connector in connectors.values() for param in connector.parameters()],
        lr=args.lr
    )
    criterion = nn.CrossEntropyLoss()

    # 準備儲存目錄
    checkpoint_dir = os.path.join(args.model_name, "pretrained_models/connectors")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eer = float('inf')

    # 訓練迴圈 (加上 alpha_align 用來控制對齊損失權重)
    for epoch in range(args.epochs):
        train_loss = train_phase_1(
            train_dataloader,
            encoders,
            wav2vec_extractor,
            preprocessors,
            connectors,
            experts,
            classifier,
            optimizer,
            criterion,
            device,
            alpha_align=1.0  # 調大或調小，用來控制對齊損失的重要性
        )
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {train_loss:.4f}")

        # 在validation_set上做驗證
        val_loss, val_eer = validation_phase(
            validation_dataloader,
            encoders,
            wav2vec_extractor,
            preprocessors,
            connectors,
            experts,
            classifier,
            criterion,
            device
        )
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation EER: {val_eer:.4f}")

        # 每個 epoch 寫一次紀錄
        with open("training_log.csv", "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}, {val_eer:.4f}\n")

        
        # 保存檢查點
        for modality, connector in connectors.items():
            torch.save(
                connector.state_dict(),
                os.path.join(checkpoint_dir, f"{modality}_connector_epoch_{epoch}.pth")
            )

        # 檢查是否是最佳 EER
        if val_eer < best_eer:
            best_eer = val_eer
            for modality, connector in connectors.items():
                torch.save(
                    connector.state_dict(),
                    os.path.join(checkpoint_dir, f"best_{modality}_connector_epoch_{epoch}.pth")
                )

    print("Training complete.")


if __name__ == '__main__':
    args = init()
    main(args)
