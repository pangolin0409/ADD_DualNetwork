from tqdm import tqdm
import torch
import random
from utils.AudioUtils import extract_mel_spectrogram, extract_pitch
import pandas as pd
import utils.eval_metrics as em
import torch.nn.functional as F
import numpy as np
from utils.loss import info_nce_loss

"""
    多模態對比學習：
    embeddings_dict: { 
        "modality1": Tensor(shape [batch, dim]),
        "modality2": Tensor(shape [batch, dim]),
        ...
    }
    做兩兩 InfoNCE 後平均
"""
def multi_modal_alignment_loss(embeddings_dict, temperature=0.07):
    modalities = list(embeddings_dict.keys())
    total_loss = 0.0
    count = 0

    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            emb_i = embeddings_dict[modalities[i]]  # shape: [batch, seq_len, dim]
            emb_j = embeddings_dict[modalities[j]]  # shape: [batch, seq_len, dim]

            # 例如用 mean pooling => [batch, dim]
            emb_i_2d = emb_i.mean(dim=1)
            emb_j_2d = emb_j.mean(dim=1)

            # 單對 InfoNCE
            pair_loss = info_nce_loss(emb_i_2d, emb_j_2d, temperature)
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
    alpha_align,
    is_train_connector=False,
    is_train_experts=False,
    is_train_classifier=False
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

    if is_train_experts:
        for expert in experts.values():
            expert.train()
            for param in expert.parameters():
                param.requires_grad = True
    else:
        for expert in experts.values():
            expert.eval()
            for param in expert.parameters():
                param.requires_grad = False

    if is_train_classifier:
        classifier.train()
        for param in classifier.parameters():
            param.requires_grad = True
    else:
        classifier.eval()
        for param in classifier.parameters():
            param.requires_grad = False

    # -----------------------------
    # 2) 設定 Connector 可否訓練
    # -----------------------------
    if is_train_connector:
        for connector in connectors.values():
            connector.train()
            for param in connector.parameters():
                param.requires_grad = True
    else:
        for connector in connectors.values():
            connector.eval()
            for param in connector.parameters():
                param.requires_grad = False

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
                preprocessed_output = get_modality_embedding(modality, audio, encoders, wav2vec_extractor, device)
                # preprocessed_output = preprocessors[modality](output)

            # Connector (可訓練)
            connector_output = connectors[modality](preprocessed_output)
            connector_outputs[modality] = connector_output  # 留待計算對齊損失

            # 不用 torch.no_grad，但仍然凍結 Expert 和 Classifier
            expert_output = experts[modality](connector_output)
            task_prompt = None
            classifier_output = classifier(embeddings=expert_output, task_prompt=task_prompt)
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
            preprocessed_output = get_modality_embedding(modality, audio, encoders, wav2vec_extractor, device)
            # preprocessed_output = preprocessors[modality](output)
            connector_output = connectors[modality](preprocessed_output)
            expert_output = experts[modality](connector_output)
            task_prompt = None
            classifier_output = classifier(expert_output, task_prompt)

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
    
    real_indices = random.sample(real_indices, 100)

    # 設定下採樣目標
    spoof_indices = random.sample(spoof_indices, 300)
    
    print(f'Real samples: {len(real_indices)}, Spoof samples: {len(spoof_indices)}')
    return real_indices, spoof_indices

"""
    這個函數用來取得模態的 embedding。
    如果是 SpeechSplit 模態，會將 32 個 mel spectrogram 丟進 encoder。
    如果是 Wav2Vec2 模態，會將 waveform 丟進 wav2vec2 encoder。
"""
def get_modality_embedding(modality, audio, encoders, wav2vec_extractor, device):
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

    return output