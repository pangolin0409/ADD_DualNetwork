from tqdm import tqdm
import torch
import random
from utils.AudioUtils import extract_mel_spectrogram, extract_pitch
import pandas as pd
import utils.eval_metrics as em
import torch.nn.functional as F
import numpy as np
from utils.loss import multi_modal_alignment_loss, length_loss, contrastive_loss
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from models.rawnet3.RawNet3 import RawNet3
from models.rawnet3.RawNetBasicBlock import Bottle2neck
from models.speechsplit.model import Generator_3
from models.speechsplit.hparams import hparams
from models.experts.ExpertMLP import ExpertTDNN
from models.connectors.Qformer import MLPConnector
from models.classifier.AudioClassifier import SimpleMLPClassifier


def get_components(args, modality_input_dims):
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

    # Load Wav2Vec2 feature extractor
    wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)

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

    # -------------------
    # 2) 初始化 connectors
    # -------------------
    connectors = {}
    for modality in encoders.keys():
        input_dim = modality_input_dims[modality]
        connector = MLPConnector(
            input_dim=input_dim, 
            output_dim=args.query_dim
        ).to(device)

        if not args.is_train_connectors:
            # 載入預訓練好的 connector
            checkpoint_path = f"./{args.model_name}/pretrained_models/connectors/best_{modality}_connector.pth"
            connector.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        
        connectors[modality] = connector

    # -------------------
    # 3) 初始化專家 (TDNN+Attention)
    # -------------------
    experts = {}
    for modality in encoders.keys():
        # 假設 ExpertTDNN(input_dim=args.query_dim, tdnn_hidden_dim=args.tdnn_hidden_dim)
        experts[modality] = ExpertTDNN(
            input_dim=args.query_dim, 
            hidden_dim=args.tdnn_hidden_dim,
            device=device
        ).to(device)

    # -------------------
    # 4) 初始化單一 MLP 分類器
    # -------------------
    # 假設輸入維度 = tdnn_hidden_dim, hidden_dim=128, output_dim=2
    classifier = SimpleMLPClassifier(
        input_dim=args.tdnn_hidden_dim,
        hidden_dim=128,
        num_classes=2
    ).to(device)

    return encoders, wav2vec_extractor, connectors, experts, classifier


def train(
    train_dataloader,
    encoders,
    wav2vec_extractor,
    connectors,
    experts,
    classifier,
    negative_queue,  # 可以是 None 或 NegativeQueue 實例
    optimizer,
    criterion,         # CrossEntropyLoss 或其他
    device,
    # 是否訓練各模組
    is_train_connector=False,
    is_train_expert=False,
    is_train_classifier=False,
    # Loss 權重
    alpha_align=0.0,     # 多模態對齊(InfoNCE)的權重 (第一階段用)
    alpha_contrast=0.0,  # 對比損失權重
    alpha_length=0.0,    # length_loss 權重
    margin=4.0,          # length_loss 用
    temperature=0.07     # contrastive_loss 溫度
):
    """
    單一函式，同時可支援第一階段/第二階段的訓練邏輯：
    - 若 is_train_connector=True => 訓練 Connector (類似第一階段)
      同時可搭配 alpha_align > 0 做多模態對齊
    - 若 is_train_expert=True, is_train_classifier=True => 訓練 Expert+Classifier (第二階段)
      可同時在每個模態上計算 length_loss, contrastive_loss
    """

    # 1) 設定 Encoders, Connectors, Experts, Classifier 之 requires_grad
    # ----------------------------------------------------------------
    # Encoders 一般都凍結
    for encoder in encoders.values():
        encoder.eval()
        for p in encoder.parameters():
            p.requires_grad = False

    # Connector
    for connector in connectors.values():
        if is_train_connector:
            connector.train()
            for p in connector.parameters():
                p.requires_grad = True
        else:
            connector.eval()
            for p in connector.parameters():
                p.requires_grad = False

    # Experts
    for expert in experts.values():
        if is_train_expert:
            expert.train()
            for p in expert.parameters():
                p.requires_grad = True
        else:
            expert.eval()
            for p in expert.parameters():
                p.requires_grad = False

    # Classifier
    if is_train_classifier:
        classifier.train()
        for p in classifier.parameters():
            p.requires_grad = True
    else:
        classifier.eval()
        for p in classifier.parameters():
            p.requires_grad = False

    # 2) 訓練循環
    total_loss_epoch = 0.0
    total_samples_epoch = 0

    for batch in tqdm(train_dataloader, desc="Train Epoch"):
        audio, labels = batch
        audio, labels = audio.to(device), labels.to(device)

        # 保存每個模態的:
        # (a) classifier CE loss
        # (b) length loss
        # (c) contrastive loss
        ce_losses = []
        length_losses = []
        contrastive_losses = []

        # 為多模態對齊 (connector align) 做準備:
        # 收集 {modality: connector_output} 用於 multi_modal_alignment_loss
        connector_outputs_dict = {}

        # -----------------------------
        # A) 逐模態 forward + 計算 loss
        # -----------------------------
        for modality in encoders.keys():
            # 1) Encoder (凍結) + wav2vec_extractor (凍結)
            with torch.no_grad():
                preprocessed_output = get_modality_embedding(
                    modality, audio, encoders, wav2vec_extractor, device
                )

            # 2) Connector (看 is_train_connector)
            connector_output = connectors[modality](preprocessed_output)
            # 收集到 dict (若要做 multi_modal_alignment_loss)
            connector_outputs_dict[modality] = connector_output

            # 3) Expert (TDNN+Attention) - 可能訓練
            expert_out = experts[modality](connector_output)  # shape (B, feat_dim)

            # 4) Classifier (可能訓練)
            logits = classifier(expert_out)  # shape (B, 2)
            loss_ce = criterion(logits, labels)
            ce_losses.append(loss_ce)

            # 5) length_loss & contrastive_loss (by 模態)
            # ------------------------------------------------------------
            #   length_loss: 針對本模態 expert_out
            #   contrastive_loss: 
            #       需要 negatives 來自 negative_queue 或其他
            # ------------------------------------------------------------
            if alpha_length > 0.0:
                l_len = length_loss(expert_out, labels, margin=margin)
                length_losses.append(l_len)
            else:
                l_len = torch.zeros(1, device=device)

            if alpha_contrast > 0.0 and negative_queue is not None:
                # negatives from queue
                negatives = negative_queue.get_negatives()

                # 這裡示範 simplest: features_q = features_k = expert_out
                # 你也可以依照真正對比設計(q, k 不同 sample / real vs fake pairing)
                # by 模態 -> enqueue "fake" from this modality
                l_con = contrastive_loss(
                    features_q=expert_out, 
                    features_k=expert_out, 
                    negatives=negatives, 
                    temperature=temperature
                )
                contrastive_losses.append(l_con)

                # enqueue 目前 batch 中 假聲embedding
                negative_queue.dequeue_and_enqueue(expert_out, labels)
            else:
                l_con = torch.zeros(1, device=device)

            # 可視需要印 debug
            # print(f"{modality} - CE={loss_ce.item():.4f}, L_len={l_len.item():.4f}, L_con={l_con.item():.4f}")

        # -----------------------------
        # B) 多模態對齊 loss (第一階段會用)
        # -----------------------------
        if alpha_align > 0.0:
            align_loss = multi_modal_alignment_loss(connector_outputs_dict)
        else:
            align_loss = torch.zeros(1, device=device)

        # -----------------------------
        # C) 合併所有 loss
        # -----------------------------
        sum_ce = torch.stack(ce_losses).mean() if len(ce_losses) > 0 else torch.zeros(1, device=device)
        sum_len = torch.stack(length_losses).mean() if len(length_losses) > 0 else torch.zeros(1, device=device)
        sum_con = torch.stack(contrastive_losses).mean() if len(contrastive_losses) > 0 else torch.zeros(1, device=device)

        # 檢查 loss 是否為 nan
        if torch.isnan(sum_ce):
            print("Warning: NaN detected in sum_ce, setting it to zero.")
            sum_ce = torch.zeros(1, device=device)
        if torch.isnan(sum_len):
            print("Warning: NaN detected in sum_len, setting it to zero.")
            sum_len = torch.zeros(1, device=device)
        if torch.isnan(sum_con):
            print("Warning: NaN detected in sum_con, setting it to zero.")
            sum_con = torch.zeros(1, device=device)
        if torch.isnan(align_loss):
            print("Warning: NaN detected in align_loss, setting it to zero.")
            align_loss = torch.zeros(1, device=device)

        total_loss = sum_ce + alpha_length * sum_len + alpha_contrast * sum_con + alpha_align * align_loss

        # -----------------------------
        # D) backward + update
        # -----------------------------
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # -----------------------------
        # E) 統計
        # -----------------------------
        bs = labels.size(0)
        total_loss_epoch += total_loss.item() * bs
        total_samples_epoch += bs

    avg_loss_epoch = total_loss_epoch / total_samples_epoch
    print(f"[train_epoch] Avg Loss = {avg_loss_epoch:.4f}")

    return avg_loss_epoch

@torch.no_grad()
def validation_phase(
    valid_dataloader,
    encoders,
    wav2vec_extractor,
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