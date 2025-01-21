import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
import argparse
from data.dataloader import RawAudio
import random
import pandas as pd
from transformers import Wav2Vec2Model,Wav2Vec2FeatureExtractor
from models.connectors.Qformer import QFormerConnector
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from models.rawnet3.RawNet3 import RawNet3
from models.rawnet3.RawNetBasicBlock import Bottle2neck
from models.speechsplit.model import Generator_3
from models.speechsplit.hparams import hparams
from utils.AudioUtils import extract_mel_spectrogram, extract_pitch
def init():
    parser = argparse.ArgumentParser(description="Train Q-Former Connector for Audio Deepfake Detection")
    # 模型參數
    parser.add_argument("--input_dim", type=int, default=1024, help="Input feature dimension from RawNet2/Wav2Vec2")
    parser.add_argument("--query_dim", type=int, default=512, help="Projected feature dimension")
    parser.add_argument("--num_queries", type=int, default=16, help="Number of learnable queries")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers")

    # 訓練參數.
    parser.add_argument('-model_name', type = str, default = 'DAC')
    parser.add_argument('-nb_samp', type = int, default = 64600)
    parser.add_argument('-alpha', type = float, default = 1.0)
    parser.add_argument('-beta', type = float, default = 0.0)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument('-nb_worker', type = int, default = 8)
    
    # 預訓練模型路徑
    parser.add_argument("--rawnet3_path", type=str, default='./pretrained_models/rawnet3/rawnet3_weights.pt', help="Path to the RawNet3 model")
    parser.add_argument("--speechsplit_path", type=str, default='./pretrained_models/speechsplit/speechsplit_weights.ckpt', help="Path to the SpeechSplit model")
    
    args = parser.parse_args()   
    return args

def contrastive_loss(emb1: torch.Tensor, emb2: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    sim = torch.nn.functional.cosine_similarity(emb1.unsqueeze(1), emb2.unsqueeze(0), dim=-1)  # 獲得樣本對相似度
    sim /= temperature
    pos_sim = torch.diag(sim)  # 正樣本相似度
    neg_sim = torch.cat([sim[~torch.eye(sim.size(0), dtype=bool)].view(sim.size(0), -1)], dim=-1)  # 負樣本相似度

    loss_pos = -torch.log(torch.exp(pos_sim) / (torch.exp(pos_sim) + torch.sum(torch.exp(neg_sim), dim=-1)))
    return loss_pos.mean()

def train_connector_with_rawnet(
    dataloader: DataLoader,
    rawnet2: nn.Module,
    connector_rawnet: nn.Module,
    connector_wav2vec: nn.Module,
    classifier: nn.Module,
    criterion: callable,
    contrastive_loss: callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    checkpoint_dir: str
) -> tuple:
    rawnet2.train()
    connector_rawnet.train()
    connector_wav2vec.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0

    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device)
    wav2vec_model.eval()

    for i, data_slice in enumerate(tqdm(dataloader)):
        audio, labels = data_slice
        audio, labels = audio.to(device), labels.to(device)

        # RawNet2 特徵提取
        rawnet_features = rawnet2(audio, is_test=True)

        # Wav2Vec2 特徵提取
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        input_values = input_values.squeeze(0)
        with torch.no_grad():
            wav2vec_features = wav2vec_model(input_values).last_hidden_state
        wav2vec_features = wav2vec_features.mean(dim=1)  # 變成 [32, 1024]

        # Connector 處理
        rawnet_proj = connector_rawnet(rawnet_features)
        wav2vec_proj = connector_wav2vec(wav2vec_features)

        # 特徵拼接
        combined_features = torch.cat([rawnet_proj.mean(dim=1), wav2vec_proj.mean(dim=1)], dim=1)

        # 分類器輸出
        outputs = classifier(combined_features)
        
        # 損失計算
        loss_classification = criterion(outputs, labels)
        loss_contrastive = contrastive_loss(rawnet_proj.mean(dim=1), wav2vec_proj.mean(dim=1))
        modality_loss = torch.mean(torch.abs(torch.nn.functional.cosine_similarity(rawnet_proj, wav2vec_proj, dim=-1)))

        loss = loss_classification + args.alpha * loss_contrastive + args.beta * modality_loss

        # 優化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 記錄損失與準確率
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    accuracy = 100.0 * correct / total

    # 儲存每個 epoch 的 Checkpoint
    checkpoint_path_rawnet = os.path.join(checkpoint_dir, "connector_rawnet.pth")
    checkpoint_path_wav2vec = os.path.join(checkpoint_dir, "connector_wav2vec.pth")
    checkpoint_path_rawnet2 = os.path.join(checkpoint_dir, "rawnet2.pth")
    torch.save(connector_rawnet.state_dict(), checkpoint_path_rawnet)
    torch.save(connector_wav2vec.state_dict(), checkpoint_path_wav2vec)
    torch.save(rawnet2.state_dict(), checkpoint_path_rawnet2)

    return total_loss / len(dataloader), accuracy


"""
    下採樣數據集，保留所有真樣本，並將假樣本的數量控制為真樣本的 target_fake_ratio 倍。
    
    Args:
        meta_path (str): 數據集的 meta 文件路徑
        dataset_name (str): 數據集名稱
        target_fake_ratio (int): 假樣本與真樣本的比例
    
    Returns:
        tuple: (真樣本索引, 下採樣的假樣本索引)
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


def train_phase_1(args, dataloader, encoders, connectors, optimizer, criterion, device):
    # Set encoders to evaluation mode
    for encoder in encoders.values():
        encoder.eval()
    for connector in connectors.values():
        connector.train()  # Train connectors only

    total_loss = 0.0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training Phase 1"):
        audio, labels = batch
        audio, labels = audio.to(device), labels.to(device)

        # Store connector outputs
        connector_outputs = []

        # Process each encoder and connector
        for modality, encoder in encoders.items():
            if "SpeechSplit" in modality:
                # Process SpeechSplit-specific encoders
                mel_spectrogram = extract_mel_spectrogram(audio)
                if modality == "SpeechSplit_timbre_content":
                    output = encoder.encoder_2(mel_spectrogram, None)
                elif modality == "SpeechSplit_pitch":
                    f0_trg = extract_pitch(audio)
                    f0_trg = torch.cat((mel_spectrogram, f0_trg), dim=1)
                    _, output = encoder.encoder_1(f0_trg)
                elif modality == "SpeechSplit_rhythm":
                    output = encoder.rhythm(mel_spectrogram.transpose(1, 2))
            else:
                # For RawNet3 and Wav2Vec2
                with torch.no_grad():
                    output = encoder(audio)

            # Pass through the corresponding connector
            connector_output = connectors[modality](output)
            connector_outputs.append(connector_output)

        # Combine connector outputs and compute loss
        connector_outputs_combined = torch.cat(connector_outputs, dim=1)
        loss = criterion(connector_outputs_combined, labels)

        # Backpropagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        total_samples += len(labels)

    avg_loss = total_loss / total_samples
    return avg_loss


def main(args):
    device = args.device
    checkpoint_dir = os.path.join(args.model_name, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Initialize datasets and dataloaders
    train_datasets = {
        "DFADD": RawAudio(path_to_database='../datasets/DFADD', meta_csv='meta.csv', return_label=True, nb_samp=args.nb_samp, part='train'),
        "CodecFake": RawAudio(path_to_database='../datasets/CodecFake', meta_csv='meta.csv', return_label=True, nb_samp=args.nb_samp, part='train'),
        "ASVspoof2021_DF": RawAudio(path_to_database='../datasets/ASVspoof2021_DF', meta_csv='meta.csv', return_label=True, nb_samp=args.nb_samp, part='train')
    }

    train_dataloader = DataLoader(ConcatDataset(train_datasets.values()), batch_size=args.batch_size, shuffle=True, num_workers=args.nb_worker)

    # Load encoders
    encoders = {
        "RawNet3": RawNet3(Bottle2neck, model_scale=8, context=True, encoder_type="ECA", nOut=256).to(device),
        "Wav2Vec2": Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(device),
        "SpeechSplit_timbre_content": Generator_3(hparams).to(device),
        "SpeechSplit_pitch": Generator_3(hparams).to(device),
        "SpeechSplit_rhythm": Generator_3(hparams).to(device),
    }
    for encoder in encoders.values():
        encoder.requires_grad_(False)  # Freeze all encoders

    # Initialize connectors
    connectors = {modality: QFormerConnector(input_dim=args.input_dim, query_dim=args.query_dim, num_queries=args.num_queries,
                                             num_heads=args.num_heads, num_layers=args.num_layers).to(device)
                  for modality in encoders.keys()}

    # Optimizer and criterion
    optimizer = optim.Adam([p for conn in connectors.values() for p in conn.parameters()], lr=args.lr)
    criterion = nn.CrossEntropyLoss()  # Loss function

    # Training loop
    for epoch in range(args.epochs):
        train_loss = train_phase_1(args, train_dataloader, encoders, connectors, optimizer, criterion, device)
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {train_loss:.4f}")

        # Save connector checkpoints
        for modality, connector in connectors.items():
            torch.save(connector.state_dict(), os.path.join(checkpoint_dir, f"{modality}_connector.pth"))

    print("Training complete.")


if __name__ == "__main__":
    args = argparse.Namespace(
        input_dim=1024, query_dim=768, num_queries=16, num_heads=8, num_layers=4,
        model_name="AudioExpertLLM", batch_size=32, lr=0.001, epochs=10, device="cuda" if torch.cuda.is_available() else "cpu",
        nb_samp=64600, nb_worker=4
    )
    main(args)