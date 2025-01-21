import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, ConcatDataset
from data.dataloader import RawAudio
from tqdm import tqdm
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from model_RawNet2 import RawNet2
from trainer import QFormerConnector, ConnectorClassifier, SparseMoE
import argparse
import pandas as pd
import random

# 初始化超參數
def init():
    parser = argparse.ArgumentParser(description="Train Q-Former Connector with MoE for Audio Deepfake Detection")

    # 模型參數
    parser.add_argument("--input_dim", type=int, default=1024, help="Input feature dimension from RawNet2/Wav2Vec2")
    parser.add_argument("--query_dim", type=int, default=512, help="Projected feature dimension")
    parser.add_argument("--num_queries", type=int, default=16, help="Number of learnable queries")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of Transformer layers")

    # Router 和 Experts 參數
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension in each expert")
    parser.add_argument("--moe_output_dim", type=int, default=256, help="Output dimension of MoE")

    # 訓練參數
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument('-nb_samp', type = int, default = 64600)
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--nb_worker", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints_second_stage", help="Directory to save checkpoints")
    parser.add_argument("--best_checkpoint_path", type=str, default="./checkpoints/best_checkpoint.pth", help="Path to the best checkpoint from the first stage")

    #DNN args
    parser.add_argument('-m_first_conv', type = int, default = 251)
    parser.add_argument('-m_in_channels', type = int, default = 1)
    parser.add_argument('-m_filts', type = list, default = [128, [128,128], [128,256], [256,256]])
    parser.add_argument('-m_blocks', type = list, default = [2, 4])
    parser.add_argument('-m_nb_fc_att_node', type = list, default = [1])
    parser.add_argument('-m_nb_fc_node', type = int, default = 1024)
    parser.add_argument('-m_gru_node', type = int, default = 1024)
    parser.add_argument('-m_nb_gru_layer', type = int, default = 1)
    parser.add_argument('-m_nb_samp', type = int, default = 64600)
    
    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            print(k, v)
            args.model[k[2:]] = v
    args.model['nb_classes'] = 2        
    
    return args


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

# 第二階段訓練函數
def train_second_stage(
    dataloader: DataLoader,
    rawnet2: nn.Module,
    wav2vec_model: nn.Module,
    connector_rawnet: nn.Module,
    connector_wav2vec: nn.Module,
    moe: nn.Module,
    classifier: nn.Module,
    criterion: callable,
    optimizer: optim.Optimizer,
    device: torch.device,
    checkpoint_dir: str
):
    rawnet2.eval()
    wav2vec_model.eval()
    connector_rawnet.eval()
    connector_wav2vec.eval()
    moe.train()
    classifier.train()

    total_loss = 0
    correct = 0
    total = 0

    for audio, labels in tqdm(dataloader):
        audio, labels = audio.to(device), labels.to(device)

        # RawNet2 特徵提取
        with torch.no_grad():
            rawnet_features = rawnet2(audio, is_test=True)

        # Wav2Vec2 特徵提取
        processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")
        input_values = processor(audio, sampling_rate=16000, return_tensors="pt").input_values.to(device)
        input_values = input_values.squeeze(0)
        with torch.no_grad():
            wav2vec_features = wav2vec_model(input_values).last_hidden_state
        wav2vec_features = wav2vec_features.mean(dim=1)

        # Connector 處理
        with torch.no_grad():
            rawnet_proj = connector_rawnet(rawnet_features)
            wav2vec_proj = connector_wav2vec(wav2vec_features)

        # 特徵拼接
        combined_features = torch.cat([rawnet_proj.mean(dim=1), wav2vec_proj.mean(dim=1)], dim=1)

        # MoE 處理
        moe_output = moe(combined_features)

        # 分類器輸出
        outputs = classifier(moe_output)

        # 損失計算
        loss = criterion(outputs, labels)

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

    # 儲存 Checkpoint
    checkpoint_path = os.path.join(checkpoint_dir, "second_stage_checkpoint.pth")
    torch.save({
        "connector_rawnet": connector_rawnet.state_dict(),
        "connector_wav2vec": connector_wav2vec.state_dict(),
        "moe": moe.state_dict(),
        "classifier": classifier.state_dict()
    }, checkpoint_path)

    return total_loss / len(dataloader), accuracy

# 主程序
def main(args):
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    training_sets = {
        "DFADD": RawAudio(path_to_database=f'../datasets/DFADD', meta_csv='meta.csv', return_label=True, nb_samp=args.nb_samp, part='train'),
        "CodecFake": RawAudio(path_to_database=f'../datasets/CodecFake', meta_csv='meta.csv', return_label=True, nb_samp=args.nb_samp, part='train'),
        "ASVspoof2021_DF": RawAudio(path_to_database=f'../datasets/ASVspoof2021_DF', meta_csv='meta.csv', return_label=True, nb_samp=args.nb_samp, part='train')
    }

    training_set_list = []
    for name, training_set in training_sets.items():
        real_indices, spoof_indices = downsample_data(meta_path=f'../datasets/{name}/train/meta.csv', dataset_name=name, target_fake_ratio=2)
        real_subset = Subset(training_set, real_indices)
        spoof_subset = Subset(training_set, spoof_indices)
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        training_set_list.append(adjusted_set)

    final_training_set = ConcatDataset(training_set_list)
    train_loader = DataLoader(final_training_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.nb_worker)


    # 初始化模型
    rawnet2 = RawNet2(args.model).to(args.device)
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(args.device)
    connector_rawnet = QFormerConnector(
        input_dim=args.input_dim, query_dim=args.query_dim, num_queries=args.num_queries,
        num_heads=args.num_heads, num_layers=args.num_layers
    ).to(args.device)
    connector_wav2vec = QFormerConnector(
        input_dim=args.input_dim, query_dim=args.query_dim, num_queries=args.num_queries,
        num_heads=args.num_heads, num_layers=args.num_layers
    ).to(args.device)
    moe = SparseMoE(input_dim=args.query_dim * 2, num_experts=args.num_experts, hidden_dim=args.hidden_dim, 
                output_dim=args.moe_output_dim, k=2).to(args.device)
    classifier = ConnectorClassifier(input_dim=args.moe_output_dim, num_classes=2).to(args.device)

    # 載入第一階段的最佳檢查點
    checkpoint = torch.load(args.best_checkpoint_path)
    rawnet2.load_state_dict(checkpoint["rawnet2"])
    connector_rawnet.load_state_dict(checkpoint["connector_rawnet"])
    connector_wav2vec.load_state_dict(checkpoint["connector_wav2vec"])

    # 定義損失函數與優化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(moe.parameters()) + list(classifier.parameters()), lr=args.lr)

    # 訓練模型，儲存最佳檢查點
    best_accuracy = 0.0
    checkpoint_dir = "./checkpoints"
    best_checkpoint_path = os.path.join(checkpoint_dir, "best_moe_checkpoint.pth")
    
    # 訓練模型
    for epoch in tqdm(range(args.epochs)):
        loss, acc = train_second_stage(
            train_loader, rawnet2, wav2vec_model, connector_rawnet, connector_wav2vec,
            moe, classifier, criterion, optimizer, args.device, args.checkpoint_dir
        )
        
         # 更新最佳檢查點
        if acc > best_accuracy:
            best_accuracy = acc
            torch.save({
                "moe": moe.state_dict(),
                "classifier": classifier.state_dict()
            }, best_checkpoint_path)

        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")

if __name__ == "__main__":
    args = init()
    main(args)
