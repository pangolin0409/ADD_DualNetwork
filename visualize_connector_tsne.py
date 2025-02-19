from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import torch
from trainer import get_modality_embedding, get_components
from torch.utils.data import DataLoader, ConcatDataset, Subset
from data.dataloader import RawAudio
from config import init
import torch.nn.functional as F
import os
from load_datasets import load_test_data

def extract_connector_features_and_labels(
    dataloader, encoders, connectors, device, wav2vec_extractor
):
    """
    通过 t-SNE 可视化 Connector 的输出。
    """
    # 存储特征和标签
    modality_features = {modality: {0: [], 1: []} for modality in encoders.keys()}
    for batch in tqdm(dataloader, desc="提取 Connector 特征"):
        audio, labels = batch
        audio, labels = audio.to(device), labels.to(device)

        for modality in encoders.keys():
            with torch.no_grad():
                # Encoder 的处理逻辑
                output = get_modality_embedding(modality=modality, encoders=encoders, audio=audio, wav2vec_extractor=wav2vec_extractor, device=device)
                # Preprocessor 和 Connector
                connector_output = connectors[modality](output)

                # 使用平均池化将特征从三维降到二维
                pooled_output = F.adaptive_avg_pool1d(connector_output.permute(0, 2, 1), output_size=1).squeeze(2)

                # 根据标签分开存储 Connector 输出
                for i, label in enumerate(labels):
                    modality_features[modality][label.item()].append(pooled_output[i].cpu().numpy())

    return modality_features

def draw_tsne(modality_features, output_dim=2, save_dir='./tsne_plots'):
    """
    使用 t-SNE 绘制特征，并保存到硬盘。
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for modality, label_dict in modality_features.items():
        plt.figure()
        for label, features in label_dict.items():
            # 将特征转换为 numpy 数组
            features = np.array(features)

            # 展平特徵數據到二維
            features = features.reshape(features.shape[0], -1)

            # 使用 t-SNE 降维
            tsne = TSNE(n_components=output_dim)
            tsne_results = tsne.fit_transform(features)

            # 绘图
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], label=f'Label {label}')
        
        plt.title(f't-SNE for {modality}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'{modality}_tsne.png'))
        plt.close()

     # 绘制所有模态的 label 0 和 label 1 的 t-SNE 图
    for label in [0, 1]:
        plt.figure()
        for modality, label_dict in modality_features.items():
            features = np.array(label_dict[label])

            # 展平特徵數據到二維
            features = features.reshape(features.shape[0], -1)

            # 使用 t-SNE 降维
            tsne = TSNE(n_components=output_dim)
            tsne_results = tsne.fit_transform(features)

            # 绘图
            plt.scatter(tsne_results[:, 0], tsne_results[:, 1], label=f'{modality}')
        
        plt.title(f't-SNE for all modalities - Label {label}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'all_modalities_label_{label}_tsne.png'))
        plt.close()

if __name__=='__main__':
    args = init()
    device = args.device

    # 定義目標維度（連接器的輸入維度）
    target_query_dim = args.query_dim  # 例如 512

    # 定義每個模態的輸入維度
    modality_input_dims = {
        "RawNet3": 256,
        "Wav2Vec2": 1024,
        "SpeechSplit_timbre_content": 2,
        "SpeechSplit_pitch": 64,
        "SpeechSplit_rhythm": 2
    }

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
        real_indices, spoof_indices = load_test_data(meta_path=f'../datasets/{name}/train/meta.csv', dataset_name=name, target_fake_ratio=2)
        real_subset = Subset(training_set, real_indices)
        spoof_subset = Subset(training_set, spoof_indices)
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        training_set_list.append(adjusted_set)

    final_training_set = ConcatDataset(training_set_list)
    train_dataloader = DataLoader(final_training_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.nb_worker)

    encoders, wav2vec_extractor, connectors, experts, classifier = get_components(args=args, modality_input_dims=modality_input_dims)


    # 提取 Connector 特征
    modality_features = extract_connector_features_and_labels(
        dataloader=train_dataloader,
        encoders=encoders,
        connectors=connectors,
        device=device,
        wav2vec_extractor=wav2vec_extractor
    )
    # 繪製 t-SNE 
    draw_tsne(modality_features)