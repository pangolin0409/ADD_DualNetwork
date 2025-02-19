from torch.utils.data import DataLoader, ConcatDataset, Subset
from data.dataloader import RawAudio
import pandas as pd
import random

"""
    載入訓練和驗證數據集。
"""
def load_datasets(sample_rate, batch_size, dataset_names, worker_size, target_fake_ratio, test=False):
    # 初始化 datasets 和 dataloaders
    training_sets = {}
    for dataset_name in dataset_names:
        training_sets[dataset_name] = RawAudio(
            path_to_database=f'../datasets/{dataset_name}',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=sample_rate,
            part='train'
        )
    
    training_set_list = []
    for name, training_set in training_sets.items():
        if test:
            real_indices, spoof_indices = downsample_test_data(meta_path=f'../datasets/{name}/train/meta.csv', dataset_name=name)
        else:
            real_indices, spoof_indices = downsample_data(meta_path=f'../datasets/{name}/train/meta.csv', dataset_name=name, target_fake_ratio=target_fake_ratio)
        real_subset = Subset(training_set, real_indices)
        spoof_subset = Subset(training_set, spoof_indices)
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        training_set_list.append(adjusted_set)

    final_training_set = ConcatDataset(training_set_list)
    train_dataloader = DataLoader(final_training_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=worker_size)

    # 初始化 datasets 和 dataloaders
    validation_sets = {}
    for dataset_name in dataset_names:
        validation_sets[dataset_name] = RawAudio(
            path_to_database=f'../datasets/{dataset_name}',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=sample_rate,
            part='validation'
        )

    validation_set_list = []
    for name, validation_set in validation_sets.items():
        real_indices, spoof_indices = downsample_data(meta_path=f'../datasets/{name}/validation/meta.csv', dataset_name=name, target_fake_ratio=target_fake_ratio)
        real_subset = Subset(validation_set, real_indices)
        spoof_subset = Subset(validation_set, spoof_indices)
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        validation_set_list.append(adjusted_set)

    final_validation_set = ConcatDataset(validation_set_list)
    validation_dataloader = DataLoader(final_validation_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=worker_size)

    return train_dataloader, validation_dataloader

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
    target_fake_count = int(len(real_indices) * target_fake_ratio)
    if len(spoof_indices) > target_fake_count:
        spoof_indices = random.sample(spoof_indices, target_fake_count)
    # real_indices = random.sample(real_indices, target_fake_count)
    print(f'Real samples: {len(real_indices)}, Spoof samples: {len(spoof_indices)}')
    return real_indices, spoof_indices


def downsample_test_data(meta_path: str, dataset_name: str) -> tuple:
    print(f"Processing dataset: {dataset_name}")
    meta = pd.read_csv(meta_path)
    
    # 提取真實和假樣本索引
    real_indices = meta[meta['label'] == 'bonafide'].index.tolist()
    spoof_indices = meta[meta['label'] == 'spoof'].index.tolist()
    
    if len(real_indices) > 100:
        real_indices = random.sample(real_indices, 100)

    # 設定下採樣目標
    if len(spoof_indices) > 200:
        spoof_indices = random.sample(spoof_indices, 200)
    
    print(f'Real samples: {len(real_indices)}, Spoof samples: {len(spoof_indices)}')
    return real_indices, spoof_indices