from torch.utils.data import DataLoader, ConcatDataset, Subset
from src.data.dataloader import RawAudio
import pandas as pd
import random

"""
    載入訓練和驗證數據集。
"""
def load_datasets(sample_rate, batch_size, dataset_names, worker_size, target_fake_ratio, part, is_downsample=False):
    # 初始化 datasets 和 dataloaders
    data_sets = {}
    for dataset_name in dataset_names:
        data_sets[dataset_name] = RawAudio(
            path_to_database=f'E:/datasets/{dataset_name}',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=sample_rate,
            part=part
        )
    
    data_set_list = []
    for name, data_set in data_sets.items():
        # 如果是下採樣模式，則進行下採樣
        if is_downsample:
            print(f"Processing dataset : {name}")
            real_indices, spoof_indices = downsample_data(meta_path=f'E:/datasets/{name}/{part}/meta.csv', dataset_name=name, target_fake_ratio=target_fake_ratio)
        else:
            meta = pd.read_csv(f'E:/datasets/{name}/{part}/meta.csv')
            real_indices = meta[meta['label'] == 'bonafide'].index.tolist()
            spoof_indices = meta[meta['label'] == 'spoof'].index.tolist()
                
        # real_indices = random.sample(real_indices, target_fake_count)
        print(f'Real samples: {len(real_indices)}, Spoof samples: {len(spoof_indices)}')
        real_subset = Subset(data_set, real_indices)
        spoof_subset = Subset(data_set, spoof_indices)
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        data_set_list.append(adjusted_set)

    final_data_set = ConcatDataset(data_set_list)
    dataloader = DataLoader(final_data_set, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=worker_size, pin_memory=True, persistent_workers=True)

    return dataloader

"""
    下採樣數據集，保留所有真樣本，並將假樣本的數量控制為真樣本的 target_fake_ratio 倍。
"""
def downsample_data(meta_path: str, dataset_name: str, target_fake_ratio: int = 2) -> tuple:
    print(f"Processing dataset: {dataset_name}")
    meta = pd.read_csv(meta_path)
    meta = meta.reset_index(drop=True)

    # 提取真實樣本
    real_indices = meta[meta['label'] == 'bonafide'].index.tolist()
    target_fake_count = int(len(real_indices) * target_fake_ratio)
    spoof_indices = stratified_spoof_sampling(meta, target_fake_count)
    return real_indices, spoof_indices

def stratified_spoof_sampling(meta: pd.DataFrame, total_target_count: int) -> list:
    sampled_indices = []
    meta = meta[meta["label"] == "spoof"].copy()
    spoof_types = meta["file_id"].dropna().unique()
    per_type_count = total_target_count // len(spoof_types)

    # 確保總數不超標的同時，類型平均
    for stype in spoof_types:
        sub_df = meta[meta["file_id"] == stype]
        if len(sub_df) >= per_type_count:
            sampled = sub_df.sample(per_type_count, random_state=42)
        else:
            sampled = sub_df
        sampled_indices.extend(sampled.index.tolist())

    # 若總數還不足，從剩餘未選的 spoof 中補足
    if len(sampled_indices) < total_target_count:
        remaining_pool = meta.drop(index=sampled_indices)
        needed = total_target_count - len(sampled_indices)
        if len(remaining_pool) >= needed:
            extra = remaining_pool.sample(needed, random_state=42)
            sampled_indices.extend(extra.index.tolist())
        else:
            sampled_indices.extend(remaining_pool.index.tolist())

    return sampled_indices

def downsample_test_data(meta_path: str, dataset_name: str) -> tuple:
    print(f"Processing dataset: {dataset_name}")
    meta = pd.read_csv(meta_path)
    
    # 提取真實和假樣本索引
    real_indices = meta[meta['label'] == 'bonafide'].index.tolist()
    spoof_indices = meta[meta['label'] == 'spoof'].index.tolist()
    
    if len(real_indices) > 1000:
        real_indices = random.sample(real_indices, 1000)

    # 設定下採樣目標
    if len(spoof_indices) > 2000:
        spoof_indices = random.sample(spoof_indices, 2000)
    
    print(f'Real samples: {len(real_indices)}, Spoof samples: {len(spoof_indices)}')
    return real_indices, spoof_indices