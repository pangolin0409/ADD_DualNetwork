import os
import gc
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
from src.data.dataloader import RawAudio
from src.models.Detector import Detector
import torch.nn as nn
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from umap import UMAP
import pandas as pd

def safe_release(*objs):
    for obj in objs:
        if obj is not None:
            if isinstance(obj, nn.Module):
                obj.cpu()
            del obj
    gc.collect()
    torch.cuda.empty_cache()

def load_model(model_class, model_path, device, **model_args):
    checkpoint = torch.load(model_path)
    epoch = checkpoint['epoch']
    model = model_class(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    model.eval()
    return model, epoch, checkpoint.get("best_temp"), checkpoint.get("best_alpha")

def load_datasets(task, nb_samp, batch_size, nb_worker):
    # 加載數據集 (改為批量推理)
    print(f"Loading dataset: {task}")
    test_set = RawAudio(
        path_to_database=f'F:/datasets//{task}',
        meta_csv='meta.csv',
        return_label=True,
        nb_samp= nb_samp,
        part='test',
    )
    testDataLoader = DataLoader(
        test_set, 
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False, 
        num_workers=nb_worker, 
        pin_memory=False, # 鎖定記憶體
        persistent_workers=False # 用同一批 worker，不再重新創建
    )
    return testDataLoader

def inference_multi_datasets(args, model_path, dataset_list):
    """
    針對多個 dataset 執行 inference，整合所有特徵、標籤、score 等資訊，回傳統一的列表。
    並準備好用於畫圖的 df。
    """
    model_args = {
        'ssl_model_name': args.ssl_model_name,
        'encoder_dim': args.encoder_dim,
        'num_experts': args.num_experts,
        'num_classes': args.num_classes,
        'max_temp': args.max_temp,
        'min_temp': args.min_temp,
        'start_alpha': args.start_alpha,
        'end_alpha': args.end_alpha,
        'warmup_epochs': args.warmup_epochs
    }

    model, epoch, temp, alpha = load_model(
        model_class=Detector,
        model_path=model_path,
        device=args.device,
        **model_args
    )

    all_scores = []
    all_labels = []
    all_features = []
    all_domains = []

    try:
        model.eval()
        for task in dataset_list:
            print(f"\n=== Running inference on dataset: {task} ===")
            test_loader = load_datasets(task, args.nb_samp, args.batch_size, args.nb_worker)

            for data_slice in tqdm(test_loader, desc=f"Inferencing {task}"):
                wave, labels, filename = data_slice
                wave = wave.to(args.device)
                labels = labels.to(args.device)

                with torch.no_grad():
                    logits, routing, fused_output, _ = model(wave=wave, temp=temp, alpha=alpha)
                    scores = softmax(logits, dim=1)[:, 1]

                all_scores.extend(scores.detach().cpu().numpy().tolist())
                all_labels.extend(labels.detach().cpu().numpy().tolist())
                all_features.extend(fused_output.detach().cpu().numpy().tolist())
                all_domains.extend([task] * len(labels))

    finally:
        safe_release(model)

    # 降維
    
    umap = UMAP(n_components=2, random_state=42)
    reduced = umap.fit_transform(all_features)

    # 準備 DataFrame
    
    df = pd.DataFrame({
        "x": reduced[:, 0],
        "y": reduced[:, 1],
        "label": all_labels,
        "domain": all_domains
    })

    return df

def visualize_spoof_by_domain(df, save_path):
    import seaborn as sns
    import matplotlib.pyplot as plt

    df_bonafide = df[df['label'] == 0]
    df_spoof = df[df['label'] == 1]

    plt.figure(figsize=(10, 8))
    
    # 畫 bonafide
    sns.scatterplot(
        data=df_bonafide, x="x", y="y",
        color="gray", label="Bona fide",
        alpha=0.2, s=8
    )

    # 畫 spoof 並根據 domain 上色
    spoof_plot = sns.scatterplot(
        data=df_spoof, x="x", y="y",
        hue="domain",
        alpha=0.6, s=8
    )

    # 取得 legend 順序並排序
    handles, labels = spoof_plot.get_legend_handles_labels()
    if "Bona fide" not in labels:
        handles.insert(0, plt.Line2D([], [], marker='o', linestyle='', color='gray', label='Bona fide'))
        labels.insert(0, "Bona fide")

    # 重新排列 legend
    order = sorted(range(len(labels)), key=lambda i: labels[i].lower())
    spoof_plot.legend(
        [handles[i] for i in order],
        [labels[i] for i in order],
        title="Spoof Domain", bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.
    )

    plt.title("Spoof (colored by domain) vs Bona fide (gray)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    from config.config import init
    args = init()
    model_path = os.path.join(args.model_folder, "LAYER_TIME_DAUL_BRANCH_da", args.checkpt_name)
    datasets = ["Asvspoof2019_LA", "Asvspoof2021_LA", "ASVspoof2021_DF", "in_the_wild", "ADD", "en-fbgkx-librispeech-2025_v1", "zh-fbgkx-aishell3-2025_v1"]
    # datasets = ["in_the_wild", "ADD"]
    df = inference_multi_datasets(args, model_path, datasets)
    df.to_csv("inference_results.csv", index=False)    
    visualize_spoof_by_domain(df, "spoof_vs_bonafide_by_domain.png")
