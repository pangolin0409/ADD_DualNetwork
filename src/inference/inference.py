import os
import gc
import wandb
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
import numpy as np
import onnxruntime as ort
from transformers import Wav2Vec2FeatureExtractor
from src.utils.eval_metrics import compute_eer, calculate_metrics
from src.data.dataloader import RawAudio
from src.models.Detector import Detector
from src.utils.common_utils import get_git_branch
from src.utils.visualize import (
    draw_expert_usage,
    draw_ft_dist,
    draw_roc_curve
)
import torch.nn as nn
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

def inference_loop(args, model, temp, alpha):
    testDataLoader = load_datasets(args.task, args.nb_samp, args.batch_size, args.nb_worker)
    score_loader = []
    label_loader = []
    local_gating_loader = []
    moe_feature_loader = []
    filename_loader = []
    # 遍歷測試數據
    for i, data_slice in enumerate(tqdm(testDataLoader)):
        wave, labels, filename = data_slice
        wave = wave.to(args.device)
        labels = labels.to(args.device)
        # 模型推理 (批量)
        with torch.no_grad():
            logits, routing, fused_output, time_pooled_feat = model(wave=wave, temp=temp, alpha=alpha)
            scores = softmax(logits, dim=1)[:, 1]

        score_loader.extend(scores.detach().cpu().numpy().tolist())
        label_loader.extend(labels.detach().cpu().numpy().tolist())
        moe_feature_loader.extend(fused_output.detach().cpu().numpy().tolist())
        local_gating_loader.extend(routing.detach().cpu().numpy().tolist())
        filename_loader.extend(filename)
    
    return score_loader, label_loader, local_gating_loader, moe_feature_loader, filename_loader

def inference(args, model_path, save_path):
    wandb.init(
        project="audio-deepfake-detection",  #專案名稱
        name=f"{get_git_branch()}_trained_on_{args.model_name}_tested_on_{args.task}",  # 實驗名稱
        config=vars(args),
    )
    model = None
    try:
        model_args = {
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
            **model_args)
        score_loader, label_loader, local_gating_loader, moe_feature_loader, filename_loader = inference_loop(args, model, temp, alpha)
    finally:
        safe_release(model)

    scores = np.array(score_loader)
    labels = np.array(label_loader)
    filenames = np.array(filename_loader)

    # 根據標籤分割分數
    nontarget_scores = scores[labels == 0]  # 負例 (bonafide)
    target_scores = scores[labels == 1]     # 正例 (spoof)
    

    # compute_eer(spoof, bonafide)
    eer, frr, far, threshold = compute_eer(target_scores, nontarget_scores)
    # 計算其他評估指標
    precision, recall, f1, cm = calculate_metrics(scores, labels, threshold)
    
    print(f'Equal Error Rate (EER): {eer}, False Rejection Rate (FFR): {frr}, False Acceptance Rate (FAR): {far}, Threshold: {threshold}, Temp: {temp}, Alpha: {alpha}, Epoch: {epoch}')
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    print(f'Confusion Matrix:\n{cm}')
    with open(os.path.join(save_path, "inference.txt"), "a") as f:
        f.write(f"Test set: {args.task}, Equal Error Rate (EER): {eer}, False Rejection Rate (FFR): {frr}, False Acceptance Rate (FAR): {far}, Threshold: {threshold}\n, Temp: {temp}, Alpha: {alpha}\n, Epoch: {epoch}\n")
        f.write(f"Precision: {precision}, Recall: {recall}, F1 Score: {f1}\n")
        f.write(f"Confusion Matrix:\n{cm}\n")

    preds = (scores >= threshold).astype(int)
    # 組成 DataFrame
    df = pd.DataFrame({
        'filename': filenames,
        'label': labels,
        'score': scores,
        'pred': preds
    })

    # 篩選誤判樣本
    fp_df = df[(df['label'] == 0) & (df['pred'] == 1)]  # False Positive
    fn_df = df[(df['label'] == 1) & (df['pred'] == 0)]  # False Negative

    # 合併錯誤樣本並加入錯誤類型欄位
    fp_df['error_type'] = 'FP'
    fn_df['error_type'] = 'FN'
    error_df = pd.concat([fp_df, fn_df], ignore_index=True)

    # 儲存為 CSV
    error_df.to_csv(os.path.join(save_path, f"misclassified_files_on_{args.task}.csv"), index=False)

    wandb.log({
        "Train set": args.model_name,
        "Test set": args.task,
        "EER": eer,
        "FFR": frr,
        "FAR": far,
        "Threshold": threshold,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'temp': temp,
        'alpha': alpha,
        "epoch": epoch,
    })


    return scores, labels, local_gating_loader, moe_feature_loader

def log_selected_plots(plot_dir, task=None, prefix=True):
    """
    自動從 plot_dir 上傳幾張精選圖表到 WandB。
    """
    files = {
        "expert_usage": f"expert_usage_on_{task}.png",
        "feature_dist": f"feature_distribution_on_{task}.png",
        "roc_curve": f"roc_curve_on_{task}.png",
    }

    log_dict = {}
    for key, fname in files.items():
        path = os.path.join(plot_dir, fname)
        if os.path.exists(path):
            img_key = f"{task}_{key}" if prefix and task else key
            log_dict[img_key] = wandb.Image(path)
        else:
            print(f"[WARN] Missing plot: {fname}")

    if log_dict:
        wandb.log(log_dict)

def main(args):
    model_path = os.path.join(args.model_folder, args.model_name, args.checkpt_name)
    save_path = os.path.join(args.log_path, args.model_name, args.checkpt_name)
    os.makedirs(save_path, exist_ok=True)
    scores, labels, local_gating_loader, moe_feature_loader = inference(args, model_path, save_path)

    plot_dir = os.path.join(args.plot_path, args.model_name)
    os.makedirs(plot_dir, exist_ok=True)

    # === 視覺化 ===
    draw_expert_usage(local_gating_loader, args.task, plot_dir)
    draw_ft_dist(moe_feature_loader, labels, args.task, plot_dir, label_names=["Bona fide", "Spoof"])
    draw_roc_curve(scores, labels, args.task, plot_dir)

    print(f"All visualizations saved in: {plot_dir}")

if __name__ == "__main__":
    from config.config import init
    args = init()
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
    main(args)