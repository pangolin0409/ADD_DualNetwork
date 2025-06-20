import os
import gc
import wandb
import torch
from torch.utils.data import DataLoader
from torch.nn.functional import softmax
from tqdm import tqdm
import numpy as np
import pandas as pd
from src.utils.eval_metrics import compute_eer, calculate_metrics
from src.data.dataloader import RawAudio
from src.utils.common_utils import get_git_branch
import torch.nn as nn
from src.utils.model_utils import select_model
from src.utils.visualize import (
    draw_roc_curve,
)
def safe_release(*objs):
    for obj in objs:
        if obj is not None:
            if isinstance(obj, nn.Module):
                obj.cpu()
            del obj
    gc.collect()
    torch.cuda.empty_cache()


def load_model(model_path, args):
    model, is_use_wav2vec_ft = select_model(args)
    checkpoint = torch.load(model_path, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.eval()
    return model, is_use_wav2vec_ft

def load_datasets(task, nb_samp, batch_size, nb_worker):
    print(f"Loading dataset: {task}")
    test_set = RawAudio(
        path_to_database=f'F:/datasets/{task}',
        meta_csv='meta.csv',
        return_label=True,
        nb_samp= nb_samp,
        part='test',
        return_wav2vec_ft=True, 
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

def inference_loop(args, model, is_use_wav2vec_ft):
    testDataLoader = load_datasets(args.task, args.nb_samp, args.batch_size, args.nb_worker)
    score_loader = []
    label_loader = []
    filename_loader = []
    # 遍歷測試數據
    for i, data_slice in enumerate(tqdm(testDataLoader)):
        wave, labels, filename, wav2vec_ft = data_slice
        labels = labels.to(args.device)
        
        # 模型推理 (批量)
        with torch.no_grad():
            if is_use_wav2vec_ft:
                wav2vec_ft = wav2vec_ft.to(args.device)
                logits = model(wav2vec_ft)
            else:
                wave = wave.to(args.device)
                logits = model(wave)
            scores = softmax(logits, dim=1)[:, 1]

        score_loader.extend(scores.detach().cpu().numpy().tolist())
        label_loader.extend(labels.detach().cpu().numpy().tolist())
        filename_loader.extend(filename)
    
    return score_loader, label_loader, filename_loader

def inference(args, model_path, save_path):
    wandb.init(
        project="audio-deepfake-detection",  #專案名稱
        name=f"{get_git_branch()}_trained_on_{args.model_name}_tested_on_{args.task}",  # 實驗名稱
        config=vars(args),
    )
    model = None
    try:
        model, is_use_wav2vec_ft = load_model(model_path, args)
        score_loader, label_loader, filename_loader = inference_loop(args, model, is_use_wav2vec_ft)
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
    
    print(f'Equal Error Rate (EER): {eer}, False Rejection Rate (FFR): {frr}')
    print(f'Precision: {precision}, Recall: {recall}, F1 Score: {f1}')
    print(f'Confusion Matrix:\n{cm}')
    with open(os.path.join(save_path, "inference.txt"), "a") as f:
        f.write(f"Test set: {args.task}, Equal Error Rate (EER): {eer}, False Rejection Rate (FFR): {frr}, False Acceptance Rate (FAR): {far}, Threshold: {threshold}\n")
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
        "Threshold": threshold
    })
    return scores, labels

def main(args):
    model_path = os.path.join(args.model_folder, args.model_name, 'best_model.pth')
    save_path = os.path.join(args.log_path, args.model_name)
    os.makedirs(save_path, exist_ok=True)
    scores, labels = inference(args, model_path, save_path)

    plot_dir = os.path.join(args.plot_path, args.model_name)
    os.makedirs(plot_dir, exist_ok=True)

    # === 視覺化 ===
    draw_roc_curve(scores, labels, args.task, plot_dir)
    
if __name__ == "__main__":
    from config.config import init
    args = init()
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
    main(args)