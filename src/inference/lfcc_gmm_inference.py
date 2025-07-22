import os
import logging
import pickle
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve
from scipy.signal import lfilter
from src.inference.LFCC_pipeline import lfcc  # 請確認你已提供此函數
from tqdm import trange, tqdm
from src.utils.eval_metrics import compute_eer, calculate_metrics

logging.basicConfig(format='%(asctime)s %(levelname)-8s %(message)s', level=logging.INFO)

def Deltas(x, width=3):
    hlen = int(np.floor(width/2))
    win = list(range(hlen, -hlen-1, -1))
    xx_1 = np.tile(x[:, 0], (1, hlen)).reshape(hlen, -1).T
    xx_2 = np.tile(x[:, -1], (1, hlen)).reshape(hlen, -1).T
    xx = np.concatenate([xx_1, x, xx_2], axis=-1)
    D = lfilter(win, 1, xx)
    return D[:, hlen*2:]

def extract_lfcc(file, num_ceps=20, order_deltas=2, low_freq=0, high_freq=4000):
    sig, fs = sf.read(file)
    lfccs = lfcc(sig=sig, fs=fs, num_ceps=num_ceps, low_freq=low_freq, high_freq=high_freq)
    if order_deltas > 0:
        feats = [lfccs]
        for _ in range(order_deltas):
            feats.append(Deltas(feats[-1]))
        lfccs = np.vstack(feats)
    return lfccs

def extract_features(file):
    return extract_lfcc(file)

def train_gmm_from_meta(meta_csv, audio_folder, audio_ext='.flac', ncomp=128):
    df = pd.read_csv(meta_csv)
    logging.info(f"Training GMM with {len(df)} files")
    gmm_dict = {}
    for label in ['bonafide', 'spoof']:
        files = df[df['label'] == label]['filename'].str.replace('.flac', '').str.replace('.wav', '')
        logging.info(f"{len(files)} files for {label}")
        data = []
        for file in files[::5]:
            path = os.path.join(audio_folder, file + audio_ext)
            if not os.path.exists(path):
                continue
            feat = extract_features(path)
            data.append(feat)
        if len(data) == 0:
            raise ValueError(f"No data found for label {label}")
        X = np.vstack(data)
        gmm = GaussianMixture(n_components=ncomp, covariance_type='diag', max_iter=10, verbose=2).fit(X)
        gmm_dict[label] = gmm
        # save the model
        with open(f'gmm_{label}.pkl', 'wb') as f:
            pickle.dump(gmm, f)
    return gmm_dict

def score_from_meta(gmm_dict, meta_csv, audio_folder, audio_ext, score_file_path):
    df = pd.read_csv(meta_csv)
    results = []
    for _, row in tqdm(df.iterrows(), desc="Scoring files", total=len(df)):
        file = row['filename'].replace('.flac', '').replace('.wav', '')
        label = row['label']
        path = os.path.join(audio_folder, file + audio_ext)
        try:
            feat = extract_features(path)
            score = gmm_dict['bonafide'].score(feat) - gmm_dict['spoof'].score(feat)
        except:
            score = 0.0
        results.append((file, score, label))
    df_out = pd.DataFrame(results, columns=['file', 'score', 'label'])
    df_out.to_csv(score_file_path, sep=' ', header=False, index=False)
    return df_out

def lfcc_gmm_cross_eval_from_meta(dataset_root, test_tasks, audio_ext='.flac'):
    if not os.path.exists('gmm_bonafide.pkl') and not os.path.exists('gmm_spoof.pkl'):
        train_path = os.path.join(dataset_root, 'Asvspoof2019_LA', 'train')
        train_meta = os.path.join(train_path, 'meta.csv')
        train_audio = os.path.join(train_path, 'audio')
        gmm_model = train_gmm_from_meta(train_meta, train_audio, audio_ext=audio_ext)
    else:
        with open('gmm_bonafide.pkl', 'rb') as f:
            gmm_bonafide = pickle.load(f)
        with open('gmm_spoof.pkl', 'rb') as f:
            gmm_spoof = pickle.load(f)
        gmm_model = {'bonafide': gmm_bonafide, 'spoof': gmm_spoof}

    os.makedirs('./logs/lfcc_gmm/', exist_ok=True)
    results = {}

    for task in tqdm(test_tasks, desc="Evaluating tasks"):
        if task == 'Asvspoof2019_LA':
            test_path = os.path.join(dataset_root, task, 'train')
        else:
            test_path = os.path.join(dataset_root, task, 'test')
        meta_csv = os.path.join(test_path, 'meta.csv')
        audio_folder = os.path.join(test_path, 'audio')

        if task in ['zh-fbgkx-aishell3-2025_v1', 'en-fbgkx-librispeech-2025_v1', 'in_the_wild', 'ADD']:
            audio_ext = '.wav'
        else:
            audio_ext = '.flac'

        score_file = os.path.join(test_path, f'gmm_scores.txt')
        if os.path.exists(score_file):
            df_scores = pd.read_csv(score_file, sep=' ', header=None, names=['file', 'score', 'label'])
            df_scores['score'] = df_scores['score'] * -1.0
        else:
            df_scores = score_from_meta(gmm_model, meta_csv, audio_folder, audio_ext, score_file)

        # 將 label 轉成 0/1
        df_scores['label_num'] = df_scores['label'].apply(lambda x: 1 if x.lower() == 'spoof' else 0)
        labels = df_scores['label_num'].values
        scores = df_scores['score'].values

        # 分開 bonafide / spoof scores
        target_scores = scores[labels == 1]     # spoof
        nontarget_scores = scores[labels == 0]  # bonafide

        # 👉 使用新的 compute_eer 函數
        eer, frr, far, threshold = compute_eer(target_scores, nontarget_scores)

        # 推論 & 儲存錯誤樣本
        preds = (scores >= threshold).astype(int)
        df_preds = pd.DataFrame({
            'filename': df_scores['file'],
            'label': labels,
            'score': scores,
            'pred': preds
        })

        fp_df = df_preds[(df_preds['label'] == 0) & (df_preds['pred'] == 1)]
        fn_df = df_preds[(df_preds['label'] == 1) & (df_preds['pred'] == 0)]
        fp_df['error_type'] = 'FP'
        fn_df['error_type'] = 'FN'
        error_df = pd.concat([fp_df, fn_df], ignore_index=True)

        os.makedirs('./logs/lfcc_gmm', exist_ok=True)
        error_path = os.path.join('./logs/lfcc_gmm', f"misclassified_files_on_{task}.csv")
        error_df.to_csv(error_path, index=False)

        # 計算 precision, recall, f1
        precision, recall, f1, cm = calculate_metrics(target_scores, nontarget_scores, threshold)

        logging.info(
            f"[{task}] EER: {eer:.4f} | FAR: {far:.4f} | FRR: {frr:.4f} | Threshold: {threshold:.4f} | "
            f"Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}"
        )

        results[task] = {
            'EER': eer,
            'Threshold': threshold,
            'Precision': precision,
            'Recall': recall,
            'F1': f1,
            'FAR': far,
            'FRR': frr
        }

    return results

def main():
    dataset_root = 'F:/datasets'
    test_tasks = ['ADD']
    results = lfcc_gmm_cross_eval_from_meta(dataset_root, test_tasks)
    for task, res in results.items():
        print(f"Task: {task}, Results: {res}")

if __name__ == "__main__":
    main()