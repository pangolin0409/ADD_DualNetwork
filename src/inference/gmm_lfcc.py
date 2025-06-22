# 根據你最新的 meta.csv 結構來更新整合
# 不再使用 protocol.txt，直接從 meta.csv 做 GMM 訓練與推論

import os
import logging
import pickle
import pandas as pd
import numpy as np
import soundfile as sf
from sklearn.mixture import GaussianMixture
from sklearn.metrics import roc_curve
from scipy.signal import lfilter
from LFCC_pipeline import lfcc  # 請確認你已提供此函數
from tqdm import trange, tqdm
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

def compute_eer_from_df(df):
    df['label_num'] = df['label'].apply(lambda x: 1 if x.lower() == 'spoof' else 0)
    scores = df['score'].values
    labels = df['label_num'].values
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    fnr = 1 - tpr
    abs_diffs = np.abs(fpr - fnr)
    min_index = np.argmin(abs_diffs)
    eer = (fpr[min_index] + fnr[min_index]) / 2
    return eer, fnr[min_index], fpr[min_index], thresholds[min_index]

def lfcc_gmm_cross_eval_from_meta(dataset_root, test_tasks, audio_ext='.flac'):
    # 訓練 on LA
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

    results = {}
    for task in tqdm(test_tasks, desc="Evaluating tasks"):
        if task == 'Asvspoof2019_LA':
            test_path = os.path.join(dataset_root, task, 'train')  # reuse training set
        else:
            test_path = os.path.join(dataset_root, task, 'test')
        meta_csv = os.path.join(test_path, 'meta.csv')
        audio_folder = os.path.join(test_path, 'audio')

        if task in ['SOTA', 'in_the_wild']:
            audio_ext = '.wav'
        else:
            audio_ext = '.flac'

        score_file = os.path.join(test_path, f'gmm_scores.txt')
        if os.path.exists(score_file):
            df_scores = pd.read_csv(score_file, sep=' ', header=None, names=['file', 'score', 'label'])
            df_scores['score'] = df_scores['score'] * -1.0
        else:
            df_scores = score_from_meta(gmm_model, meta_csv, audio_folder, audio_ext, score_file)
        eer, frr, far, threshold = compute_eer_from_df(df_scores)
        logging.info(f"[{task}] EER: {eer:.4f} | FAR: {far:.4f} | FRR: {frr:.4f} | Threshold: {threshold:.4f}")
        results[task] = eer
    return results

# 執行範例
results = lfcc_gmm_cross_eval_from_meta('E:/datasets', ['Asvspoof2019_LA', 'Asvspoof2021_DF', 'in_the_wild', 'en-fbgkx-librispeech-2025_v1'])
print(results)
