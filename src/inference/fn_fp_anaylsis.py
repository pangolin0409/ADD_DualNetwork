import pandas as pd
import matplotlib.pyplot as plt
import os

# 固定 TTS 系統的顏色
TTS_COLOR_MAP = {
    'bark': '#1f77b4',      # 藍色，穩重又清晰（比 skyblue 飽和）
    'f5tts': '#d62728',     # 紅色，強烈醒目（比 salmon 對比高）
    'kokoro': '#2ca02c',    # 綠色，鮮明不刺眼（比 lightgreen 更穩定）
    'xtts': '#9467bd',      # 紫色，高辨識度且不常見
    'xtts2': '#9467bd',      # 紫色，高辨識度且不常見
    'gptsovits': '#ff7f0e', # 橘色，活潑且與其他區分明顯
}

# 分析 False Negatives（假語音被判成真）
def analyze_false_negatives(meta, misclassified):
    fn_df = misclassified[(misclassified['label'] == 1) & (misclassified['pred'] == 0)].copy()
    spoof_total_counts = (meta['label'] == 'spoof').sum()
    print(f"Total spoof files: {spoof_total_counts}")

    # 從檔名提取 TTS 名稱
    fn_df['tts_system'] = fn_df['filename'].str.split('_').str[1].str.replace('.wav', '', regex=False)
    # tts_systems is xtts2 replace with xtts
    fn_df['tts_system'] = fn_df['tts_system'].replace({'xtts2': 'xtts'})

    # 計算每個 TTS 的誤判數
    fn_counts = fn_df['tts_system'].value_counts()
    tts_names = fn_counts.index.tolist()
    fn_rates = fn_counts / spoof_total_counts
    total_fn_rate = fn_df.shape[0] / spoof_total_counts

    return fn_rates, tts_names, total_fn_rate

# 分析 False Positives（真語音被判成假）
def analyze_false_positives(meta, misclassified):
    fp_df = misclassified[(misclassified['label'] == 0) & (misclassified['pred'] == 1)]
    fp_counts = fp_df.shape[0]
    bonafide_total_counts = (meta['label'] == 'bonafide').sum()
    fp_rate = fp_counts / bonafide_total_counts
    return fp_rate

# 畫圖：False Negative / False Positive Rates
def plot_rates(rates, names, title, ylabel, save_path):
    if len(names) == 1 and names[0].lower() == 'bonafide':
        colors = ['gray']
    else:
        colors = [TTS_COLOR_MAP.get(name, 'black') for name in names]

    plt.figure(figsize=(8, 5))
    plt.bar(names, rates, color=colors)
    plt.title(title)
    plt.xlabel("TTS System" if len(names) > 1 else "Bonafide")
    plt.ylabel(ylabel)
    plt.ylim(0, 0.15)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# 計算總體 FRR/FAR 並列印
def frr_far(meta, misclassified):
    spoof_total_counts = (meta['label'] == 'spoof').sum()
    bonafide_total_counts = (meta['label'] == 'bonafide').sum()

    print(f"Total spoof files: {spoof_total_counts}")
    print(f"Total bonafide files: {bonafide_total_counts}")

    fn = ((misclassified['label'] == 1) & (misclassified['pred'] == 0)).sum()
    fp = ((misclassified['label'] == 0) & (misclassified['pred'] == 1)).sum()

    frr = fn / spoof_total_counts
    far = fp / bonafide_total_counts

    print(f"False Rejection Rate (FRR): {frr:.4f}")
    print(f"False Acceptance Rate (FAR): {far:.4f}")

def main(args):
    meta = pd.read_csv(f"F:/datasets/{args.task}/test/meta.csv")
    misclassified = pd.read_csv(f"./logs/{args.model_name}/misclassified_files_on_{args.task}.csv")

    # 分析
    frr_far(meta, misclassified)
    fn_rates, tts_names, total_fn_rate = analyze_false_negatives(meta, misclassified)
    fp_rate = analyze_false_positives(meta, misclassified)

    print(f"Total False Negative Rate: {total_fn_rate:.4f}")
    print(f"False Positive Rate: {fp_rate:.4f}")

    # 檔案路徑與命名
    save_path_prefix = f"./plot/{args.model_name}/"
    os.makedirs(save_path_prefix, exist_ok=True)
    fn_rates_file = f"{save_path_prefix}fn_rates_{args.task}.png"
    fp_rates_file = f"{save_path_prefix}fp_rates_{args.task}.png"

    # 畫圖
    plot_rates(fn_rates.values, fn_rates.index.tolist(),
               f"False Negative Rate per TTS System for {args.model_name} on {args.task}",
               'False Negative Rate', fn_rates_file)

    plot_rates([fp_rate], ['Bonafide'],
               f"False Positive Rate for {args.model_name} on {args.task}",
               'False Positive Rate', fp_rates_file)

if __name__ == "__main__":
    from config.config import init
    args = init()
    main(args)
