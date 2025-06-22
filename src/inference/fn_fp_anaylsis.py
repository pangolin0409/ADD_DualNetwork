import pandas as pd
import matplotlib.pyplot as plt
import os

# 只針對 FN 做分析（假語音被判成真）
def analyze_false_negatives(meta, misclassified):
    fn_df = misclassified[(misclassified['label'] == 1) & (misclassified['pred'] == 0)]
    spoof_total_counts = len(meta['label']== 'spoof')

    # 從檔名中提取 TTS 名稱，例如 xxx_xtts.wav → xtts
    fn_df['tts_system'] = fn_df['filename'].str.split('_').str[1].str.replace('.wav', '')

    # 計算每個 TTS 的誤判數（False Negatives）
    fn_counts = fn_df['tts_system'].value_counts()
    tts_names = fn_df['tts_system'].unique()

    fn_rates = fn_counts / spoof_total_counts

    return fn_rates, tts_names

# 只針對 FP 做分析（真語音被判成假）
def analyze_false_positives(meta, misclassified):
    fp_df = misclassified[(misclassified['label'] == 0) & (misclassified['pred'] == 1)]
    fp_counts = fp_df.shape[0]
    print(f"Total False Positives: {fp_counts}")
    bonafide_total_counts = len(meta['label']== 'bonafide')
    fp_rates = fp_counts / bonafide_total_counts

    return fp_rates

def plot_rates(rates: list, tts_names: list, title: str, ylabel: str, save_path: str):
    colors = ['skyblue', 'lightgreen', 'salmon', 'lightcoral', 'lightsalmon']
    colors = colors[:len(tts_names)]  # 確保顏色數量與 TTS 系統數量一致
    xlabel = "TTS System" if len(tts_names) < 1 else "Bonafide"
    plt.figure(figsize=(8, 5))
    plt.bar(tts_names, rates, color=colors)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.ylim(0, 0.3)
    plt.xticks(rotation=45)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)

def main(args):
    # 載入誤判紀錄（FN 和 FP）
    meta = pd.read_csv(f"F:/datasets/{args.task}/test/meta.csv")
    misclassified = pd.read_csv(f"./logs/{args.model_name}/misclassified_files_on_{args.task}.csv")
    fn_rates, tts_names = analyze_false_negatives(meta, misclassified)
    fp_rates =analyze_false_positives(meta, misclassified)
    save_path_prefix = f"./plot/{args.model_name}/"
    os.makedirs(save_path_prefix, exist_ok=True)
    fn_rates_file = f"{save_path_prefix}fn_rates_{args.task}.png"
    fp_rates_file = f"{save_path_prefix}fp_rates_{args.task}.png"

    # 標題包含 model name 跟 task name
    fn_title = f"False Negative Rate per TTS System for {args.model_name} on {args.task}"
    fp_title = f"False Positive Rate for {args.model_name} on {args.task}"
    plot_rates(fn_rates, tts_names, fn_title, 'False Negative Rate', fn_rates_file)
    plot_rates(fp_rates, ['Bonafide'], fp_title, 'False Positive Rate', fp_rates_file)

if __name__ == "__main__":
    from config.config import init
    args = init()
    main(args)