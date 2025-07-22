import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 固定 TTS 顯示順序

#TTS_ORDER = ['bark', 'f5tts', 'kokoro', 'xtts', 'gptsovits']
# 按照 fbgkx 的順序排列
TTS_ORDER = ['bonafide']


# 偵測器清單（固定順序）
DETECTOR_NAMES = [
    "Ours",
    "XLSR_SLS",
    "XLSR_MLP",
    "AASSIST",
    "RawNet2",
    "LFCC_GMM",
]

# 固定每個偵測器的顏色
DETECTOR_COLOR_MAP = {
    "Ours": "#8c564b", # 棕
    "XLSR_SLS": "#1f77b4",                  # 藍
    "XLSR_MLP": "#ff7f0e",          # 橘
    "AASSIST": "#2ca02c",               # 綠
    "RawNet2": "#d62728",              # 紅
    "LFCC_GMM": "#9467bd",             # 紫
}

def analyze_false_positives(meta_df, misclassified_df):
    bonafide_total = (meta_df['label'] == 'bonafide').sum()
    fp_df = misclassified_df[(misclassified_df['label'] == 0) & (misclassified_df['pred'] == 1)].copy()

    # 建立虛擬分類群（因為 bonafide 不會有 TTS 名）
    fp_df['tts_system'] = 'bonafide'

    fp_count = fp_df.shape[0]
    fp_rate = fp_count / bonafide_total
    return {'bonafide': fp_rate}  # {tts: rate}



def plot_grouped_fn_bars(fn_rate_dict, tts_order, detector_names, save_path, title="False Positive Rates"):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    x = np.arange(len(tts_order))
    bar_width = 0.12
    total_width = bar_width * len(detector_names)

    plt.figure(figsize=(12, 6))

    for idx, detector in enumerate(detector_names):
        tts_rates = [fn_rate_dict.get(detector, {}).get(tts, 0) for tts in tts_order]
        positions = x - total_width / 2 + idx * bar_width + bar_width / 2

        color = DETECTOR_COLOR_MAP.get(detector, 'gray')
        bars = plt.bar(positions, tts_rates, width=bar_width, label=detector, color=color)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.005,
                f'{height:.2%}',
                ha='center',
                va='bottom',
                fontsize=8
            )

    plt.xticks(x, tts_order, rotation=45)
    plt.ylabel("False Positive Rate")
    plt.gca().yaxis.set_major_formatter(PercentFormatter(xmax=1.0))  # ✅ Y 軸也顯示百分比
    plt.ylim(0, 0.6)
    plt.title(title)
    plt.legend(title="Detector")
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved plot: {save_path}")


def main(args):
    task = args.task
    print(f"🔍 Analyzing false negatives for task: {task}")
    meta_path = f"F:/datasets/{task}/test/meta.csv"
    meta_df = pd.read_csv(meta_path)

    fp_rate_dict = {}  # detector_name -> {tts -> rate}

    for detector in DETECTOR_NAMES:
        misclassified_path = f"./logs/{detector}/misclassified_files_on_{task}.csv"
        if not os.path.exists(misclassified_path):
            print(f"⚠️  Missing file: {misclassified_path}")
            continue

        misclassified_df = pd.read_csv(misclassified_path)
        fp_rates = analyze_false_positives(meta_df, misclassified_df)
        fp_rate_dict[detector] = fp_rates

    save_dir = f"./plot/{task}/"
    os.makedirs(save_dir, exist_ok=True)

    plot_grouped_fn_bars(
        fp_rate_dict,
        TTS_ORDER,
        DETECTOR_NAMES,
        save_path=os.path.join(save_dir, "grouped_fp_bar.png"),
        title=f"False Positive Rates of Detectors on Spoofed TTS Samples ({task})"
    )


if __name__ == "__main__":
    from config.config import init
    args = init()
    main(args)
