import pandas as pd
import matplotlib.pyplot as plt

# 載入誤判紀錄（FN 和 FP）
df = pd.read_csv("misclassified_files_on_en-fbgkx-librispeech-2025_v1.csv")

# 只針對 FN 做分析（假語音被判成真）
fn_df = df[(df['label'] == 1) & (df['pred'] == 0)]

# 從檔名中提取 TTS 名稱，例如 xxx_xtts.wav → xtts
fn_df['tts_system'] = fn_df['filename'].str.split('_').str[1].str.replace('.wav', '')


# 計算每個 TTS 的誤判數（False Negatives）
fn_counts = fn_df['tts_system'].value_counts()
tts_names = fn_df['tts_system'].unique()
print(tts_names)
total = 50000
tts_systems = []
for count, tts in zip(fn_counts, tts_names):
    print(f"TTS System: {tts}, False Negatives Rate: {count/total:.4f}")
    print(f"Total False Negatives for {tts}: {count}")

fn_counts = fn_counts /50000
plt.figure(figsize=(8, 5))
plt.bar(tts_names, fn_counts, color=['skyblue', 'lightgreen', 'salmon', 'lightcoral', 'lightsalmon'])
plt.title('False Negative Rate per TTS System')
plt.xlabel('TTS System')
plt.ylabel('False Negative Rate')
plt.ylim(0, 0.07)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 只針對 FP 做分析（真語音被判成假）
fp_df = df[(df['label'] == 0) & (df['pred'] == 1)]
# 從檔名中提取 TTS 名稱，例如 xxx_xtts.wav → xtts
fp_counts = fp_df.shape[0]
print(f"Total False Positives: {fp_counts}")
fp_rates = fp_counts / 25000

plt.figure(figsize=(8, 5))
plt.bar("boafide", fp_rates, color=['skyblue'])
plt.title('False Positive Rate per TTS System')
plt.xlabel('TTS System')
plt.ylabel('False Positive Rate')
plt.ylim(0, 0.1)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
