import pandas as pd
import matplotlib.pyplot as plt
# 讀入資料
meta = pd.read_csv("F:/datasets/ASVspoof2021_DF/test/meta.csv")      # 你的 FN/FP 錯誤檔案
misclassified= pd.read_csv("misclassified_files.csv")               # 原始 meta 檔
meta['label'] = meta['label'].map({'bonafide': 0, 'spoof': 1})


print(f"Attack Types in meta:{meta['attack_type'].value_counts()}")

total_sample_counts = meta.shape[0]
print(f"Total samples in meta: {total_sample_counts}")
fake_sample_counts = meta[meta['label'] == 1].shape[0]
print(f"Total fake samples in meta: {fake_sample_counts}")
real_sample_counts = meta[meta['label'] == 0].shape[0]
print(f"Total real samples in meta: {real_sample_counts}")

# 用 filename 合併
merged = pd.merge(misclassified, meta, on="filename", how="left")
# 只針對 FN 做分析（假語音被判成真）
fn_df = merged[(merged['label_y'] == 1) & (merged['pred'] == 0)]
fn_counts = fn_df['attack_type'].value_counts()
tts_names = fn_df['attack_type'].unique()

for count, tts in zip(fn_counts, tts_names):
    print(f"TTS System: {tts}, False Negatives Rate: {count/fake_sample_counts:.4f}")
    print(f"Total False Negatives for {tts}: {count}")

fn_rates = fn_counts / fake_sample_counts

plt.figure(figsize=(8, 5))
plt.bar(tts_names, fn_rates, color=['skyblue', 'lightgreen', 'salmon', 'lightcoral', 'lightsalmon'])
plt.title('False Negative Rate per TTS System')
plt.xlabel('TTS System')
plt.ylabel('False Negative Rate')
plt.ylim(0, 0.1)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 只針對 FP 做分析（真語音被判成假）
fp_df = merged[(merged['label_y'] == 0) & (merged['pred'] == 1)]
fp_counts = fp_df['attack_type'].value_counts()
tts_names_fp = fp_df['attack_type'].unique()
for count, tts in zip(fp_counts, tts_names_fp):
    print(f"TTS System: {tts}, False Positives Rate: {count/real_sample_counts:.4f}")
    print(f"Total False Positives for {tts}: {count}")

fp_rates = fp_counts / real_sample_counts
plt.figure(figsize=(8, 5))
plt.bar(tts_names_fp, fp_rates, color=['skyblue', 'lightgreen', 'salmon', 'lightcoral', 'lightsalmon'])
plt.title('False Positive Rate per TTS System') 
plt.xlabel('TTS System')
plt.ylabel('False Positive Rate')   
plt.ylim(0, 0.1)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
