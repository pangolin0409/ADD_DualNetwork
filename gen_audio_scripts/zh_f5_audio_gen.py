import os
import pandas as pd
import subprocess
import time
import soundfile as sf
from tqdm import tqdm
import sys

# ==== 參數設定 ====
REF_TEXT_PATH = "ref_text_clean.csv"
BASE_AUDIO_DIR = "F:/datasets/LibriSpeech/train/audio"
OUTPUT_DIR = "generated_audio_f5"
META_OUT_PATH = "F:\datasets\en-fbgkx-librispeech-2025_v1\meta.csv"
BATCH_SIZE = 50
SAMPLE_TARGET = 1
SLEEP_BETWEEN_BATCHES = 2

# ==== 確保輸出資料夾存在 ====
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 刪除不足 3 秒的音檔 ====
def is_too_short(path):
    try:
        f = sf.SoundFile(path)
        return len(f) / f.samplerate < 3.0
    except:
        return True

deleted = 0
# for f in os.listdir(OUTPUT_DIR):
#     if f.endswith(".wav"):
#         path = os.path.join(OUTPUT_DIR, f)
#         if is_too_short(path):
#             os.remove(path)
#             deleted += 1
print(f"🧹 已刪除 {deleted} 個短於 3 秒的音檔")

# ==== 計算還要補幾筆 ====
existing_files = {
    os.path.splitext(f)[0].replace("_f5tts", "")
    for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".wav")
}
remaining = SAMPLE_TARGET - len(existing_files)
print(f"🎯 目前已有 {len(existing_files)} 筆，需要補 {remaining} 筆")

if remaining <= 0:
    print("✅ 已經達到 1 萬筆，不需補資料")
    sys.exit(0)

# ==== 載入語料 ====
df = pd.read_csv(REF_TEXT_PATH)

# ==== 篩選對應 3~10 秒的原始語音檔案 ====
def is_duration_ok(row):
    fname = row["filename"]
    audio_path = os.path.join(BASE_AUDIO_DIR, fname)
    try:
        f = sf.SoundFile(audio_path)
        dur = len(f) / f.samplerate
        return 3.0 <= dur <= 10.0
    except:
        return False

df["valid_duration"] = df.apply(is_duration_ok, axis=1)
df = df[df["valid_duration"]].reset_index(drop=True)
print(f"✅ 共 {len(df)} 筆語料長度在 3~10 秒間")

# ==== 排除已生成的 ====
df["base_name"] = df["filename"].str.replace(".wav", "")
df = df[~df["base_name"].isin(existing_files)].reset_index(drop=True)
print(f"🧽 排除已生成，剩下 {len(df)} 筆可用")

if len(df) < remaining:
    print(f"⚠️ 可用語料只剩 {len(df)} 筆，不足 {remaining}，將全數使用")
    selected_df = df
else:
    selected_df = df.sample(n=remaining, random_state=42).reset_index(drop=True)

# ==== 分批處理 ====
meta_rows = []
total = len(selected_df)

print(f"🚀 開始處理 {total} 筆資料")

for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch_df = selected_df.iloc[start:end].reset_index(drop=True)

    print(f"\n🚀 處理 Batch {start}-{end - 1} ...")

    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {start}-{end - 1}"):
        ref_audio_name = row["filename"]
        ref_transcript = row["transcript"]
        ref_audio_path = os.path.join(BASE_AUDIO_DIR, ref_audio_name)

        base_name = os.path.splitext(ref_audio_name)[0]
        output_filename = f"{base_name}_f5tts.wav"
        out_wav_path = os.path.join(OUTPUT_DIR, output_filename)

        cmd = [
            "f5-tts_infer-cli",
            "--model", "F5TTS_v1_Base",
            "--ref_audio", f"\"{ref_audio_path}\"",
            "--ref_text", f"\"{ref_transcript}\"",
            "--gen_text", f"\"{ref_transcript}\"",
            "--output_file", f"\"{out_wav_path}\""
        ]

        try:
            subprocess.run(" ".join(cmd), shell=True, check=True)
        except subprocess.CalledProcessError as e:
            print(f"❌ 命令失敗：{e}. 跳過此檔案 {ref_audio_name}")
            continue

        meta_rows.append({
            "id": len(meta_rows) + 1,
            "label": "spoof",
            "reference_audio_filename": ref_audio_name,
            "tts": "f5-tts",
            "output_filename": output_filename
        })

    print(f"⏸️ 批次完成，等待 {SLEEP_BETWEEN_BATCHES} 秒...\n")
    time.sleep(SLEEP_BETWEEN_BATCHES)

# ==== 儲存 meta.csv ====
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(META_OUT_PATH, index=False)
print(f"\n✅ 全部完成！已生成 {len(meta_df)} 筆語音，meta.csv 已儲存。")
