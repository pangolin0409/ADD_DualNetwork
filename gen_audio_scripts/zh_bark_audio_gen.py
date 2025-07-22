import os
import pandas as pd
import time
import numpy as np
import soundfile as sf
from tqdm import tqdm
from transformers import AutoProcessor, BarkModel
import torch
import random
import sys

# ==== 參數設定 ====
REF_TEXT_PATH = "F:/datasets/AISHELL-3/test/cleaned_aishell3_transcript.csv"
OUTPUT_DIR = "F:/datasets/generated_audio_bark/audio"
META_OUT_PATH = "F:/datasets/generated_audio_bark/meta_bark.csv"
SAMPLE_TARGET = 10000
BATCH_SIZE = 50
SLEEP_BETWEEN_BATCHES = 2
BARK_SPEAKERS = [f"v2/zh_speaker_{i}" for i in range(10)]

# ==== 初始化 Bark ====
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to(device)
sample_rate = model.generation_config.sample_rate

# ==== 準備資料 ====
os.makedirs(OUTPUT_DIR, exist_ok=True)

existing_files = {
    os.path.splitext(f)[0].replace("_bark", "")
    for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".wav")
}
remaining = SAMPLE_TARGET - len(existing_files)
print(f"🎯 已有 {len(existing_files)} 筆，需要補 {remaining} 筆")

if remaining <= 0:
    print("✅ 數量達標，不需補資料")
    sys.exit(0)

df = pd.read_csv(REF_TEXT_PATH)

# ==== 篩選長度在 3~10 秒之間的原始音檔 ====
def is_duration_ok(row):
    audio_path = os.path.join("F:/datasets/AISHELL-3/test/audio", row["filename"])
    try:
        with sf.SoundFile(audio_path) as f:
            dur = len(f) / f.samplerate
            return 3.0 <= dur <= 10.0
    except:
        return False

df["valid_duration"] = df.apply(is_duration_ok, axis=1)
df = df[df["valid_duration"]].reset_index(drop=True)

# 排除已產生過的語音
df["base_name"] = df["filename"].str.replace(".wav", "")
df = df[~df["base_name"].isin(existing_files)].reset_index(drop=True)

if len(df) < remaining:
    print(f"⚠️ 可用語料只有 {len(df)} 筆，不足 {remaining}，將全部使用")
    selected_df = df
else:
    selected_df = df.sample(n=remaining, random_state=42).reset_index(drop=True)

# ==== Bark 語音生成 ====
meta_rows = []
total = len(selected_df)
print(f"🚀 開始處理 {total} 筆資料")

for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch_df = selected_df.iloc[start:end].reset_index(drop=True)
    print(f"\n🚀 處理 Batch {start}-{end - 1} ...")

    for idx, row in tqdm(batch_df.iterrows(), total=len(batch_df), desc=f"Batch {start}-{end - 1}"):
        transcript = row["transcript"]
        base_name = os.path.splitext(row["filename"])[0]
        output_filename = f"{base_name}_bark.wav"
        out_wav_path = os.path.join(OUTPUT_DIR, output_filename)
        speaker = random.choice(BARK_SPEAKERS)

        try:
            inputs = processor(transcript, voice_preset=speaker, return_tensors="pt")
            inputs['attention_mask'] = (inputs['input_ids'] != processor.tokenizer.pad_token_id).long()
            inputs = {k: v.to(device) for k, v in inputs.items()}

            audio_array = model.generate(**inputs).cpu().numpy().squeeze()
            audio_array = np.clip(audio_array, -1.0, 1.0)
            sf.write(out_wav_path, audio_array, samplerate=sample_rate)

            meta_rows.append({
                "id": len(meta_rows) + 1,
                "label": "spoof",
                "reference_audio_filename": row["filename"],
                "tts": "bark",
                "speaker": speaker,
                "output_filename": output_filename
            })

        except Exception as e:
            print(f"❌ 生成失敗：{e}，跳過 {row['filename']}")
            continue

    print(f"⏸️ 批次完成，暫停 {SLEEP_BETWEEN_BATCHES} 秒...\n")
    time.sleep(SLEEP_BETWEEN_BATCHES)

# ==== 儲存 meta.csv ====
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(META_OUT_PATH, index=False)
print(f"\n✅ Bark 全部完成！已生成 {len(meta_df)} 筆語音，meta 已儲存。")
