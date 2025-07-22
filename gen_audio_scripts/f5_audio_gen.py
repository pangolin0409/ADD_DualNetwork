import pandas as pd
import subprocess
import os
import random
import time
from tqdm import tqdm

# ==== 參數設定 ====
REF_TEXT_PATH = "ref_text_clean.csv"
OUTPUT_DIR = "generated_audio_f5"
META_OUT_PATH = "meta.csv"
BASE_AUDIO_DIR = "F:/datasets/LibriSpeech/train/audio"
BATCH_SIZE = 1
SAMPLE_SIZE = 1
SLEEP_BETWEEN_BATCHES = 2  # 秒

# ==== 確保輸出資料夾存在 ====
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 載入資料 ====
ref_df = pd.read_csv(REF_TEXT_PATH)
ref_df = ref_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)  # 隨機打亂資料
total = len(ref_df)
print(f"🔍 讀取 {total} 筆資料，開始處理...")
# ==== 建立 meta 資料 ====
meta_rows = []

# ==== 分批處理 ====
for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch_gen = ref_df.iloc[start:end].reset_index(drop=True)

    print(f"\n🚀 處理 Batch {start}-{end - 1} ...")

    for idx, row in tqdm(batch_gen.iterrows(), total=len(batch_gen), desc=f"Batch {start}-{end - 1}"):
        # 對應 reference audio
        ref_audio_name = batch_gen.loc[idx, "filename"]
        ref_transcript = batch_gen.loc[idx, "transcript"]
        speaker_id, chapter_id, *_ = ref_audio_name.replace(".flac", "").split("-")
        ref_audio_path = os.path.join(BASE_AUDIO_DIR, ref_audio_name)

        if not os.path.exists(ref_audio_path):
            print(f"❌ 找不到檔案：{ref_audio_path}. 跳過此檔案")
            continue

        # 輸出檔名：sentence_id + ref_audio_name
        output_filename = f"{ref_audio_name.replace('.flac', '')}_f5tts.wav"
        out_wav_path = os.path.join(OUTPUT_DIR, output_filename)

        # f5-tts 命令
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

        # 記錄 meta row
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
META_OUT_PATH = os.path.join(OUTPUT_DIR, META_OUT_PATH)
meta_df.to_csv(META_OUT_PATH, index=False)
print(f"\n✅ 全部完成！已生成 {len(meta_df)} 筆語音，meta.csv 已儲存。")
