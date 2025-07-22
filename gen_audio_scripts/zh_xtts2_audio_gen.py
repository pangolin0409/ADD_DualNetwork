import os
import pandas as pd
import time
import numpy as np
import torch
import soundfile as sf
from tqdm import tqdm
from TTS.api import TTS

# ==== XTTS v2 批次生成設定 ====
REF_TEXT_PATH = "F:/datasets/AISHELL-3/test/cleaned_aishell3_transcript.csv"
OUTPUT_DIR = "F:/datasets/generated_audio_xtts2_zh/audio"
META_OUT_PATH = "F:/datasets/generated_audio_xtts2_zh/meta_xtts2_zh.csv"
BASE_AUDIO_DIR = "F:/datasets/AISHELL-3/test/audio"
BATCH_SIZE = 50
SAMPLE_SIZE = 10000
SLEEP_BETWEEN_BATCHES = 2  # 秒
LANGUAGE = "zh-cn"

# ==== 建立資料夾 ====
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==== 載入資料 ====
ref_df = pd.read_csv(REF_TEXT_PATH)

# ==== 檢查已生成的檔案 ====
generated_files = {
    os.path.splitext(f)[0].replace("_xtts2", "")
    for f in os.listdir(OUTPUT_DIR)
    if f.endswith(".wav")
}

ref_df = ref_df[~ref_df["filename"].str.replace(".wav", "").isin(generated_files)].reset_index(drop=True)
if len(generated_files) > SAMPLE_SIZE:
    print(f"⚠️ 已有超過 {SAMPLE_SIZE} 筆資料，不需生成更多。")
    exit()

SAMPLE_SIZE -= len(generated_files)
ref_df = ref_df.sample(n=SAMPLE_SIZE, random_state=42).reset_index(drop=True)
total = len(ref_df)

# ==== 載入 XTTS 模型 ====
os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2")
tts.to("cuda" if torch.cuda.is_available() else "cpu")

# ==== 開始批次處理 ====
meta_rows = []
print(f"🔍 準備生成 {total} 筆資料")

for start in range(0, total, BATCH_SIZE):
    end = min(start + BATCH_SIZE, total)
    batch = ref_df.iloc[start:end].reset_index(drop=True)
    print(f"\n🚀 處理 Batch {start}-{end - 1} ...")

    for idx, row in tqdm(batch.iterrows(), total=len(batch), desc=f"Batch {start}-{end - 1}"):
        fname = row["filename"]
        transcript = row["transcript"]
        ref_audio_path = os.path.join(BASE_AUDIO_DIR, fname)
        base_name = os.path.splitext(fname)[0]
        output_filename = f"{base_name}_xtts2.wav"
        out_path = os.path.join(OUTPUT_DIR, output_filename)

        if not os.path.exists(ref_audio_path):
            print(f"❌ 找不到檔案：{ref_audio_path}，跳過")
            continue

        try:
            tts.tts_to_file(
                text=transcript,
                speaker_wav=ref_audio_path,
                language=LANGUAGE,
                file_path=out_path
            )

            meta_rows.append({
                "id": len(meta_rows) + 1,
                "label": "spoof",
                "reference_audio_filename": fname,
                "tts": "xtts-v2",
                "output_filename": output_filename
            })
        except Exception as e:
            print(f"❌ 生成失敗：{e}，跳過 {fname}")
            continue

    # 每批等待
    time.sleep(SLEEP_BETWEEN_BATCHES)

# ==== 儲存 meta 資訊 ====
meta_df = pd.DataFrame(meta_rows)
meta_df.to_csv(META_OUT_PATH, index=False)
print(f"\n✅ 全部完成！共生成 {len(meta_df)} 筆資料，meta.csv 已儲存。")
