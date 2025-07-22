import pandas as pd
import time
from tqdm import tqdm
import soundfile as sf
from GPT_SoVITS.inference_cli import synthesize

# 設定參數
GPT_MODEL = "./GPT_SoVITS/pretrained_models/s2Gv4.pth"
SOVITS_MODEL = "./GPT_SoVITS/pretrained_models/vocoder.pth"
REF_TEXT_PATH = "C:/Users/USER/Desktop/Gordon/ref_text_clean.csv"
OUTPUT_DIR = "C:/Users/USER/Desktop/Gordon/generated_audio_gpt_sovits"
META_OUT_PATH = "meta_gpt_sovits.csv"
BASE_AUDIO_DIR = "C:/Users/USER/Desktop/Gordon/datasets/LibriSpeech/train/audio"
TMP_TXT_DIR = "tmp_txt"
LANGUAGE = "英文"
TARGET_COUNT = 10000
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_TXT_DIR, exist_ok=True)

ref_df = pd.read_csv(REF_TEXT_PATH)
ref_df["base_name"] = ref_df["filename"].str.replace(".flac", "", regex=False)
ref_df = ref_df.sample(n=TARGET_COUNT, random_state=42).reset_index(drop=True)

meta_rows = []

for start in range(0, TARGET_COUNT, BATCH_SIZE):
    end = min(start + BATCH_SIZE, TARGET_COUNT)
    batch = ref_df.iloc[start:end].reset_index(drop=True)

    for idx, row in tqdm(batch.iterrows(), total=len(batch)):
        ref_audio_name = row["filename"]
        transcript = row["transcript"]
        base_name = row["base_name"]

        ref_audio_path = os.path.join(BASE_AUDIO_DIR, ref_audio_name)
        ref_text_path = os.path.join(TMP_TXT_DIR, f"{base_name}_ref.txt")
        target_text_path = os.path.join(TMP_TXT_DIR, f"{base_name}_target.txt")
        output_wav_path = os.path.join(OUTPUT_DIR, f"{base_name}_gptsovits.wav")

        if not os.path.exists(ref_audio_path):
            continue
        if os.path.exists(output_wav_path):
            continue

        with open(ref_text_path, "w", encoding="utf-8") as f:
            f.write(transcript)
        with open(target_text_path, "w", encoding="utf-8") as f:
            f.write(transcript)

        try:
            synthesize(
                GPT_model_path=GPT_MODEL,
                SoVITS_model_path=SOVITS_MODEL,
                ref_audio_path=ref_audio_path,
                ref_text_path=ref_text_path,
                ref_language=LANGUAGE,
                target_text_path=target_text_path,
                target_language=LANGUAGE,
                output_path=OUTPUT_DIR
            )
            os.rename(os.path.join(OUTPUT_DIR, "output.wav"), output_wav_path)

            meta_rows.append({
                "id": len(meta_rows) + 1,
                "label": "spoof",
                "reference_audio_filename": ref_audio_name,
                "tts": "gpt-sovits",
                "output_filename": f"{base_name}_gptsovits.wav"
            })

        except Exception as e:
            print(f"生成失敗：{e}")
            continue

    meta_df = pd.DataFrame(meta_rows)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, META_OUT_PATH), index=False)
    time.sleep(SLEEP_BETWEEN_BATCHES)

print(f"✅ 完成所有生成，共 {len(meta_rows)} 筆語音")
