import os
import pandas as pd
import time
from tqdm import tqdm
import soundfile as sf
from GPT_SoVITS.inference_cli import synthesize

# === Helper Functions ===
def duration_in_range(audio_path):
    try:
        f = sf.SoundFile(audio_path)
        duration = len(f) / f.samplerate
        return 3.0 <= duration <= 10.0
    except Exception as e:
        print(f"⚠️ 無法讀取音檔 {audio_path}：{e}")
        return False

def append_to_finished(filename, finished_path="finished.txt"):
    with open(finished_path, "a", encoding="utf-8") as f:
        f.write(filename + "\n")

def is_finished(filename, finished_set):
    return filename in finished_set

# === Config ===
GPT_MODEL = "./GPT_SoVITS/pretrained_models/s1v3.ckpt"
SOVITS_MODEL = "./GPT_SoVITS/pretrained_models/vocoder.pth"
REF_TEXT_PATH = "F:/datasets/AISHELL-3/test/cleaned_aishell3_transcript.csv"
OUTPUT_DIR = "F:/datasets/generated_audio_gpt_sovits_zh/audio"
META_OUT_PATH = "F:/datasets/generated_audio_gpt_sovits_zh/meta_gpt_sovits_zh.csv"
FINISHED_PATH = "finished.txt"
BASE_AUDIO_DIR = "F:/datasets/AISHELL-3/test/audio"
TMP_TXT_DIR = "tmp_txt"
LANGUAGE = "中文"
TARGET_COUNT = 10000
BATCH_SIZE = 10
SLEEP_BETWEEN_BATCHES = 2

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(TMP_TXT_DIR, exist_ok=True)

# === Load finished files if exists ===
if os.path.exists(FINISHED_PATH):
    with open(FINISHED_PATH, "r", encoding="utf-8") as f:
        finished_set = set(line.strip() for line in f.readlines())
else:
    finished_set = set()

# === Load and filter data ===
ref_df = pd.read_csv(REF_TEXT_PATH)
print(f"📖 開始讀取參考語料：{REF_TEXT_PATH}")

ref_df["full_path"] = ref_df["filename"].apply(lambda x: os.path.join(BASE_AUDIO_DIR, x))
ref_df = ref_df[ref_df["full_path"].apply(duration_in_range)]

# remove already finished
ref_df = ref_df[~ref_df["filename"].isin(finished_set)].reset_index(drop=True)

print(f"✅ 有效音檔數量：{len(ref_df)}")

# # sample
ref_df["base_name"] = ref_df["filename"].str.replace(".flac", "", regex=False)
ref_df = ref_df.sample(n=min(TARGET_COUNT, len(ref_df)), random_state=42).reset_index(drop=True)

# === Processing loop ===
for start in range(0, len(ref_df), BATCH_SIZE):
    end = min(start + BATCH_SIZE, len(ref_df))
    batch = ref_df.iloc[start:end].reset_index(drop=True)

    for idx, row in tqdm(batch.iterrows(), total=len(batch), desc=f"處理中 {start}-{end}"):
        ref_audio_name = row["filename"]
        if is_finished(ref_audio_name, finished_set):
            continue

        transcript = row["transcript"]
        base_name = row["base_name"]
        ref_audio_path = os.path.join(BASE_AUDIO_DIR, ref_audio_name)
        ref_text_path = os.path.join(TMP_TXT_DIR, f"{base_name}_ref.txt")
        target_text_path = os.path.join(TMP_TXT_DIR, f"{base_name}_target.txt")
        output_wav_path = os.path.join(OUTPUT_DIR, f"{base_name}_gptsovits.wav")

        if not os.path.exists(ref_audio_path):
            print(f"❌ 找不到音檔 {ref_audio_path}")
            continue
        if os.path.exists(output_wav_path):
            print(f"⏭ 已存在音檔 {output_wav_path}")
            append_to_finished(ref_audio_name)
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

            # append to meta file
            meta_row = pd.DataFrame([{
                "id": "",  # 可選擇不編 id
                "label": "spoof",
                "reference_audio_filename": ref_audio_name,
                "tts": "gpt-sovits",
                "output_filename": f"{base_name}_gptsovits.wav"
            }])
            meta_row.to_csv(META_OUT_PATH, mode="a", index=False, header=not os.path.exists(META_OUT_PATH))

            # mark as finished
            append_to_finished(ref_audio_name)
            finished_set.add(ref_audio_name)

        except Exception as e:
            print(f"❌ 生成失敗 [{ref_audio_name}]：{e}")
            raise

    time.sleep(SLEEP_BETWEEN_BATCHES)

print(f"🏁 所有音檔處理完畢。成功數：{len(finished_set)}")
