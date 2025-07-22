import os
import random
import pandas as pd
import soundfile as sf
from tqdm import tqdm
from kokoro import KPipeline

# ==== Step 0: 初始化 ====
input_csv_path = "./cleaned_aishell3_transcript.csv"
output_dir = "./datasets/generated_audio_kokoro_zh"
os.makedirs(output_dir, exist_ok=True)

# Kokoro 中文模型與說話人配置
pipeline = KPipeline(lang_code='z')
speakers = [
    "zf_xiaobei", "zf_xiaoni", "zf_xiaoxiao", "zf_xiaoyi",
    "zm_yunjian", "zm_yunxi", "zm_yunxia", "zm_yunyang"
]

# ==== Step 1: 刪除不足 3 秒的音檔 ====
def is_too_short(path):
    try:
        f = sf.SoundFile(path)
        return len(f) / f.samplerate < 3.0
    except:
        return True

deleted_count = 0
for fname in os.listdir(output_dir):
    if fname.endswith(".wav"):
        path = os.path.join(output_dir, fname)
        if is_too_short(path):
            os.remove(path)
            deleted_count += 1

print(f"🧹 Deleted {deleted_count} short audio files (<3s).")

# ==== Step 2: 決定需要補多少筆 ====
existing_wavs = [f for f in os.listdir(output_dir) if f.endswith(".wav")]
remaining = 10000 - len(existing_wavs)
print(f"🎯 Need to generate {remaining} more audios.")

if remaining <= 0:
    print("✅ 已經有 1 萬筆，不需要補資料")
    exit()

# ==== Step 3: 載入語料並篩選 3-10 秒音檔對應的 transcript ====
df = pd.read_csv(input_csv_path)

# 計算原始對應音檔長度，假設你有一個音檔資料夾 ./datasets/aishell3_wav
def is_duration_ok(row):
    fname = row["filename"]
    wav_path = os.path.join("./datasets/AISHELL-3/test/audio", fname)
    try:
        f = sf.SoundFile(wav_path)
        dur = len(f) / f.samplerate
        return 3.0 <= dur <= 10.0
    except Exception as e:
        raise e

df["is_ok"] = df.apply(is_duration_ok, axis=1)
df_filtered = df[df["is_ok"]].reset_index(drop=True)
finished_set = set()
for fname in os.listdir(output_dir):
    if fname.endswith(".wav") and "_kokoro" in fname:
        finished_set.add(fname)  # 加上完整檔名比較安全


df_filtered = df_filtered[~df_filtered["filename"].apply(lambda x: f"{os.path.splitext(x)[0]}_kokoro.wav" in finished_set)]

if len(df_filtered) < remaining:
    print(f"⚠️ 可用的語料只有 {len(df_filtered)}，不夠補滿 {remaining} 筆")
else:
    df_filtered = df_filtered.sample(n=remaining, random_state=42)
# ==== Step 4: 開始生成音檔 ====
meta = []

for idx, row in tqdm(df_filtered.iterrows(), total=len(df_filtered)):
    text = row["transcript"]
    base_name = os.path.splitext(row["filename"])[0]
    speaker = random.choice(speakers)

    try:
        generator = pipeline(text, voice=speaker)
        for i, (_, _, audio) in enumerate(generator):
            filename = f"{base_name}_kokoro.wav"
            out_path = os.path.join(output_dir, filename)
            sf.write(out_path, audio, 24000)

            meta.append({
                "id": len(meta) + 1,
                "filename": filename,
                "speaker": speaker,
                "tts": "kokoro",
                "label": "spoof"
            })
            break  # 儲存第一段即可
    except Exception as e:
        print(f"❌ Failed at {base_name}: {e}")

# ==== Step 5: 儲存 meta 資訊 ====
pd.DataFrame(meta).to_csv(os.path.join(output_dir, "meta_kokoro.csv"), index=False)
print(f"✅ Finished generating {len(meta)} audios.")
