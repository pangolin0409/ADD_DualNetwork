import os
import pandas as pd
import torch
import numpy as np
import random
import argparse
import scipy
from tqdm import tqdm
from transformers import AutoProcessor, BarkModel

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default="final_transcripts_gpt4.csv", help='Input CSV file with sentences')
parser.add_argument('--output_dir', type=str, default='generated_audio_bark', help='Directory to save audio')
parser.add_argument('--meta_csv', type=str, default='meta_bark_part.csv', help='Path to save metadata CSV')
args = parser.parse_args()

# Constants
SPEAKERS = [f"v2/en_speaker_{i}" for i in range(10)]
os.makedirs(args.output_dir, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Bark model and processor
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark").to(device)
sample_rate = model.generation_config.sample_rate

# Load only the first 80 rows
df = pd.read_csv(args.input)
df = df.head(80)

meta = []

for idx, row in tqdm(df.iterrows(), total=len(df)):
    sentence_id = row['sentence_id']
    category = row['category']
    transcript = row['transcript']

    speaker = random.choice(SPEAKERS)

    try:
        inputs = processor(transcript, voice_preset=speaker, return_tensors="pt")
        inputs['attention_mask'] = (inputs['input_ids'] != processor.tokenizer.pad_token_id).long()
        inputs = {k: v.to(device) for k, v in inputs.items()}

        audio_array = model.generate(**inputs).cpu().numpy().squeeze()
        audio_array = np.clip(audio_array, -1.0, 1.0)  # 避免爆音或失真

        filename = f"{sentence_id:04d}_bark.wav"
        filepath = os.path.join(args.output_dir, filename)
        scipy.io.wavfile.write(filepath, rate=sample_rate, data=audio_array)

        meta.append({
            "id": len(meta) + 1,
            "sentence_id": sentence_id,
            "category": category,
            "filename": filename,
            "speaker": speaker,
            "tts": "bark",
            "label": "spoof"
        })
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error at sentence_id={sentence_id}: {e}")

# Save meta CSV
pd.DataFrame(meta).to_csv(args.meta_csv, index=False)
print(f"✅ Finished generating {len(meta)} audio files.")
