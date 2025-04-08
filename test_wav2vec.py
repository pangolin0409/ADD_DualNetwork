import os
import torch
import h5py
import soundfile as sf
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio.functional as F
from queue import Empty

# 🧪 修改這個資料夾路徑
AUDIO_FOLDER = "./"
MODEL_CKPT = "./pretrained_models/wav2vec2-xls-r-300m"
OUTPUT_PATH = "./test_output.h5"

def worker(audio_queue, result_queue, sample_rate):
    print("[✅ Worker] Started")
    while True:
        item = audio_queue.get()
        if item is None:
            break
        file_path = item
        try:
            waveform, sr = sf.read(file_path)
            if sr != sample_rate:
                waveform = F.resample(torch.tensor(waveform).float(), orig_freq=sr, new_freq=sample_rate).numpy()

            if waveform.size == 0 or np.max(np.abs(waveform)) == 0:
                print(f"[⚠️ Skip] {file_path} is empty or silence")
                continue

            waveform = waveform / np.max(np.abs(waveform))
            waveform = waveform.reshape(1, -1)[0].astype(np.float32)

            uid = os.path.splitext(os.path.basename(file_path))[0]
            result_queue.put((uid, waveform))
            print(f"[📤 Worker] Sent {uid} to result_queue")
        except Exception as e:
            print(f"[❌ Worker Error] {file_path}: {e}")

def model_proc(result_queue, output_queue, model_ckpt, target_layers, sample_rate):
    print("[✅ Model] Started")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_ckpt)
    model = Wav2Vec2Model.from_pretrained(model_ckpt, output_hidden_states=True).cuda().eval()

    while True:
        item = result_queue.get()
        if item is None:
            break
        uid, waveform = item
        try:
            inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
            input_values = inputs.input_values.cuda()
            attention_mask = inputs.attention_mask.cuda()

            with torch.no_grad():
                outputs = model(input_values, attention_mask=attention_mask)

            features = {}
            for layer_id in target_layers:
                hs = outputs.hidden_states[layer_id].squeeze(0).cpu().half().numpy()
                features[f"layer_{layer_id}"] = hs

            output_queue.put((uid, features))
            print(f"[📤 Model] Put {uid} into output_queue")
        except Exception as e:
            print(f"[❌ Model Error] {uid}: {e}")
        finally:
            print("[✅ Model] Sending sentinel to output_queue", flush=True)
            output_queue.put(None)
            print("[✅ Model] Exiting model process", flush=True)
            return  # <--- 明確退出模型進程

    output_queue.put(None)

def main():
    sample_rate = 16000
    target_layers = [19, 20, 23]

    audio_files = sorted([
        os.path.join(AUDIO_FOLDER, f)
        for f in os.listdir(AUDIO_FOLDER)
        if f.endswith(".wav")
    ])[:1]  # 只取一筆

    if not audio_files:
        print("[❌] 沒有找到任何 .wav 檔案")
        return

    audio_queue = Queue()
    result_queue = Queue()
    output_queue = Queue()

    audio_queue.put(audio_files[0])
    audio_queue.put(None)

    w = Process(target=worker, args=(audio_queue, result_queue, sample_rate))
    m = Process(target=model_proc, args=(result_queue, output_queue, MODEL_CKPT, target_layers, sample_rate))

    w.start()
    m.start()

    with h5py.File(OUTPUT_PATH, "w") as h5f:
        while True:
            try:
                item = output_queue.get(timeout=10)
            except Empty:
                print("[⏳ Main] Waiting for model output...")
                continue
            finally:
                print("[✅ Model] Sending sentinel to output_queue", flush=True)
                output_queue.put(None)
            if item is None:
                print("[✅ Main] Received model sentinel, done.")
                break
            uid, features = item
            print(f"[✅ Save] Writing {uid}")
            grp = h5f.create_group(uid)
            for layer, data in features.items():
                grp.create_dataset(layer, data=data, compression="gzip")

    w.join()
    m.join()
    print("[✅ All Done]")

if __name__ == "__main__":
    # main()
    file_path = "E:\\datasets/Asvspoof2019_LA\\test\\audio\\LA_E_1903269.flac"  # 替換為你的檔案路徑
    try:
        waveform, sr = sf.read(file_path)
        print(f"[DEBUG] type={type(waveform)} | shape={getattr(waveform, 'shape', 'N/A')} | dtype={getattr(waveform, 'dtype', 'N/A')}")
        print(f"[DEBUG] waveform[:5] = {waveform[:5]}")
    except Exception as e:
        print(f"[❌ Error] {file_path}: {e}")
