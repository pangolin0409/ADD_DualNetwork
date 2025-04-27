import os
import torch
import h5py
import soundfile as sf
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, Queue, cpu_count
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
import torchaudio.functional as F
from queue import Empty

def worker_process(audio_queue: Queue, result_queue: Queue, sample_rate: int):
    print("[✅ Worker] Started.", flush=True)
    nb_samp = 64000  # 固定裁切長度（4秒）
    while True:
        item = audio_queue.get()
        if item is None:
            print("[❌ Worker] Received termination signal.", flush=True)
            break
        file_path = item
        try:
            waveform, sr = sf.read(file_path)
            if sr != sample_rate:
                waveform = F.resample(torch.tensor(waveform).float(), orig_freq=sr, new_freq=sample_rate).numpy()

            if waveform.size == 0:
                continue
            waveform = waveform / np.max(np.abs(waveform))
            waveform = waveform.reshape(1, -1)
            nb_time = waveform.shape[1]

            if nb_time > nb_samp:
                start_idx = np.random.randint(0, nb_time - nb_samp)
                waveform = waveform[:, start_idx:start_idx+nb_samp][0]
            elif nb_time < nb_samp:
                nb_dup = int(nb_samp / nb_time) + 1
                waveform = np.tile(waveform, (1, nb_dup))[:, :nb_samp][0]
            else:
                waveform = waveform[0]

            uid = os.path.splitext(os.path.basename(file_path))[0]
            result_queue.put((uid, waveform.astype(np.float32)))
        except Exception as e:
            print(f"[Worker Error] {file_path}: {e}")
            continue

def model_process(result_queue: Queue, output_queue: Queue, model_ckpt, target_layers, sample_rate):
    print("[✅ Model] Started.", flush=True)
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_ckpt)
    model = Wav2Vec2Model.from_pretrained(model_ckpt, output_hidden_states=True).to(device)
    model.eval()
    try:
        while True:
            item = result_queue.get()
            if item is None:
                print("[❌ Model] Received termination signal.", flush=True)
                break
            uid, waveform = item
            try:
                waveform = np.array(waveform, dtype=np.float32)
                inputs = processor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding=True)
                input_values = inputs.input_values.to(device)
                attention_mask = inputs.attention_mask.to(device)

                with torch.no_grad():
                    outputs = model(input_values, attention_mask=attention_mask)

                features = {}
                for layer_id in target_layers:
                    if layer_id >= len(outputs.hidden_states):
                        continue
                    hs = outputs.hidden_states[layer_id].squeeze(0).cpu().half().numpy()
                    features[f"layer_{layer_id}"] = hs

                output_queue.put((uid, features))
            except Exception as e:
                print(f"[Model Error] {uid}: {e}")
                continue
    finally:
        print("[✅ Model] Sending sentinel to output_queue", flush=True)
        output_queue.put(None)
        print("[✅ Model] Exiting model process", flush=True)
        return

def extract_and_save_features_shared_model(audio_folder, output_h5_path, model_ckpt, sample_rate, target_layers):
    audio_files = sorted([
        os.path.join(audio_folder, f)
        for f in os.listdir(audio_folder)
        if f.endswith(".wav") or f.endswith(".flac")
    ])

    num_worker = min(cpu_count(), 8)
    num_model = min(cpu_count(), 8)

    # 1. 創建 queue
    audio_queue = Queue(maxsize=1000)
    result_queue = Queue(maxsize=1000)
    output_queue = Queue(maxsize=1000)

    # 2. 啟動 model process
    model_processes = []
    for _ in range(num_model):
        p = Process(target=model_process, args=(result_queue, output_queue, model_ckpt, target_layers, sample_rate))
        p.start()
        model_processes.append(p)

    # 3. 啟動 worker process
    worker_processes = []
    for _ in range(num_worker):
        p = Process(target=worker_process, args=(audio_queue, result_queue, sample_rate))
        p.start()
        worker_processes.append(p)

    # (可選) 先啟動一個 Thread / Process 來消費 output_queue (或放在主線程) 
    # 但你目前的設計：主程式自己消費 output_queue → 下面就開始了

    # 4. 馬上開始消費 output_queue (不要等塞完路徑)
    #    開個 while True loop or 用 multiprocessing.Process 皆可 
    sentinel_count = 0
    saved_count = 0
    total_audio = len(audio_files)
    h5f = h5py.File(output_h5_path, "w")

    # 用一個 function 來消費 output_queue
    def consume_output_queue():
        nonlocal sentinel_count, saved_count
        pbar = tqdm(total=total_audio, desc="Saving features")
        while True:
            try:
                item = output_queue.get(timeout=5)
            except Empty:
                print("[⏳ Main] Waiting for model output...")
                continue

            if item is None:
                sentinel_count += 1
                if sentinel_count == num_model:
                    # 收到所有 model 的哨兵
                    break
                else:
                    continue

            # 正常資料
            uid, features = item
            grp = h5f.create_group(uid)
            for layer_name, data in features.items():
                grp.create_dataset(layer_name, data=data, compression="gzip")
            saved_count += 1
            pbar.update(1)
        pbar.close()

    # 開一個 Process 或直接在主線程跑也行
    # 這裡示範在主線程跑:
    from threading import Thread
    consumer_thread = Thread(target=consume_output_queue)
    consumer_thread.start()

    # 5. 塞音檔路徑到 audio_queue (生產)
    for i, path in enumerate(audio_files):
        audio_queue.put(path)
        if i % 100 == 0:
            print(f"[📥 AudioQ] Added {i}/{total_audio}")

    # 6. 塞完後給 worker 結束訊號
    for _ in range(num_worker):
        audio_queue.put(None)

    # 7. 等 worker 結束
    for p in worker_processes:
        p.join()

    # 8. worker 都結束後，再送哨兵給 model
    for _ in range(num_model):
        result_queue.put(None)

    # 9. 等 model 結束
    for p in model_processes:
        p.join()

    # 10. 等 output consumer 結束
    consumer_thread.join()

    h5f.close()
    print(f"[Done] Saved features: {saved_count} to {output_h5_path}")

if __name__ == '__main__':
    dataset_list = ['Asvspoof2019_LA', 'in_the_wild', 'ASVspoof2021_DF']
    part_ = ['test', 'validation', 'train']

    for dataset in dataset_list:
        for part in part_:
            base_folder = save_folder = os.path.join('E:/datasets/', dataset, part)
            audio_folder = os.path.join(base_folder, 'audio')
            output_h5_path = os.path.join(save_folder, f"wav2vec2_layers_192023_{part}.h5")

            extract_and_save_features_shared_model(
                audio_folder=audio_folder,
                output_h5_path=output_h5_path,
                model_ckpt="./pretrained_models/wav2vec2-xls-r-300m",
                sample_rate=16000,
                target_layers=[19, 20, 23]
            )
