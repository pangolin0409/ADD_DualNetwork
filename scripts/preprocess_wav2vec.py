import os 
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from src.data.dataloader import PreprocessRawAudio  # 你的自定義Dataset

# CUDA 加速設置
torch.backends.cudnn.benchmark = True   # 對固定大小輸入加速
torch.backends.cudnn.deterministic = False  # 配合 benchmark 使用

def extract_representation(dataset_name, sample_rate=16000, batch_size=128):
    """
    將 dataset_name 下 train/validation/test 三個分割的音檔，批量處理成 wav2vec 特徵並儲存.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 載入 wav2vec2
    model_ckpt = './pretrained_models/wav2vec2-xls-r-300m'
    processor = Wav2Vec2FeatureExtractor.from_pretrained(model_ckpt, output_hidden_states=True)
    model = Wav2Vec2Model.from_pretrained(model_ckpt, output_hidden_states=True).to(device)
    model.eval()

    for part_ in ['test', 'train', 'validation']:
        # 1) 建立 Dataset
        raw_audio_dataset = PreprocessRawAudio(
            path_to_database=f'../datasets/{dataset_name}',
            meta_csv='meta.csv',
            nb_samp=64600,
            part=part_,
        )

        # 2) 建立 DataLoader，批量處理
        data_loader = DataLoader(
            raw_audio_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=24,      # i9-12代: 16 到 32 之間
            pin_memory=True,     # 加速 CPU 到 GPU 的資料拷貝
            drop_last=False
        )

        # 3) 建立輸出資料夾
        target_dir = os.path.join(f'F:/datasets/{dataset_name}', part_, 'wav2vec2')
        os.makedirs(target_dir, exist_ok=True)

        print(f"Start processing {dataset_name} - {part_} ...")

        for batch in tqdm(data_loader, desc=f"[{dataset_name}/{part_}]"):
            waveforms, labels, file_names = batch
            
            waveforms_list = [w.cpu().numpy() for w in waveforms]

            # 5) 送進 processor
            inputs = processor(
                waveforms_list,
                sampling_rate=sample_rate,
                return_tensors="pt",
                padding=True
            )
            input_values = inputs.input_values.to(device, non_blocking=True)
            attention_mask = inputs.attention_mask.to(device, non_blocking=True)

            # 6) forward 到 wav2vec model
            with torch.no_grad():
                with torch.amp.autocast(device_type="cuda"):  # 使用正確的 AMP 寫法
                    outputs = model(input_values, attention_mask=attention_mask)

            
            hidden_states = outputs.last_hidden_state  # => [B, T', 1024]
            print(f'{hidden_states.shape}')
            # 7) 分別存檔
            B = hidden_states.size(0)
            for i in range(B):
                feat_i = hidden_states[i].cpu()
                fname_i = file_names[i]
                base_name = os.path.splitext(fname_i)[0]
                out_path = os.path.join(target_dir, f"{base_name}.pt")

                torch.save(feat_i, out_path)

        print(f"Done with {part_} sets from {dataset_name}")

if __name__ == '__main__':
    dataset_list = ['Asvspoof2019_LA']
    for dataset_name in dataset_list:
        extract_representation(dataset_name=dataset_name, sample_rate=16000, batch_size=64)
