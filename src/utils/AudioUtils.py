import math
import torch
import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from skimage.filters import gabor
from torch.nn import functional as F
def extract_mel_spectrogram(audio, sr=16000, n_mels=80, max_len_pad=192):
    if isinstance(audio, torch.Tensor):
        # 移到 CPU，並轉 numpy
        audio = audio.detach().cpu().numpy()

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels)
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # 填充或截斷到固定長度
    mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, max_len_pad - mel_spectrogram.shape[1])), 'constant', constant_values=0)
    mel_spectrogram = torch.tensor(mel_spectrogram).unsqueeze(0)
    return mel_spectrogram

# 如果需要提取基頻 (f0)
def extract_pitch(audio, sr=16000, n_fft=2048, hop_length=512, max_len_pad=192):
    y = audio.detach().cpu().numpy() if isinstance(audio, torch.Tensor) else audio
    pitch, _ = librosa.core.piptrack(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length)

    f0 = [np.max(frame) if np.max(frame) > 0 else 0 for frame in pitch.T]
    f0 = np.array(f0)
    f0 = np.pad(f0, (0, max_len_pad - len(f0)), 'constant', constant_values=0)

    f0_quantized = np.zeros((max_len_pad, 257))
    for i, f in enumerate(f0):
        bin_idx = min(int(f // 10), 256)  # 限制 bin_idx 的最大值為 256
        f0_quantized[i, bin_idx] = 1

    f0_tensor = torch.tensor(f0_quantized).unsqueeze(0).transpose(1, 2)
    return f0_tensor.float()  # 確保輸出類型為 float

def extract_DI_formants_with_praat_batch(
    audio_paths,          # 批量音檔路徑 (List[str])
    time_step=0.01,
    max_formant=5500,
    max_freq_for_map=6000,
    freq_resolution=50,
    target_freq_bins=50,
    target_time_bins=100
):
    """
    1) 對多個音檔進行 DI-Formants 特徵擷取流程：
       - 使用 Praat（Burg 方法）提取 F1, F2, F3 軌跡
       - 根據 F1, F2, F3 建立二值時頻圖
       - 對二值圖應用 Gabor 濾波，得到 4-channel 特徵，形狀 (4, f_bins, t_bins)
    2) 由於不同音檔可能導致 f_bins 或 t_bins 不一致，
       因此利用 interpolate 將每個結果調整為固定形狀 (4, target_freq_bins, target_time_bins)
    3) 最終回傳 tensor 形狀為 (B, 4, target_freq_bins, target_time_bins)
    """
    batch_di_formants = []

    for audio_path in audio_paths:
        # (A) 利用 Praat 提取 Formant Trajectories (F1, F2, F3)
        times, F1, F2, F3 = extract_formant_trajectories(
            audio_path, 
            time_step=time_step,
            max_formant=max_formant
        )
        # (B) 根據 F1, F2, F3 建立 binary 時頻圖
        binary_map = create_time_frequency_map(
            F1, F2, F3,
            max_freq=max_freq_for_map,
            freq_resolution=freq_resolution
        )
        # (C) 使用 Gabor 濾波，獲得 DI-Formants 特徵，shape 預期為 (4, f_bins, t_bins)
        di_formants_4ch = apply_gabor_filters(binary_map)
        # di_formants_4ch 可能各音檔 time_bins 不一致，例如有的為 120，有的為 100
        
        # (D) 利用 F.interpolate 調整 di_formants_4ch 至固定形狀 (4, target_freq_bins, target_time_bins)
        # 先轉成 tensor，並增加 batch 維度
        di_tensor = torch.tensor(di_formants_4ch, dtype=torch.float32).unsqueeze(0)  # shape: [1, 4, f_bins, t_bins]
        di_resized = F.interpolate(di_tensor, size=(target_freq_bins, target_time_bins), mode='bilinear', align_corners=False)
        di_resized = di_resized.squeeze(0)  # shape: [4, target_freq_bins, target_time_bins]
        
        # 轉回 numpy (或直接保留 tensor也可)
        batch_di_formants.append(di_resized.cpu().numpy())

    # 將所有音檔 DI-Formants 組合成一個 tensor，形狀 [B, 4, target_freq_bins, target_time_bins]
    batch_di_formants = np.stack(batch_di_formants, axis=0)
    batch_di_formants_tensor = torch.tensor(batch_di_formants, dtype=torch.float32)
    
    return batch_di_formants_tensor

def extract_formant_trajectories(audio_path, time_step=0.01, max_formant=5500, n_formants=5):
    """
    使用 Praat (Burg 演算法) 追蹤語音中的 formants。
    回傳 (times, F1, F2, F3)，皆為 numpy array。
    """
    sound = parselmouth.Sound(audio_path)
    
    # Praat: To Formant (burg)
    formant_object = sound.to_formant_burg(
        time_step=time_step,
        max_number_of_formants=n_formants,
        maximum_formant=max_formant,
        window_length=0.025,
        pre_emphasis_from=50
    )
    
    # 取得整段音檔的時間長度
    duration = sound.get_total_duration()
    times = np.arange(0, duration, time_step)
    
    # 只取前三條 formants (F1, F2, F3)
    F1_list, F2_list, F3_list = [], [], []
    
    for t in times:
        f1 = formant_object.get_value_at_time(formant_number=1, time=t) or 0.0
        f2 = formant_object.get_value_at_time(formant_number=2, time=t) or 0.0
        f3 = formant_object.get_value_at_time(formant_number=3, time=t) or 0.0
        
        F1_list.append(f1)
        F2_list.append(f2)
        F3_list.append(f3)
    
    F1 = np.array(F1_list, dtype=np.float32)
    F2 = np.array(F2_list, dtype=np.float32)
    F3 = np.array(F3_list, dtype=np.float32)
    
    return times, F1, F2, F3

def create_time_frequency_map(F1, F2, F3, time_step=0.01, max_freq=6000, freq_resolution=50):
    """
    將 (F1, F2, F3) 映射到二維 binary map:
    - 橫軸: 時間索引 (等同於 F1, F2, F3 的 frame)
    - 縱軸: 頻率 bins (0 ~ max_freq), freq_resolution決定 bin size
    """
    T = len(F1)  # 時間維度數量
    freq_bins = max_freq // freq_resolution  # e.g. 6000//50=120

    binary_map = np.zeros((freq_bins, T), dtype=np.float32)
    
    for t in range(T):
        for f_val in [F1[t], F2[t], F3[t]]:
            if f_val > 0:
                bin_idx = int(f_val // freq_resolution)
                if bin_idx < freq_bins:
                    binary_map[bin_idx, t] = 1.0
    return binary_map

def gabor_filter_2d(image, theta, lam=4, psi=0, sigma=2, gamma=0.5):
    """
    單方向 Gabor 濾波器
    """
    frequency = 1.0 / lam
    filtered, _ = gabor(image, frequency=frequency, theta=theta, sigma_x=sigma, sigma_y=sigma/gamma)
    return filtered

def apply_gabor_filters(binary_map):
    """
    分別以 4 個方向(0, π/4, π/2, 3π/4)做 Gabor 濾波，
    回傳 shape = (4, H, W)
    """
    lam = 4
    sigma = lam/2
    gamma = 0.5
    thetas = [0, math.pi/4, math.pi/2, 3*math.pi/4]

    filtered_channels = []
    for theta in thetas:
        out = gabor_filter_2d(
            image=binary_map,
            theta=theta,
            lam=lam,
            sigma=sigma,
            gamma=gamma
        )
        filtered_channels.append(out)
    
    # 堆疊成 (4, H, W)
    return np.stack(filtered_channels, axis=0)