import math
import torch
import librosa
import numpy as np
import parselmouth
from parselmouth.praat import call
from skimage.filters import gabor
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


def extract_formant_trajectories(audio_path, time_step=0.01, max_formant=5500, n_formants=5):
    """
    使用 Praat (Burg 演算法) 追蹤語音中的 formants。
    回傳 (times, F1, F2, F3)，皆為 numpy array。
    
    參數說明:
    - time_step: 計算 formant 時的時間間距 (秒)，如 0.01 即每 10ms 估計一次
    - max_formant: 最高 formant 頻率上限，女性或童聲建議可更高 (5500~6000)
    - n_formants: 預設分析幾條 formants (此處先設 5，實際可視需求調)
    """
    sound = parselmouth.Sound(audio_path)
    
    # Praat: To Formant (burg)
    #   參數: 
    #       1) time_step   2) max_number_of_formants 
    #       3) max_formant 4) window_length
    #       5) pre_emphasis_from
    formant_object = sound.to_formant_burg(
        time_step=time_step,
        max_number_of_formants=n_formants,
        maximum_formant=max_formant,  # ← 改為 maximum_formant
        window_length=0.025,
        pre_emphasis_from=50
    )


    # 取得整段音檔的時間長度
    duration = sound.get_total_duration()
    times = np.arange(0, duration, time_step)
    
    # 只取前三條 formants (F1, F2, F3)
    F1_list, F2_list, F3_list = [], [], []
    
    for t in times:
        # 該時間點落在哪一個 frame
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
            if f_val > 0:  # 如果沒估到可能是0
                bin_idx = int(f_val // freq_resolution)
                if bin_idx < freq_bins:
                    binary_map[bin_idx, t] = 1.0
    return binary_map

def gabor_filter_2d(image, theta, lam=4, psi=0, sigma=2, gamma=0.5):
    """
    單方向 Gabor 濾波器:
    - lam: wavelength (4)
    - theta: [0, π/4, π/2, 3π/4]
    - psi: phase offset (0)
    - sigma: Gaussian envelope (lambda/2)
    - gamma: aspect ratio (0.5)
    
    若 skimage.filters.gabor 不可用，可自行實作
    """
    # frequency = 1 / lam
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
