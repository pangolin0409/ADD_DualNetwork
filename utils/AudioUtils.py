import torch
import librosa
import numpy as np

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