from utils.AudioUtils import extract_formant_trajectories, create_time_frequency_map, apply_gabor_filters
def extract_DI_formants_with_praat(
    audio_path,
    time_step=0.01,
    max_formant=5500,
    max_freq_for_map=6000,
    freq_resolution=50
):
    """
    1) 利用 Praat (Burg) 拿到 F1,F2,F3 vs. time
    2) 轉成 binary time-frequency map
    3) 進行 Gabor 濾波 (4方向)
    回傳 shape = (4, freq_bins, time_bins)
    """
    # (A) Formant Trajectories
    times, F1, F2, F3 = extract_formant_trajectories(
        audio_path, 
        time_step=time_step,
        max_formant=max_formant
    )
    # (B) 建立 binary map
    binary_map = create_time_frequency_map(
        F1, F2, F3,
        max_freq=max_freq_for_map,
        freq_resolution=freq_resolution
    )
    # binary_map.shape = (freq_bins, T)

    # (C) Gabor 濾波
    di_formants_4ch = apply_gabor_filters(binary_map)
    # di_formants_4ch.shape = (4, freq_bins, time_bins)

    return di_formants_4ch

if __name__ == "__main__":
    audio_path = "demo.wav"
    di_formants = extract_DI_formants_with_praat(audio_path)
    print("DI-Formants shape:", di_formants.shape)
    # (4, freq_bins, time_bins)
    
    # 如果要送到 PyTorch 模型，可以這樣轉:
    import torch
    di_formants_tensor = torch.from_numpy(di_formants).unsqueeze(0)  
    print("DI-Formants Tensor shape:", di_formants_tensor.shape)
    # shape = (batch=1, 4, freq_bins, time_bins)
    # 後面就能餵給 CNN or 其它深度模型。