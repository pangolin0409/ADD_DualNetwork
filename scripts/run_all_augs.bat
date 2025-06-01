@echo off
call conda activate audio
setlocal EnableDelayedExpansion

:: 設定參數組
set "GROUPS=basic_noise white_noise_variants environmental_noise volume_shift time_warp resample fade_effects echo_effects time_shift combo_stable combo_plus_resample combo_echoes full_combo"
set "TASKS=in_the_wild ASVspoof2021_DF SOTA Asvspoof2019_LA"

:: 每組 augmentation group 做一次訓練 + 多個推理
for %%G in (%GROUPS%) do (
    set MODEL=DAUG_%%G
    echo ============================
    echo 🔧 Training with group: %%G
    echo ============================

    python main.py --experiment train --model_name !MODEL! --aug_group %%G

    for %%T in (%TASKS%) do (
        echo Running inference: %%T with model !MODEL!
        python main.py --experiment inference --task %%T --model_name !MODEL! --checkpt_name checkpt_epoch_5.pth
        timeout /t 5 >nul
    )
)

echo === All groups and tasks done ===
pause
