@echo off
call conda activate gordon

REM === 定義模型與資料集 ===
set MODELS= SLS
@REM set MODELS=SLS WAV2VEC_MLP ASSIST  
set TASKS=zh-fbgkx-aishell3-2025_v1 en-fbgkx-librispeech-2025_v1 ASVspoof2021_DF

REM === 依序訓練每個模型 ===
@REM for %%M in (%MODELS%) do (
@REM     echo ===============================
@REM     echo 🔧 Training model: %%M
@REM     echo ===============================
@REM     python main.py --experiment baseline_train --model_name %%M
@REM )
REM === 對每個模型測試所有資料集 ===

for %%M in (%MODELS%) do (
    for %%T in (%TASKS%) do (
        echo ---------------------------------------
        echo 🚀 Running inference: %%M on %%T
        echo ---------------------------------------
        python main.py --experiment baseline_inference --task %%T --model_name %%M
        timeout /t 3 >nul
    )
)

echo ✅ All models and datasets completed.
pause