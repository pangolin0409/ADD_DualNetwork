@echo off
call conda activate gordon

REM === 定義模型與資料集 ===
set MODELS= SLS WAV2VEC_MLP ASSIST RawNet2
@REM set MODELS=SLS WAV2VEC_MLP ASSIST
set TASKS=zh-fbgkx-aishell3-2025_v1 en-fbgkx-librispeech-2025_v1

for %%M in (%MODELS%) do (
    for %%T in (%TASKS%) do (
        echo ---------------------------------------
        echo 🚀 Running inference: %%M on %%T
        echo ---------------------------------------
        python main.py --experiment fp_fn_anaysis --task %%T --model_name %%M
        timeout /t 3 >nul
    )
)

echo ✅ All models and datasets completed.
pause