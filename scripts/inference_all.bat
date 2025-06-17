@echo off
call conda activate audio
set MODEL=LAYER_TIME_DAUL_BRANCH_DA

python main.py --experiment train --model_name "%MODEL%"

set TASKS="en-fbgkx-librispeech-2025_v1" "in_the_wild" "ASVspoof2021_DF" "ADD" "zh-fbgkx-aishell3-2025_v1"
@REM set TASKS="ADD" "zh-fbgkx-aishell3-2025_v1" "Asvspoof2019_LA" 

for %%T in (%TASKS%) do (
    echo Running task: %%~T
    python main.py --experiment inference --task %%~T --model_name "%MODEL%" --checkpt_name checkpt_epoch_5.pth
    echo Task %%~T finished. Cleaning up...
    timeout /t 10 >nul
)

echo All tasks completed.
pause
