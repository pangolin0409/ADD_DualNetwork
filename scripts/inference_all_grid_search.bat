@echo off
setlocal EnableDelayedExpansion
call conda activate audio
set MODEL=NO_DATA_AUG_ASV2019

REM 訓練一次（只要你還沒訓練過）
REM python main.py --experiment train --model_name "%MODEL%"

set TASKS=in_the_wild SOTA Asvspoof2019_LA ASVspoof2021_DF

for /L %%E in (0,1,13) do (
    set CHECKPT=checkpt_epoch_%%E.pth
    echo === Running inference for epoch %%E ===
    echo Using checkpoint: !CHECKPT!
    echo Model: %MODEL%
    for %%T in (%TASKS%) do (
        echo Running task %%T with checkpoint epoch %%E
        python main.py --experiment inference --task %%T --model_name %MODEL% --checkpt_name !CHECKPT!
        timeout /t 5 >nul
    )
)

echo === All inferences completed ===
pause
