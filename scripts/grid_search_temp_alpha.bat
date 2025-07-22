@echo off
call conda activate audio

setlocal enabledelayedexpansion
set MAXTEMPS=2.0 2.25 2.5
set ALPHAS=0.5 0.6 0.7 0.8

for %%M in (%MAXTEMPS%) do (
    for %%A in (%ALPHAS%) do (
        set MODEL=EXP_TMP%%M%%_ALPHA%%A%%
        echo Training model !MODEL! with max_temp=%%M and end_alpha=%%A
        python main.py --experiment train --model_name !MODEL! --max_temp %%M --end_alpha %%A

        set TASKS="in_the_wild" "ASVspoof2021_DF" "ADD" "zh-fbgkx-aishell3-2025_v1" "en-fbgkx-librispeech-2025_v1" 
        for %%T in (%TASKS%) do (
            python main.py --experiment inference --task %%T --model_name !MODEL!
            timeout /t 5 >nul
        )
    )
)
endlocal
pause
