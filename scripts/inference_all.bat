@echo off
call conda activate gordon
set MODEL=NO_DSD_ASV019

python main.py --experiment train --model_name "%MODEL%"

set TASKS="Asvspoof2019_LA" "in_the_wild" "ASVspoof2021_DF"

for %%T in (%TASKS%) do (
    echo Running task: %%~T
    python main.py --experiment inference --task %%~T --model_name "%MODEL%"
    echo Task %%~T finished. Cleaning up...
    timeout /t 10 >nul
)

echo All tasks completed.
pause
