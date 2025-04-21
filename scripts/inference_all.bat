@echo off
call conda activate gordon

REM Define tasks with double quotes to avoid splitting
set TASKS="Asvspoof2019_LA" "in_the_wild" "ASVspoof2021_DF"

for %%T in (%TASKS%) do (
    echo Running task: %%~T
    python main.py --experiment inference --task %%~T --model_name "DSD_ASV019"
)

echo All tasks completed.
pause
