@echo off
call conda activate gordon
@REM python .\train_main.py
set TASKS=Asvspoof2019_LA in_the_wild ASVspoof2021_DF
for %%T in (%TASKS%) do (
    echo Running task: %%T
    python .\inference.py --task %%T --model_name "DSD_ASV019"
)

echo All tasks completed.
pause
