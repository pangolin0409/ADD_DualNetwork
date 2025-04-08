@echo off

python .\train_main.py
@REM python .\train_main_baseline.py
@REM set TASKS=Asvspoof2019_LA ASVspoof2021_DF CodecFake DFADD in_the_wild
set TASKS=Asvspoof2019_LA in_the_wild ASVspoof2021_DF
for %%T in (%TASKS%) do (
    echo Running task: %%T
    python .\inference.py --task %%T --model_name "DSD_ASV019"
    @REM python .\inference_baseline.py --task %%T --model_name "DSD_ASV019"
)

echo All tasks completed.
pause
