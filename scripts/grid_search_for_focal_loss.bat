@echo off
call conda activate audio

setlocal enabledelayedexpansion

set MODEL=GRID_SEARCH_FOCAL_LOSS

for %%G in (1.0 2.0 3.0 4.0) do (
  for %%A in (0.7 0.75 0.8 0.85 0.9) do (
    
    echo ===============================
    echo Running gamma=%%G, alpha=%%A
    echo Model name: "%MODEL%"
    python main.py --experiment train --model_name "%MODEL%" --focal_gamma %%G --focal_alpha %%A --num_epochs 2
  )
)
