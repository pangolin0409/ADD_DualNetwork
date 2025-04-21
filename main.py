# main.py
from config.config import init
from src.train.train_main_baseline import main as baseline_main
from src.train.train_main import main as train_main
from src.inference.inference import main as inference_main

def main():
    args = init()
    if args.experiment == "baseline":
        baseline_main(args)
    elif args.experiment == "train":
        train_main(args)
    elif args.experiment == "inference":
        inference_main(args)
    else:
        raise ValueError("Unknown experiment type")

if __name__ == "__main__":
    main()
