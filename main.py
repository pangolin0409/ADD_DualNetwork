# main.py
from config.config import init
import src.train.train_main as moe
import src.train.train_main_baseline as baseline
import src.train.train_layer_selector as selector


def main():
    args = init()
    if args.experiment == "baseline":
        baseline.train_main(args)
    elif args.experiment == "layer_selector":
        selector.train_main(args)
    elif args.experiment == "moe_full":
        moe.train_main(args)
    else:
        raise ValueError("Unknown experiment type")

if __name__ == "__main__":
    main()
