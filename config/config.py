import argparse
import yaml
import torch

def flatten_nested_dict(d, parent_key='', sep='.'):
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_nested_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)

    # 👇 先 parse 一次 CLI
    args, unknown = parser.parse_known_args()

    # 👇 如果有 YAML，載入並扁平化
    if args.config:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        flat_cfg = flatten_nested_dict(yaml_cfg)

        # 👇 根據 YAML 自動加 argparse 參數
        for k, v in flat_cfg.items():
            parser.add_argument(f'--{k}', type=type(v), default=v)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    args = parser.parse_args()
    return args
