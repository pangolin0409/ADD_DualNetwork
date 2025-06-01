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

def init():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./config/base.yml", help='Path to the YAML config file')
    parser.add_argument('--aug_group_file', type=str, default="./config/aug_group.yml", help='Path to aug group YAML')

    args, unknown = parser.parse_known_args()

    # 讀主設定檔
    if args.config:
        with open(args.config, 'r') as f:
            yaml_cfg = yaml.safe_load(f)
        flat_cfg = flatten_nested_dict(yaml_cfg)
        for k, v in flat_cfg.items():
            arg = k.split('.')[-1]
            parser.add_argument(f'--{arg}', type=type(v), default=v)

    # 補上 device、experiment 等
    parser.add_argument("--experiment", type=str, default="moe_full")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    # 第二輪 parse 所有參數
    args = parser.parse_args()
    
    # 載入 augmentation group 設定
    if args.aug_group:
        with open(args.aug_group_file, 'r') as f:
            aug_groups = yaml.safe_load(f)
        args.selected_augs = aug_groups[args.aug_group]
    
    return args
