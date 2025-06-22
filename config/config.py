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
    
    #rawnet2_config
    parser.add_argument('-m_first_conv', type = int, default = 251)
    parser.add_argument('-m_in_channels', type = int, default = 1)
    parser.add_argument('-m_filts', type = list, default = [128, [128,128], [128,256], [256,256]])
    parser.add_argument('-m_blocks', type = list, default = [2, 4])
    parser.add_argument('-m_nb_fc_att_node', type = list, default = [1])
    parser.add_argument('-m_nb_fc_node', type = int, default = 1024)
    parser.add_argument('-m_gru_node', type = int, default = 1024)
    parser.add_argument('-m_nb_gru_layer', type = int, default = 1)
    parser.add_argument('-m_nb_samp', type = int, default = 64600)

    # 第二輪 parse 所有參數
    args = parser.parse_args()
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            print(k, v)
            args.model[k[2:]] = v
    
    return args
