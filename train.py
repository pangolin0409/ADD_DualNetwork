from models.rawnet3.RawNet3 import RawNet3
from models.rawnet3.RawNetBasicBlock import Bottle2neck
import torch
import argparse
import soundfile as sf
import numpy as np

def init():
    parser = argparse.ArgumentParser(description="Train Q-Former Connector for Audio Deepfake Detection")
    # 模型參數
    parser.add_argument("--rawnet3_path", type=str, default='./pretrained_weights/rawnet3/rawnet3_weights.pt', help="Path to the RawNet3 model")

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    args = parser.parse_args()
    return args

def main(args):
    rawnet_model = RawNet3(
        Bottle2neck,
        model_scale=8,
        context=True,
        summed=True,
        encoder_type="ECA",
        nOut=256,
        out_bn=False,
        sinc_stride=10,
        log_sinc=True,
        norm_sinc="mean",
        grad_mult=1,
    )

    rawnet_model.load_state_dict(
        torch.load(
            args.rawnet3_path,
            map_location=lambda storage, loc: storage,
        )["model"]
    )
    rawnet_model.to(args.device)
    rawnet_model.eval()
    audio, _ = sf.read('./demo.wav')
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(args.device)
    with torch.no_grad():
        output = rawnet_model(audio)
    print(output)


if __name__ == "__main__":
    args = init()
    main(args)