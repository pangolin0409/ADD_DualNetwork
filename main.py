# main.py
from config.config import init
from src.train.train_main_baseline import main as baseline_main
from src.train.train_main import main as train_main
from src.inference.inference import main as inference_main
from src.inference.inference_baseline import main as baseline_inference
from src.inference.fn_anaylsis import main as fn_anaylsis_main
from src.inference.fp_analysis import main as fp_anaylsis_main
from src.utils.common_utils import send_discord
import yaml
import os

def main():
    args = init()

    secret_yml_path = args.secret_yml_path
    if os.path.exists(secret_yml_path):
        with open(secret_yml_path, 'r') as f:
            secrets = yaml.safe_load(f)
        # 移除 wandb 登入流程，僅保留讀取檔案以便後續擴充其他密鑰
    else:
        print("[警告] 找不到 secrets.yml，略過可選密鑰載入")

    webhook = args.discord_webhook
    send_discord(f"🚀 任務啟動：{args.experiment} | 模型：{args.model_name}", webhook)
    try:
        if args.experiment == "baseline_train":
            baseline_main(args)
            send_discord(f"✅ baseline 訓練完成：{args.model_name}", webhook)
        if args.experiment == "baseline_inference":
            baseline_inference(args)
            send_discord(f"📊 推理完成：{args.model_name} on {args.task}", webhook)
        elif args.experiment == "train":
            train_main(args)
            send_discord(f"✅ 訓練完成：{args.model_name}", webhook)
        elif args.experiment == "inference":
            inference_main(args)
            send_discord(f"📊 推理完成：{args.model_name} on {args.task}", webhook)
        elif args.experiment == "fn_anaysis":
            fn_anaylsis_main(args)
            send_discord(f"📊 完成型1型2錯誤統計圖表：{args.model_name} on {args.task}", webhook)
        elif args.experiment == "fp_anaysis":
            fp_anaylsis_main(args)
            send_discord(f"📊 完成型1型2錯誤統計圖表：{args.model_name} on {args.task}", webhook)
    except Exception as e:
        print(f"Error: {e}")
        send_discord(f"❌ 訓練失敗：{args.model_name}", webhook)
        raise e

if __name__ == "__main__":
    main()