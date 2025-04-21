# main.py
from config.config import init
from src.train.train_main_baseline import main as baseline_main
from src.train.train_main import main as train_main
from src.inference.inference import main as inference_main
from src.utils.common_utils import send_discord

def main():
    args = init()
    webhook = args.discord_webhook
    send_discord(f"🚀 任務啟動：{args.experiment} | 模型：{args.model_name}", webhook)
    try:
        if args.experiment == "baseline":
            baseline_main(args)
            send_discord(f"✅ baseline 訓練完成：{args.model_name}", webhook)
        elif args.experiment == "train":
            train_main(args)
            send_discord(f"✅ 訓練完成：{args.model_name}", webhook)
        elif args.experiment == "inference":
            inference_main(args)
            send_discord(f"📊 推理完成：{args.model_name} on {args.task}", webhook)
        else:
            raise ValueError("Unknown experiment type")
    except Exception as e:
        print(f"Error: {e}")
        send_discord(f"❌ 訓練失敗：{args.model_name}", webhook)
        raise e

if __name__ == "__main__":
    main()
