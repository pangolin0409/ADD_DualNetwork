# grid_search_infer.py
import os
import numpy as np
import pandas as pd
import onnxruntime as ort
from transformers import Wav2Vec2FeatureExtractor
from config.config import init
from src.inference.inference import inference_loop, load_model, safe_release
from src.utils.eval_metrics import compute_eer
from src.models.Detector import Detector

def run_grid_search(args):
    model_path = os.path.join(args.model_folder, args.model_name, 'best_model.pth')
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    os.makedirs("grid_results", exist_ok=True)

    try:
        onnx_session = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"])
        model_args = {
            'encoder_dim': args.encoder_dim,
            'num_experts': args.num_experts,
            'num_classes': args.num_classes,
            'max_temp': args.max_temp,
            'min_temp': args.min_temp,
            'warmup_epochs': args.warmup_epochs,
            'processor': processor,
            'onnx_session': onnx_session,
        }

        model, _, _, _ = load_model(
            model_class=Detector,
            model_path=model_path,
            device=args.device,
            **model_args)
        
        temps = [0.5, 1.0, 1.5, 2.0, 2.5]
        alphas = [0.3, 0.5, 0.7, 1.0]
        tasks = ["in_the_wild", "SOTA", "Asvspoof2019_LA", "ASVspoof2021_DF"]

        for t in tasks:
            print(f"\n========== Grid Search on [{t}] ==========")
            args.task = t
            results = []

            for temp in temps:
                for alpha in alphas:
                    print(f"[Grid Search][{t}] temp={temp}, alpha={alpha}")
                    score_loader, label_loader, local_gating_loader, moe_feature_loader = inference_loop(args, model, temp, alpha)
                    scores = np.array(score_loader)
                    labels = np.array(label_loader)
                    nontarget_scores = scores[labels == 0]  # 負例 (bonafide)
                    target_scores = scores[labels == 1]     # 正例 (spoof)
                    eer, frr, far, threshold = compute_eer(target_scores, nontarget_scores)
                    results.append((temp, alpha, eer))

            results_df = pd.DataFrame(results, columns=['Temperature', 'Alpha', 'EER'])
            results_df = results_df.sort_values("EER")
            results_df.to_csv(os.path.join("grid_results", f'grid_search_results_on_{t}.csv'), index=False)
    finally:
        safe_release(model, onnx_session)

if __name__ == "__main__":
    args = init()
    run_grid_search(args)
