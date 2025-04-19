from data.dataloader import RawAudio
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
from utils.eval_metrics import compute_eer
from torch.nn.functional import softmax
import numpy as np
from models.Detector import Detector, LayerSelectorMoE, Classifier
from transformers import Wav2Vec2FeatureExtractor
import onnxruntime as ort
import gc
from src.utils.visualize import (
    draw_expert_usage,
    draw_ft_dist,
    draw_roc_curve,
    draw_confidence_histogram,
    draw_score_kde,
    draw_expert_heatmap,
    draw_selector_distribution,
)

def init():
    ort.preload_dlls(cuda=True, cudnn=True, msvc=True)
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--encoder_dim', type=int, default=1024, help="Dimension of the encoder output")
    parser.add_argument('--routing_dim', type=int, default=4, help="Dimension of the routing network")
    parser.add_argument('--select_k', type=int, default=5, help="Number of layers to select")
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--expert_dim', type=int, default=128, help="Dimension of the routing network")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes for classification")
    parser.add_argument("--wav2vec_path", type=str, default='./pretrained_models/wav2vec2-xls-r-300m', help="Path to the wav2vec model")
    parser.add_argument("--onnx_path", type=str, default='./onnx_models/xlsr_base_fp16.onnx', help="Path to the ONNX model")
    parser.add_argument("--plot_path", type=str, default='./plot', help="Path to the plot")
    parser.add_argument("--log_path", type=str, default='./log', help="Path to the log")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model", default='./checkpoints')
    parser.add_argument('-n', '--model_name', type=str, help="the name of the model", required=False, default='DSD_ASV019')
    parser.add_argument('--layer_select_model_name', type=str, default='LS_ASV019')
    parser.add_argument('-s', '--score_dir', type=str, help="folder path for writing score", default='./scores')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would like to score on", required=False, default='19eval')
    parser.add_argument('-nb_samp', type=int, default=64600)
    parser.add_argument('-nb_worker', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for inference")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args

def load_model(model_class, model_path, device, **model_args):
    checkpoint = torch.load(model_path, map_location=device)

    # 如果 checkpoint 是 dict，且包含 classifier
    classifier_state_dict = None
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'selector' in checkpoint:
            state_dict = checkpoint['selector']
        else:
            state_dict = checkpoint
        if 'classifier' in checkpoint:
            classifier_state_dict = checkpoint['classifier']
    else:
        return checkpoint.to(device)

    model = model_class(**model_args).to(device)

    # 載入主網路
    model.load_state_dict(state_dict, strict=False)

    # 如果 model 有 classifier 屬性，載入 classifier 參數
    if hasattr(model, 'classifier') and classifier_state_dict is not None:
        model.classifier.load_state_dict(classifier_state_dict)

    model.eval()
    return model


def load_datasets(task, nb_samp, batch_size, nb_worker):
    # 加載數據集 (改為批量推理)
    print(f"Loading dataset: {task}")
    test_set = RawAudio(
        path_to_database=f'F:/datasets/{task}',
        meta_csv='meta.csv',
        return_label=True,
        nb_samp= nb_samp,
        part='test',
    )
    testDataLoader = DataLoader(
        test_set, 
        batch_size=batch_size,
        shuffle=False, 
        drop_last=False, 
        num_workers=nb_worker, 
        pin_memory=True, # 鎖定記憶體
        persistent_workers=True # 用同一批 worker，不再重新創建
    )
    return testDataLoader

def inference_loop(args, selector, model):
    testDataLoader = load_datasets(args.task, args.nb_samp, args.batch_size, args.nb_worker)

    score_loader = []
    label_loader = []
    local_gating_loader = []
    moe_feature_loader = []
    topk_idx_loader = []

    # 遍歷測試數據
    for i, data_slice in enumerate(tqdm(testDataLoader)):
        waveforms, labels = data_slice
        waveforms = waveforms.to(args.device, non_blocking=True)
        labels = labels.to(args.device, non_blocking=True)
        # wav2vec_fts = wav2vec_fts.to(device, non_blocking=True)
        # 模型推理 (批量)
        with torch.no_grad():
            projected, scores, topk_idx, selected_layers = selector(waveforms)
            logits, routing, moe_output, fused_output = model(selected_layers=selected_layers)
            scores = softmax(logits, dim=1)[:, 1]
        
        print(f"[DEBUG] waveforms shape: {waveforms.shape}")
        print(f"[DEBUG] topk_idx shape: {topk_idx.shape}")
        print(f"[DEBUG] selected_layers shape: {selected_layers.shape}")
        print(f"[DEBUG] routing shape: {routing.shape}")
        print(f"[DEBUG] fused_output shape: {fused_output.shape}")
        print(f"[DEBUG] scores shape: {scores.shape}")

        # 保存專家使用量
        local_gating_loader.extend(routing.detach().cpu().numpy().tolist())

        # 保存分數和標籤
        score_loader.extend(scores.detach().cpu().numpy().tolist())
        label_loader.extend(labels.detach().cpu().numpy().tolist())
        moe_feature_loader.extend(fused_output.detach().cpu().numpy().tolist())
        topk_idx_loader.extend(topk_idx.detach().cpu().numpy().tolist())
    
    return score_loader, label_loader, local_gating_loader, moe_feature_loader, topk_idx_loader

def inference(args, model_path, selector_path, save_path):
    processor = Wav2Vec2FeatureExtractor.from_pretrained(args.wav2vec_path)
    classifier = Classifier(input_dim=args.expert_dim, num_classes=args.num_classes).to(args.device)

    model_args = {
    'encoder_dim': args.encoder_dim,
    'expert_dim': args.expert_dim,
    'num_experts': args.routing_dim,
    'top_k': args.top_k,
    'num_classes': args.num_classes,
    'classifier': classifier,
    }
    model = None
    selector = None
    try:
        onnx_session = ort.InferenceSession(args.onnx_path, providers=["CUDAExecutionProvider"])
        selector_args = {
            'topk': args.select_k,
            'processor': processor,
            'onnx_session':onnx_session,
        }
        selector = load_model(
            model_class=LayerSelectorMoE,
            model_path=selector_path,
            device=args.device,
            **selector_args)

        model = load_model(
            model_class=Detector,
            model_path=model_path,
            device=args.device,
            **model_args)

        score_loader, label_loader, local_gating_loader, moe_feature_loader, topk_idx_loader = inference_loop(args, selector, model)
    except Exception as e:
        print(f"Error during inference: {e}")
        raise e
    finally:
        del model
        del selector
        gc.collect()

    scores = np.array(score_loader)
    labels = np.array(label_loader)

    # 根據標籤分割分數
    nontarget_scores = scores[labels == 0]  # 負例 (bonafide)
    target_scores = scores[labels == 1]     # 正例 (spoof)

    # compute_eer(spoof, bonafide)
    eer, frr, far, threshold = compute_eer(target_scores, nontarget_scores)
    
    print(f'Equal Error Rate (EER): {eer}, False Rejection Rate (FFR): {frr}, False Acceptance Rate (FAR): {far}, Threshold: {threshold}')
    with open(os.path.join(save_path, "inference.txt"), "a") as f:
        f.write(f"Test sets: {args.task}, EER: {eer:.4f}, FFR: {frr:.4f}, FAR: {far:.4f}, Threshold: {threshold:.4f}\n")

    return scores, labels, local_gating_loader, moe_feature_loader, topk_idx_loader

def main():
    args = init()
    model_path = os.path.join(args.model_folder, args.model_name, 'best_model.pth')
    selector_path = os.path.join(args.model_folder, args.layer_select_model_name, 'best_model.pth')
    save_path = os.path.join(args.log_path, args.model_name)
    os.makedirs(save_path, exist_ok=True)
    scores, labels, local_gating_loader, moe_feature_loader, topk_idx_loader = inference(args, model_path, selector_path, save_path)

    plot_dir = os.path.join(args.plot_path, args.model_name)
    os.makedirs(plot_dir, exist_ok=True)

    # === 視覺化 ===
    draw_expert_usage(local_gating_loader, "Local", args.task, plot_dir)
    draw_ft_dist(moe_feature_loader, labels, args.task, plot_dir, label_names=["Bona fide", "Spoof"])
    draw_roc_curve(scores, labels, args.task, plot_dir)
    draw_confidence_histogram(scores, labels, args.task, plot_dir)
    draw_score_kde(scores, labels, args.task, plot_dir)
    draw_expert_heatmap(local_gating_loader, args.task, plot_dir)
    draw_selector_distribution(topk_idx_loader, labels, plot_dir, args.select_k, args.task)
    print(f"All visualizations saved in: {plot_dir}")

if __name__ == "__main__":
    if torch.cuda.is_available():
        torch.multiprocessing.set_start_method('spawn', force=True)
    main()