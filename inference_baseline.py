from data.dataloader import RawAudio
from torch.utils.data import DataLoader
import torch
import os
from tqdm import tqdm
import argparse
from utils.eval_metrics import compute_eer
from torch.nn.functional import softmax
import numpy as np
from models.classifier.ASSIST import AasistEncoder
import json
from train_main_baseline import DownStreamLinearClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

torch.multiprocessing.set_start_method('spawn', force=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init():
    parser = argparse.ArgumentParser("load model scores")
    parser.add_argument('--encoder_dim', type=int, default=1024, help="Dimension of the encoder output")
    parser.add_argument('--routing_dim', type=int, default=4, help="Dimension of the routing network")
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--expert_dim', type=int, default=128, help="Dimension of the routing network")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes for classification")
    parser.add_argument("--wav2vec_path", type=str, default='./pretrained_models/wav2vec2-xls-r-300m', help="Path to the wav2vec model")
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model", default='./checkpoints')
    parser.add_argument('-n', '--model_name', type=str, help="the name of the model", required=False, default='DSD_ASV019')
    parser.add_argument('-s', '--score_dir', type=str, help="folder path for writing score", default='./scores')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would like to score on", required=False, default='19eval')
    parser.add_argument('-nb_samp', type=int, default=64600)
    parser.add_argument('-nb_worker', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for inference")  # 新增 batch_size 參數
    parser.add_argument("--gpu", type=str, help="GPU index", default="2")
    parser.add_argument("--aasist_config_path", type=str, default='./config/AASIST.conf', help="Path to the AASIST model config")
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    return args

    
def load_model(model_class, model_path, device, aasist_encoder):
    # 初始化模型
    model = model_class(aasist_encoder).to(device)
    
    # 載入檢查點
    checkpoint = torch.load(model_path, map_location=device)
    
    # 如果是 state_dict，就用 load_state_dict 載入
    if isinstance(checkpoint, dict) or isinstance(checkpoint, torch.collections.OrderedDict):
        model.load_state_dict(checkpoint)
    else:
        # 如果是完整模型，直接載入
        model = checkpoint.to(device)

    model.eval()
    return model

def draw_expert_usage(gating, gating_type, task):
    gating_array = np.array(gating)  # shape: [N, num_experts]
    avg_usage = gating_array.mean(axis=0)  # 每個 expert 的平均使用量

    plt.figure(figsize=(8, 4))
    sns.barplot(x=list(range(len(avg_usage))), y=avg_usage)
    plt.title(f"{gating_type} expert Usage on {task}")
    plt.xlabel("Expert ID")
    plt.ylabel("Avg Gating Score")
    plt.savefig(f"{gating_type} expert_usage_{task}.png")
    plt.close()

def draw_ft_dist(features, labels,task, label_names=None):
    features = np.array(features)
    labels = np.array(labels)
    
    tsne = TSNE(n_components=2, perplexity=10, init='pca', learning_rate='auto', random_state=42)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(label_names or []):
        idx = (labels == i)
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=name, s=8, alpha=0.5)
    plt.legend()
    plt.title(f"MoE Feature Distribution on {task}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"feature_distribution_{task}.png")
    plt.close()
    
def test_on_desginated_datasets(task, model_path, save_path):
    with open(args.aasist_config_path, "r") as f_json:
        aasist_config = json.loads(f_json.read())
    aasist_model_config = aasist_config["model_config"]
    aasist_encoder = AasistEncoder(aasist_model_config).to(device)
    
    model = load_model(
        model_class=DownStreamLinearClassifier,
        model_path=model_path,
        device=device,
        aasist_encoder=aasist_encoder,)
    model.eval()

    # 加載數據集 (改為批量推理)
    print(f"Loading dataset: {task}")
    match task:
        case 'ASVspoof2021_DF':
            path_to_wav2vec_ft = f'D:/datasets/{task}'
        case _:
            path_to_wav2vec_ft = f'E:/datasets/{task}'
    test_set = RawAudio(
        path_to_database=f'../datasets/{task}',
        meta_csv='meta.csv',
        return_label=True,
        nb_samp=args.nb_samp,
        part='test',
        wav2vec_path_prefix=path_to_wav2vec_ft,
    )
    testDataLoader = DataLoader(
        test_set, 
        batch_size=args.batch_size,   # 批量推理
        shuffle=False, 
        drop_last=False, 
        num_workers=args.nb_worker, 
        pin_memory=True
    )

    # 用於保存結果
    score_loader = []
    label_loader = []
    # local_gating_loader = []
    # moe_feature_loader = []
    model.eval()

    # 遍歷測試數據
    for i, data_slice in enumerate(tqdm(testDataLoader)):
        waveforms, labels, wav2vec_fts = data_slice
        # waveforms = waveforms.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        # wav2vec_fts = wav2vec_fts.to(device, non_blocking=True)
        wav2vec_ft = wav2vec_fts[:,2,:,:].to(device)
        # 模型推理 (批量)
        with torch.no_grad():
            logits = model(wav2vec_ft)
            scores = softmax(logits, dim=1)[:, 1]
        
        # 保存分數和標籤
        score_loader.extend(scores.detach().cpu().numpy().tolist())
        label_loader.extend(labels.detach().cpu().numpy().tolist())

    scores = np.array(score_loader)
    labels = np.array(label_loader)

    # 根據標籤分割分數
    nontarget_scores = scores[labels == 0]  # 正例 (bonafide)
    target_scores = scores[labels == 1]     # 負例 (spoof)

    eer, frr, far, threshold = compute_eer(target_scores, nontarget_scores)
    
    print(f'Equal Error Rate (EER): {eer}, False Rejection Rate (FFR): {frr}, False Acceptance Rate (FAR): {far}, Threshold: {threshold}')
    with open(os.path.join(save_path, "inference.txt"), "a") as f:
        f.write(f"Test sets: {task}, EER: {eer:.4f}, FFR: {frr:.4f}, FAR: {far:.4f}, Threshold: {threshold:.4f}\n")

    # draw_ft_dist(moe_feature_loader, labels, task, label_names=["Bona fide", "Spoof"])
    
if __name__ == "__main__":
    args = init()
    model_path = os.path.join(args.model_folder, args.model_name, 'best_model.pth')
    save_path = os.path.join(args.model_folder, args.model_name)
    test_on_desginated_datasets(args.task, model_path, save_path)