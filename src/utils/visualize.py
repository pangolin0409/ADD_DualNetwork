import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
from collections import Counter

def draw_expert_usage(gating, task, save_path):
    gating_array = np.array(gating)  # shape: [N, num_experts]
    avg_usage = gating_array.mean(axis=0)
    std_usage = gating_array.std(axis=0)

    x = np.arange(len(avg_usage))

    plt.figure(figsize=(8, 4))
    plt.bar(x, avg_usage, yerr=std_usage, capsize=5)
    plt.title(f"Expert Usage on {task}")
    plt.xlabel("Expert ID")
    plt.ylabel("Avg Gating Score")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"expert_usage_on_{task}.png"))
    plt.close()

def draw_ft_dist(features, labels, task, save_path, label_names=None):
    features = np.array(features)
    labels = np.array(labels)

    tsne = TSNE(n_components=2, perplexity=10, init='pca', learning_rate='auto', random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    for i, name in enumerate(label_names or []):
        idx = (labels == i)
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=name, s=8, alpha=0.5)
    plt.legend()
    plt.title(f"MoE Feature Distribution on {task}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"feature_distribution_on_{task}.png"))
    plt.close()

def draw_norm(features, labels, task, save_path):
    # 把 list of list 轉為 numpy array（shape: [N, D]）
    features = np.array(features)
    labels = np.array(labels)

    # 布林 mask indexing 成立
    real = features[labels == 0]
    fake = features[labels == 1]

    real_norms = np.linalg.norm(real, ord=2, axis=1)
    fake_norms = np.linalg.norm(fake, ord=2, axis=1)

    # 畫圖
    plt.hist(real_norms, bins=50, alpha=0.6, label="Real")
    plt.hist(fake_norms, bins=50, alpha=0.6, label="Fake")
    plt.xlabel("L2 Norm")
    plt.ylabel("Count")
    plt.title(f"Feature Norm Distribution ({task})")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_path, f"feature_norm_on_{task}.png"))
    plt.close()

def draw_roc_curve(scores, labels, task, save_path):
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve on {task}')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"roc_curve_on_{task}.png"))
    plt.close()

def draw_confidence_histogram(scores, labels, task, save_path):
    plt.figure(figsize=(8, 5))
    plt.hist(scores[labels == 0], bins=50, alpha=0.6, label='Bonafide')
    plt.hist(scores[labels == 1], bins=50, alpha=0.6, label='Spoof')
    plt.title(f'Score Distribution on {task}')
    plt.xlabel('Spoof Probability')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"score_histogram_on_{task}.png"))
    plt.close()

def draw_score_kde(scores, labels, task, save_path):
    plt.figure(figsize=(8, 5))
    sns.kdeplot(scores[labels == 0], label='Bonafide', fill=True, alpha=0.5)
    sns.kdeplot(scores[labels == 1], label='Spoof', fill=True, alpha=0.5)
    plt.title(f'Score Density Estimation on {task}')
    plt.xlabel('Spoof Probability')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"score_kde_on_{task}.png"))
    plt.close()

def draw_expert_heatmap(routing_matrix, task, save_path, max_samples=100):
    matrix = np.array(routing_matrix)
    matrix = matrix[:max_samples]  # avoid overload

    plt.figure(figsize=(10, 6))
    sns.heatmap(matrix, cmap='YlGnBu', cbar=True)
    plt.title(f"Expert Routing Heatmap (First {max_samples} samples) on {task}")
    plt.xlabel("Expert ID")
    plt.ylabel("Sample Index")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"routing_heatmap_on_{task}.png"))
    plt.close()

def draw_selector_distribution(topk_idx_list, labels, save_path, k, task, num_layers=24):
    """
    Args:
        topk_idx_list: List of [K] arrays (每筆樣本選中的 layer indices)
        labels: List of labels (0/1)
        save_path: 存圖目錄
        num_layers: Wav2Vec2 total layer數（通常是 24）
    """
    all_topk = np.concatenate(topk_idx_list).astype(int)
    counter = Counter(all_topk)
    freq = np.zeros(num_layers)
    for i in range(num_layers):
        freq[i] = counter[i]

    freq = freq / freq.sum()

    plt.figure(figsize=(10, 4))
    sns.barplot(x=list(range(num_layers)), y=freq)
    plt.title(f"Top-{k} Layer Selection Frequency (Inference) on {task}")

    plt.xlabel("Layer Index")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"selector_layer_freq_on_{task}.png"))
    plt.close()

def draw_ft_direction(features, labels, task, save_path, label_names=None):
    features = np.array(features)
    labels = np.array(labels)

    # 🔁 步驟 1：normalize to unit vector（即映射到球面）
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / (norms + 1e-6)

    # 🔁 步驟 2：t-SNE 作投影
    tsne = TSNE(n_components=2, perplexity=10, init='pca', learning_rate='auto', random_state=42, max_iter=1000)
    reduced = tsne.fit_transform(features)

    # 🔁 步驟 3：畫圖
    plt.figure(figsize=(10, 8))
    for i, name in enumerate(label_names or []):
        idx = (labels == i)
        plt.scatter(reduced[idx, 0], reduced[idx, 1], label=name, s=8, alpha=0.5)
    plt.legend()
    plt.title(f"**Spherical Feature Direction Distribution** on {task}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"spherical_direction_on_{task}.png"))
    plt.close()

def draw_spherical_feature_stats(features, labels, task, save_path):
    features = np.array(features)
    labels = np.array(labels)

    # ✅ Normalize to unit sphere
    features = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-6)

    # ✅ 只畫前幾維比較有意義（比如 5 維）
    dims_to_plot = 5
    dim_indices = np.arange(min(dims_to_plot, features.shape[1]))

    real = features[labels == 0]
    fake = features[labels == 1]

    plt.figure(figsize=(12, 8))
    for i, dim in enumerate(dim_indices):
        plt.subplot(2, 3, i + 1)
        plt.hist(real[:, dim], bins=50, alpha=0.5, label='Real')
        plt.hist(fake[:, dim], bins=50, alpha=0.5, label='Fake')
        plt.title(f"Dim {dim} Distribution")
        plt.xlabel("Value on Unit Sphere")
        plt.ylabel("Count")
        plt.legend()

    plt.suptitle(f"Spherical Feature Dimension-wise Distribution ({task})")
    plt.tight_layout()
    os.makedirs(save_path, exist_ok=True)
    plt.savefig(os.path.join(save_path, f"spherical_feature_stats_{task}.png"))
    plt.close()

