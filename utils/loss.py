import torch
import torch.nn.functional as F

"""
    多模態對比學習：
    embeddings_dict: { 
        "modality1": Tensor(shape [batch, dim]),
        "modality2": Tensor(shape [batch, dim]),
        ...
    }
    做兩兩 InfoNCE 後平均
"""
def multi_modal_alignment_loss(embeddings_dict, temperature=0.07):
    modalities = list(embeddings_dict.keys())
    total_loss = 0.0
    count = 0

    for i in range(len(modalities)):
        for j in range(i + 1, len(modalities)):
            emb_i = embeddings_dict[modalities[i]]  # shape: [batch, seq_len, dim]
            emb_j = embeddings_dict[modalities[j]]  # shape: [batch, seq_len, dim]

            # 例如用 mean pooling => [batch, dim]
            emb_i_2d = emb_i.mean(dim=1)
            emb_j_2d = emb_j.mean(dim=1)

            # 單對 InfoNCE
            pair_loss = info_nce_loss(emb_i_2d, emb_j_2d, temperature)
            total_loss += pair_loss
            count += 1

    if count > 0:
        total_loss /= count
    return total_loss


def info_nce_loss(emb_a, emb_b, temperature=0.07):
    """
    對兩個模態做 InfoNCE。
    假設 emb_a.shape = (batch_size, dim), emb_b.shape = (batch_size, dim)，
    批次對應索引 (i) 代表同一筆資料的對應。

    1. 先算出兩兩之間的相似度 (batch_size x batch_size)
    2. 令對角線為正配對，其餘為負配對
    3. 用 cross-entropy 讓相似度最高的成為真實對象

    Returns:
        scalar 損失值(越小越好)。
    """

    # 先做 L2 normalize，避免向量長度影響
    emb_a = F.normalize(emb_a, dim=1)  # [batch, dim]
    emb_b = F.normalize(emb_b, dim=1)  # [batch, dim]

    # 計算相似度矩陣 (a x b^T)，大小 [batch, batch]
    sim_matrix = torch.matmul(emb_a, emb_b.t())  # [batch, batch]
    sim_matrix = sim_matrix / temperature

    # labels: 0 ~ batch-1，每一行 i 的正例就是 column i
    batch_size = emb_a.size(0)
    labels = torch.arange(batch_size).to(emb_a.device)

    # 令 a->b 與 b->a 各算一次 cross_entropy
    loss_a2b = F.cross_entropy(sim_matrix, labels)
    loss_b2a = F.cross_entropy(sim_matrix.t(), labels)

    loss = (loss_a2b + loss_b2a) / 2
    return loss

"""
    計算對比式學習 Loss

    Args:
        features_q (torch.Tensor): 模型萃取人聲表徵
        features_k (torch.Tensor): 模型萃取人聲表徵
        negatives (torch.Tensor): 負向樣本
        temperature (float): temperature
    Returns:
        torch.Tensor: Length loss.
"""
def contrastive_loss(features_q: torch.Tensor, features_k: torch.Tensor, negatives: torch.Tensor, temperature: float) -> torch.Tensor:
    features_q = features_q / features_q.norm(dim=1, keepdim=True)
    features_k = features_k / features_k.norm(dim=1, keepdim=True)
    negatives = negatives / negatives.norm(dim=1, keepdim=True)

    real_utterance_similarity = torch.exp(torch.sum(features_q * features_k, dim=-1) / temperature)
    fake_utterance_similarity = torch.exp(torch.matmul(features_q, negatives.T) / temperature)

    # 避免數值不穩定
    epsilon = 1e-8  # 平滑項
    real_utterance_similarity = torch.clamp(real_utterance_similarity, min=epsilon)
    fake_utterance_similarity_sum = torch.clamp(fake_utterance_similarity.sum(dim=-1), min=epsilon)

    loss = -torch.log(real_utterance_similarity / (real_utterance_similarity + fake_utterance_similarity_sum))
    return loss.mean()

"""
    計算真假人聲長度 Loss

    Args:
        features (torch.Tensor): 模型萃取人聲表徵
        labels (torch.Tensor): 真假人聲標籤
        margin (float): Margin for the loss calculation.

    Returns:
        torch.Tensor: Length loss.
"""
def length_loss(features: torch.Tensor, labels: torch.Tensor, margin: float, weight: float = 1.0) -> torch.Tensor:
    # 假音頻標籤為 1
    fake_features = features[labels == 1]  # 假音頻（合成人聲）
    # 真音頻標籤為 0
    real_features = features[labels == 0]  # 真音頻

    # 真實音頻損失：讓 norm 趨近於 0
    real_loss = weight * torch.norm(real_features, p=2, dim=1).mean()
    # 假音頻損失：讓 norm 超過 margin
    fake_loss = torch.relu(margin - torch.norm(fake_features, p=2, dim=1)).mean()
    
    return real_loss + fake_loss