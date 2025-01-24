import torch
import torch.nn.functional as F

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
