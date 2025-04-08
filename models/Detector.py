import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans

###########################################
# Router 模組（包含 Data Augmentation）
###########################################
class TopkRouter(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128, 
                 aug_mode='interpolation', aug_prob=0.5, noise_scale=0.05):
        """
        Args:
            input_dim (int): 輸入特徵維度。
            num_experts (int): 專家數量。
            hidden_dim (int): 隱藏層維度。
            aug_mode (str): 擴增模式，可選 'interpolation' 或 'extrapolation' 或 'none'
            aug_prob (float): 每筆樣本進行 mixup 風格擴增的機率。
            noise_scale (float): 加性 Gaussian Noise 的標準差。
        """
        super(TopkRouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.aug_mode = aug_mode
        self.aug_prob = aug_prob
        self.noise_scale = noise_scale

    def forward(self, x, augment=False, aug_mask=None):
        """
        x: (batch, input_dim)
        augment: 是否啟用 routing data augmentation。
        """
        out = self.fc1(x)
        out = self.relu(out)
        logits = self.fc2(out)
        routing = self.softmax(logits)  # (batch, num_experts)
        if augment and self.aug_mode != 'none' and aug_mask is not None:
            routing = self.augment_routing(routing, aug_mask, mode=self.aug_mode)
        return routing

    def augment_routing(self, routing, mask, mode='interpolation'):
        """
        只對 routing vector 中 mask 為 True 的樣本進行增強。
        mask: (batch,) bool tensor
        """
        device = routing.device
        routing_aug = routing.clone()
        indices = mask.nonzero(as_tuple=True)[0]

        if len(indices) == 0:
            return routing  # 沒有任何要增強的樣本

        selected = routing[indices]

        if mode == 'interpolation':
            perm = torch.randperm(selected.size(0))
            selected_perm = selected[perm]
            lam = torch.rand(selected.size(0), 1, device=device)
            augmented = lam * selected + (1 - lam) * selected_perm

        elif mode == 'extrapolation':
            batch_mean = selected.mean(dim=0, keepdim=True)
            lam = torch.rand(selected.size(0), 1, device=device)
            augmented = selected + lam * (selected - batch_mean)

        else:
            augmented = selected

        noise = torch.randn_like(augmented) * self.noise_scale
        augmented = augmented + noise
        augmented = augmented / (augmented.sum(dim=-1, keepdim=True) + 1e-8)

        routing_aug[indices] = augmented
        return routing_aug


###########################################
# MoE 模組：包含 top-k routing、Limp Loss 與 Load Balance Loss
###########################################
class SparseMoE(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        """
        Args:
            in_dim (int): 輸入特徵維度。
            out_dim (int): 每個 expert 的輸出維度。
            num_experts (int): 專家數量。
            top_k (int): 每筆樣本選擇 top-k expert。
        """
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        # 不需要將 top_k 傳入 Router，Router 只輸出 routing vector
        self.router = TopkRouter(in_dim, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x, router_aug=False, aug_mask=None):
        # x: [B, in_dim]
        routing_weights = self.router(x, augment=router_aug, aug_mask= aug_mask)  # (batch, num_experts)
        topk_values, topk_indices = torch.topk(routing_weights, self.top_k, dim=1)
        
        B = x.size(0)
        # 計算 expert 輸出
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))  # 每個: (B, out_dim)
        expert_outputs = torch.stack(expert_outputs, dim=1)  # (B, num_experts, out_dim)
        
        # 初始化 fused output
        out_dim = expert_outputs.shape[-1]
        fused = torch.zeros(B, out_dim, device=x.device)
        
        # 對每個 expert
        for expert_idx in range(self.num_experts):
            # 找出哪些樣本在 top_k 中選中了當前 expert_idx
            mask = (topk_indices == expert_idx)  # [B, top_k]
            sample_mask = mask.any(dim=1)         # [B]
            if sample_mask.sum() == 0:
                continue
            x_subset = x[sample_mask]  # (B_subset, in_dim)
            # expert 輸出
            expert_out = self.experts[expert_idx](x_subset)  # (B_subset, out_dim)
            # 從 routing_weights 中取出該 expert 的分數
            gating_subset = routing_weights[sample_mask, expert_idx]  # (B_subset)
            weighted_out = expert_out * gating_subset.unsqueeze(-1)
            fused[sample_mask] += weighted_out
        return fused, routing_weights  # 返回 fused output 與原 routing_weights

    def compute_limp_loss(self, routing_weights):
        # 計算單個樣本 routing variance
        mean_val = torch.mean(routing_weights, dim=1, keepdim=True)
        var = torch.mean((routing_weights - mean_val) ** 2, dim=1)
        limp_loss = -torch.mean(var)  # maximize variance -> minimize negative variance
        return limp_loss

    def compute_load_balance_loss(self, routing_weights):
        # 計算 batch 中每個 expert 的平均 routing weight
        batch_mean = torch.mean(routing_weights, dim=0)  # (num_experts,)
        ideal = 1.0 / self.num_experts
        load_loss = torch.mean((batch_mean - ideal) ** 2)
        return load_loss
    
    def compute_entropy_loss(self, routing_weights):
        # routing_weights: [batch, num_experts]
        ent = - (routing_weights * torch.log(routing_weights + 1e-6)).sum(dim=1).mean()
        return ent


###########################################
# MoCo-style Memory Bank (Prototype Memory)
###########################################
class MemoryBank(nn.Module):
    def __init__(self, feature_dim, num_classes, momentum=0.99):
        super(MemoryBank, self).__init__()
        self.register_buffer('prototypes', torch.zeros(num_classes, feature_dim))
        self.momentum = momentum

    def update(self, features, labels):
        for i in range(features.size(0)):
            label = labels[i]
            self.prototypes[label] = self.momentum * self.prototypes[label] + (1 - self.momentum) * features[i]
        self.prototypes = F.normalize(self.prototypes, p=2, dim=1)

class UnknownMemoryBank(nn.Module):
    def __init__(self, feature_dim, num_clusters=3, momentum=0.99):
        super().__init__()
        self.register_buffer('unknown_proto', torch.zeros(num_clusters, feature_dim))
        self.momentum = momentum
        self.initialized = False

    def update_from_kmeans(self, new_centroids):
        # new_centroids: (K, D) from sklearn KMeans
        new_centroids = F.normalize(new_centroids, dim=1)
        if not self.initialized:
            self.unknown_proto = new_centroids.clone().detach()
            self.initialized = True
        else:
            self.unknown_proto = (
                self.momentum * self.unknown_proto + (1 - self.momentum) * new_centroids
            )
        self.unknown_proto = F.normalize(self.unknown_proto, dim=1)

    def get(self):
        return self.unknown_proto

###########################################
# Contrastive Loss（NT-Xent）
###########################################
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        features = F.normalize(features, p=2, dim=1)
        similarity_matrix = torch.matmul(features, features.t())
        logits = similarity_matrix / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eye(logits.shape[0], dtype=torch.bool, device=logits.device)
        logits_masked = logits.masked_fill(mask, -1e9)
        positive_mask = torch.eq(labels, labels.t()).float().to(logits.device) - torch.eye(logits.shape[0], device=logits.device)
        log_prob = F.log_softmax(logits_masked, dim=1)
        loss = - (positive_mask * log_prob).sum(dim=1) / (positive_mask.sum(dim=1) + 1e-9)
        loss = loss.mean()
        return loss

class AttentionPool(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, use_layernorm=True):
        super().__init__()
        self.hidden_dim = hidden_dim if hidden_dim is not None else input_dim
        self.attn_fc1 = nn.Linear(input_dim, self.hidden_dim)
        self.attn_fc2 = nn.Linear(self.hidden_dim, 1)
        self.tanh = nn.Tanh()
        self.use_layernorm = use_layernorm
        if use_layernorm:
            self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, input_dim)  ← Wav2Vec2 hidden states
        Returns:
            pooled: Tensor of shape (B, input_dim)
        """
        attn_score = self.attn_fc2(self.tanh(self.attn_fc1(x)))  # (B, T, 1)
        attn_weights = torch.softmax(attn_score, dim=1)          # (B, T, 1)
        pooled = torch.sum(x * attn_weights, dim=1)              # (B, input_dim)
        if self.use_layernorm:
            pooled = self.norm(pooled)
        return pooled

class SelfAttentionPooling(nn.Module):
    def __init__(self, input_dim=1024, num_heads=4, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=2048,       # 控制容量，避免 overfit
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)  # 建議最多 2
        self.norm = nn.LayerNorm(input_dim)

    def forward(self, x):  # x: [B, T, 1024]
        encoded = self.encoder(x)
        pooled = encoded.mean(dim=1)
        return self.norm(pooled)

class LayerWiseSENet(nn.Module):
    def __init__(self, num_layers=3, feature_dim=1024, reduction=16):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.reduction = reduction

        self.fc1 = nn.Linear(num_layers * feature_dim, (num_layers * feature_dim) // reduction)
        self.fc2 = nn.Linear((num_layers * feature_dim) // reduction, num_layers)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, layer_feats):
        """
        layer_feats: Tensor of shape [B, L, T, D]
        B: batch size, L: number of layers (e.g., 3), T: time steps, D: feature dim (e.g., 1024)
        """
        B, L, T, D = layer_feats.shape

        # Step 1: mean pooling over time: [B, L, D]
        pooled = layer_feats.mean(dim=2)

        # Step 2: flatten features: [B, L * D]
        pooled_flat = pooled.reshape(B, L * D)

        # Step 3: pass through FC to get weights
        x = F.relu(self.fc1(pooled_flat))             # [B, hidden]
        weights = self.softmax(self.fc2(x))           # [B, L]

        # Step 4: apply weights: [B, L, 1, 1] * [B, L, T, D] → [B, T, D]
        weights = weights.unsqueeze(-1).unsqueeze(-1)  # [B, L, 1, 1]
        fused = (weights * layer_feats).sum(dim=1)     # sum over L

        return fused, weights  # fused: [B, T, D], weights: [B, L]
    
class Detector(nn.Module):
    def __init__(self, encoder_dim=1024, expert_dim=128, num_experts=4, top_k=2, num_classes=2, dropout=0.1):
        super(Detector, self).__init__()
        self.senet = LayerWiseSENet(feature_dim=encoder_dim, num_layers=3, reduction=16)
        self.moe = SparseMoE(in_dim=encoder_dim, out_dim=expert_dim, num_experts=num_experts, top_k=top_k)
        self.classifier = nn.Sequential(
            nn.Linear(expert_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        self.norm = nn.LayerNorm(encoder_dim)
    def forward(self, wav2vec_fts=None, router_aug=False, aug_mask=None):
        fused, fused_weights = self.senet(wav2vec_fts) # (batch, T, encoder_dim)
        max_pooled = torch.max(fused, dim=1)[0]
        max_pooled = self.norm(max_pooled)
        moe_output, routing = self.moe(max_pooled, router_aug, aug_mask)  # (batch, encoder_dim), (batch, num_experts)
        logits = self.classifier(moe_output)  # (batch, num_classes)
        return logits, routing, moe_output
