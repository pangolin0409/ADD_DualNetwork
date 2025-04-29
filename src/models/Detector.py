import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
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
    
###########################################
# Router 模組（包含 Data Augmentation）
###########################################
class TopkRouter(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128, router_temperature=2.0):
        super(TopkRouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.noise_linear = nn.Linear(input_dim, num_experts)
        self.router_temperature = router_temperature
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        out = self.dropout(self.relu(self.fc1(x)))  # (batch, hidden_dim)
        logits = self.fc2(out)  # (batch, num_experts)
        noise_scale = F.softplus(self.noise_linear(x))  # 保證是正值
        noise = torch.randn_like(logits) * noise_scale
        logits = logits + noise
        routing = self.softmax(logits / self.router_temperature)
        return routing

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

###########################################
# Classifier 模組
###########################################
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

###########################################
# Detector 模組：結合 SparseMoE 與 Classifier
###########################################
class Detector(nn.Module):
    def __init__(self, encoder_dim=1024, num_experts=24, num_classes=2, router_temperature=2.0, processor=None, onnx_session=None):
        super().__init__()
        self.num_experts = num_experts
        self.processor = processor
        self.session = onnx_session

        # 一個大MLP處理所有層
        self.mlp = nn.Sequential(
            nn.Linear(encoder_dim, encoder_dim//2),
            nn.ReLU(),
            nn.Linear(encoder_dim//2, encoder_dim//2),
        )

        # Attention pooling也是一個大的Linear
        self.attn_score = nn.Linear(encoder_dim//2, 1)

        # Router
        self.router = TopkRouter((encoder_dim*num_experts)//2, num_experts, router_temperature=router_temperature)

        self.router_temperature = router_temperature
        self.pre_norm = nn.LayerNorm(encoder_dim//2)
        self.classifier = Classifier(input_dim=encoder_dim//2, num_classes=num_classes)

    def forward(self, wave, is_aug=False, aug_mask=None, aug_method="additive_noise"):
       
        all_hidden_states = self.extract_features_from_onnx(wave)  
        layer_outputs = all_hidden_states[:, 1:]  # [B, 24, T, D]
        B, L, T, D = layer_outputs.shape

        # -- MLP處理
        x = layer_outputs.reshape(B * L, T, D)  # (B*24, T, D)
        x = self.mlp(x)  # (B*24, T, expert_dim)
        x = x.reshape(B, L, T, -1)  # (B, 24, T, expert_dim)

        # -- Attention Pooling
        score = self.attn_score(x)  # (B, 24, T, 1)
        weight = torch.softmax(score, dim=2)  # (B, 24, T, 1)
        pooled = (x * weight).sum(dim=2)  # (B, 24, expert_dim)

        # -- Router
        router_input = pooled.reshape(B, -1)  # (B, 24*expert_dim)
        routing_weights = self.router(router_input)  # (B, num_experts)

        # -- Fusion
        fused = (pooled * routing_weights.unsqueeze(-1)).sum(dim=1)  # (B, expert_dim)

        fused = self.apply_augmentation(fused, is_aug, aug_mask, aug_method)  # (B, expert_dim)
        fused = self.pre_norm(fused)
        logits = self.classifier(fused)

        return logits, routing_weights, fused

    def apply_augmentation(self, feature, is_aug, aug_mask, aug_method): 
        if not is_aug or aug_mask is None:
            return feature
        
        feature = feature.clone()

        if aug_method == "additive_noise":
            noise_std=0.05
            noise = torch.randn_like(feature[aug_mask]) * noise_std
            feature[aug_mask] = feature[aug_mask] + noise
        elif aug_method == "affine":
            scale_range=(0.9, 1.1)
            shift_range=(-0.1, 0.1)
            scale = torch.empty((feature[aug_mask].size(0), 1), device=feature.device).uniform_(*scale_range)
            shift = torch.empty((feature[aug_mask].size(0), 1), device=feature.device).uniform_(*shift_range)
            feature[aug_mask] = scale * feature[aug_mask] + shift

        return feature

    def extract_features_from_onnx(self, waveform):
        """
        使用 ONNX 將 raw waveform 轉成 hidden states (25 layers)
        """
        inputs = self.processor(
            waveform, sampling_rate=16000, return_tensors="np"
        )
        input_values = inputs["input_values"]
        attention_mask = inputs["attention_mask"]

        # squeeze 處理 (batch, 1, seq_len) → (batch, seq_len)
        if input_values.ndim == 3 and input_values.shape[0] == 1:
            input_values = np.squeeze(input_values, axis=0)
        if attention_mask.ndim == 3 and attention_mask.shape[0] == 1:
            attention_mask = np.squeeze(attention_mask, axis=0)

        input_values = input_values.astype(np.float16)
        attention_mask = attention_mask.astype(np.int64)

        # ONNX forward
        outputs = self.session.run(None, {
            "input_values": input_values,
            "attention_mask": attention_mask
        })

        # (25, B, T, D) → (B, 25, T, D)
        all_hidden_states = np.array(outputs[1])
        all_hidden_states = np.transpose(all_hidden_states, (1, 0, 2, 3))

        return torch.tensor(all_hidden_states, dtype=torch.float32).to(
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    def compute_limp_loss(self, routing_weights):
        # routing_weights: [batch, num_experts]
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
        ent = - (routing_weights * torch.log(routing_weights + 1e-6)).sum(dim=1).mean()
        return ent
    
    def compute_router_supervised_loss(self, routing_weights, logits, labels):
        preds = torch.argmax(logits, dim=1)  # [B]

        # Step 2: classify correct / incorrect
        correct_mask = (preds == labels).float()  # [B]
        incorrect_mask = 1.0 - correct_mask        # [B]

        # Step 3: compute routing entropy
        entropy = - (routing_weights * torch.log(routing_weights + 1e-6)).sum(dim=1)  # [B]

        # Step 4: apply only on incorrect samples
        loss_router_supervised = (incorrect_mask * entropy).mean()

        return loss_router_supervised