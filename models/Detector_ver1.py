import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchaudio

# 1. 多尺度注意力（保留原有實作，用於 wav2vec 分支時可用）
class MultiScaleAttention(nn.Module):
    def __init__(self, feature_dim=1024, num_heads=4, kernel_sizes=[3, 5, 7, 11]):
        super(MultiScaleAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        assert feature_dim % len(kernel_sizes) == 0, f"Feature dimension {feature_dim} cannot be divided by {len(kernel_sizes)}!"
        conv_out_dim = feature_dim // len(kernel_sizes)
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(in_channels=feature_dim, out_channels=conv_out_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # x: [B, seq_len, feature_dim]
        x_conv = x.permute(0, 2, 1)  # [B, feature_dim, seq_len]
        scale_features = [F.relu(conv(x_conv)) for conv in self.scale_convs]
        x_scaled = torch.cat(scale_features, dim=1)  # [B, feature_dim, seq_len]
        x_scaled = x_scaled.permute(2, 0, 1)  # [seq_len, B, feature_dim]
        attn_output, _ = self.self_attn(x_scaled, x_scaled, x_scaled)
        attn_output = attn_output.permute(1, 0, 2)  # [B, seq_len, feature_dim]
        x = self.norm(x + self.dropout(attn_output))
        return x

# 2. 多通道注意力（原有實作）
class MultiChannelAttention(nn.Module):
    def __init__(self, num_channels, reduction_ratio=16):
        super(MultiChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(num_channels, num_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(num_channels // reduction_ratio, num_channels),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x: [B, num_channels, time]
        x_squeezed = self.avg_pool(x).squeeze(-1)  # [B, num_channels]
        attn_weights = self.fc(x_squeezed).unsqueeze(-1)  # [B, num_channels, 1]
        return x * attn_weights

# 3. Top-K Router（向量輸入版）
class TopkRouter(nn.Module):
    def __init__(self, in_dim, num_experts, top_k):
        super(TopkRouter, self).__init__()
        self.top_k = top_k
        self.linear = nn.Linear(in_dim, num_experts)
    
    def forward(self, x):
        # x: [B, in_dim]
        logits = self.linear(x)  # [B, num_experts]
        topk_logits, indices = logits.topk(self.top_k, dim=-1)
        zeros = torch.full_like(logits, float('-inf'))
        sparse_logits = zeros.scatter(-1, indices, topk_logits)
        router_output = F.softmax(sparse_logits, dim=-1)  # [B, num_experts]
        return router_output, indices

# 4. MoE 模組（向量輸入版）
class SparseMoE(nn.Module):
    def __init__(self, in_dim, out_dim, num_experts, top_k):
        super(SparseMoE, self).__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = TopkRouter(in_dim, num_experts, top_k)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.ReLU(),
                nn.Linear(out_dim, out_dim)
            ) for _ in range(num_experts)
        ])
    
    def forward(self, x):
        # x: [B, in_dim]
        gating, indices = self.router(x)  # gating: [B, num_experts], indices: [B, top_k]
        B, in_dim = x.shape
        # 推算 expert 輸出維度（假設所有 expert 輸出維度相同）
        out_dim = self.experts[0](x[:1]).shape[-1]
        fused = torch.zeros(B, out_dim, device=x.device)
        
        # 對每個 expert 進行向量化計算，只處理被選中的樣本
        for expert_idx in range(self.num_experts):
            # 找出哪些樣本在 top_k 中選中了當前 expert_idx
            mask = (indices == expert_idx)  # [B, top_k]，布林矩陣
            sample_mask = mask.any(dim=1)     # [B]，若該樣本的 top_k 中至少有一個是 expert_idx 則為 True
            if sample_mask.sum() == 0:
                continue
            # 擷取被選中 expert_idx 的樣本
            x_subset = x[sample_mask]  # [B_subset, in_dim]
            # 只對這些樣本計算該 expert 的輸出
            expert_out = self.experts[expert_idx](x_subset)  # [B_subset, out_dim]
            # 從 gating 中取出這些樣本對 expert_idx 的 gating 分數
            gating_subset = gating[sample_mask, expert_idx]  # [B_subset]
            # 計算加權輸出：每個樣本 expert 輸出乘上 gating 分數
            weighted_out = expert_out * gating_subset.unsqueeze(-1)  # [B_subset, out_dim]
            # 將加權結果加回 fused 輸出中（這裡採用累加策略）
            fused[sample_mask] += weighted_out
        return fused, gating

# 5. 門控融合（向量級，融合 local 與 global 分支）
class GatedFusionVector(nn.Module):
    def __init__(self, local_dim, global_dim, fusion_dim):
        super(GatedFusionVector, self).__init__()
        self.fc_gate = nn.Sequential(
            nn.Linear(local_dim + global_dim, fusion_dim),
            nn.ReLU(),
            nn.Linear(fusion_dim, 2),
            nn.Sigmoid()
        )
        self.fc_fuse = nn.Linear(local_dim + global_dim, fusion_dim)
    
    def forward(self, local_feat, global_feat):
        # local_feat: [B, local_dim], global_feat: [B, global_dim]
        combined = torch.cat([local_feat, global_feat], dim=-1)
        gate_weights = self.fc_gate(combined)  # [B, 2]
        fused = gate_weights[:, 0:1] * local_feat + gate_weights[:, 1:2] * global_feat
        fused = self.fc_fuse(torch.cat([local_feat, global_feat], dim=-1))
        return fused

# 7. 最終模型：融合 LFCC+LCNN (Local) 與 wav2vec (Global)
class Detector(nn.Module):
    def __init__(self, wav2vec_dim=1024, fusion_hidden_dim=256,
                 lcnn_enc_dim=128, num_classes=2,
                 local_num_experts=3, global_num_experts=3, top_k=1):
        super(Detector, self).__init__()
        # LFCC 模組：假設輸入 [B, waveform_length] -> 輸出 [B, H, W]（例如 H=64, W=64）
        self.lfcc = torchaudio.transforms.LFCC(
                    sample_rate=16000,
                    n_filter=128,         # 濾波器數量，可根據需求調整
                    f_min=0.0,
                    f_max=8000.0,         # 由於採樣率 16000，f_max 默認為 8000 Hz
                    n_lfcc=40,            # 最終 LFCC 係數數量
                    dct_type=2,
                    norm='ortho',
                    log_lf=False          # 是否取對數
                )
        # LCNN：輸入 [B, 1, H, W]，輸出向量 [B, lcnn_enc_dim]
        self.lcnn = LCNN(enc_dim=lcnn_enc_dim)
        # Local MoE：將 LCNN 輸出向量融合，輸入維度 lcnn_enc_dim，輸出 fusion_hidden_dim
        self.local_moe = SparseMoE(in_dim=lcnn_enc_dim, out_dim=fusion_hidden_dim,
                             num_experts=local_num_experts, top_k=top_k)
        # Global MoE：對 wav2vec 特徵做時間平均，輸入維度 wav2vec_dim，輸出 fusion_hidden_dim
        self.global_moe = SparseMoE(in_dim=wav2vec_dim, out_dim=fusion_hidden_dim,
                              num_experts=global_num_experts, top_k=top_k)
        # 融合門控：融合來自 local 與 global 的向量，輸出 fusion_hidden_dim
        self.gated_fusion = GatedFusionVector(local_dim=fusion_hidden_dim,
                                              global_dim=fusion_hidden_dim,
                                              fusion_dim=fusion_hidden_dim)
        # 分類器：輸入融合後向量，輸出 num_classes
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim, fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim // 2, num_classes)
        )
    
    def forward(self, wave, wav2vec_feat):
        # wave: [B, waveform_length]，原始波形
        # wav2vec_feat: [B, T, D]，例如 T=10, D=1024
        # Local branch:
        # LFCC 輸入需為 [B, waveform_length]，因此如果 wave 有多餘的通道維度，先 squeeze
        lfcc_feat = self.lfcc(wave)  # [B, H, W]，例如 [B, 64, 64]
        lfcc_feat = lfcc_feat.unsqueeze(1)  # [B, 1, H, W]，符合 LCNN 輸入要求
        local_vec = self.lcnn(lfcc_feat)  # [B, lcnn_enc_dim]
        local_out, local_gating = self.local_moe(local_vec)  # [B, fusion_hidden_dim]

        # Global branch:
        # 對 wav2vec_feat 進行時間平均池化，得到 [B, wav2vec_dim]
        global_vec = wav2vec_feat.mean(dim=1)
        global_out, global_gating = self.global_moe(global_vec)  # [B, fusion_hidden_dim]
        # 融合：
        fused_feat = self.gated_fusion(local_out, global_out)  # [B, fusion_hidden_dim]
        
        # 分類：
        logits = self.classifier(fused_feat)  # [B, num_classes]
        return logits, local_gating, global_gating

class MaxFeatureMap2D(nn.Module):
    """ Max feature map (along 2D)
    MaxFeatureMap2D(max_dim=1)
    l_conv2d = MaxFeatureMap2D(1)
    data_in = torch.rand([1, 4, 5, 5])
    data_out = l_conv2d(data_in)
    Input:
    ------
    data_in: tensor of shape (batch, channel, ...)
    Output:
    -------
    data_out: tensor of shape (batch, channel//2, ...)
    Note
    ----
    By default, Max-feature-map is on channel dimension,
    and maxout is used on (channel ...)
    """

    def __init__(self, max_dim=1):
        super(MaxFeatureMap2D, self).__init__()
        self.max_dim = max_dim

    def forward(self, inputs):
        # suppose inputs (batchsize, channel, length, dim)

        shape = list(inputs.size())

        if self.max_dim >= len(shape):
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But input has %d dimensions" % (len(shape)))
        if shape[self.max_dim] // 2 * 2 != shape[self.max_dim]:
            print("MaxFeatureMap: maximize on %d dim" % (self.max_dim))
            print("But this dimension has an odd number of data")
        shape[self.max_dim] = shape[self.max_dim] // 2
        shape.insert(self.max_dim, 2)

        # view to (batchsize, 2, channel//2, ...)
        # maximize on the 2nd dim
        m, i = inputs.view(*shape).max(self.max_dim)
        return m
        
class LCNN(nn.Module):
    def __init__(self, enc_dim):
        super(LCNN, self).__init__()
        self.enc_dim = enc_dim
        self.conv1 = nn.Sequential(nn.Conv2d(1, 64, (5, 5), 1, padding=(2, 2)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 1), (2, 1)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv3 = nn.Sequential(nn.Conv2d(32, 96, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 1), (2, 1)),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv4 = nn.Sequential(nn.Conv2d(48, 96, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(48, affine=False))
        self.conv5 = nn.Sequential(nn.Conv2d(48, 128, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 1), (2, 1)))
        self.conv6 = nn.Sequential(nn.Conv2d(64, 128, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(64, affine=False))
        self.conv7 = nn.Sequential(nn.Conv2d(64, 64, (3, 3), 1, padding=(1, 1)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv8 = nn.Sequential(nn.Conv2d(32, 64, (1, 1), 1, padding=(0, 0)),
                                   MaxFeatureMap2D(),
                                   nn.BatchNorm2d(32, affine=False))
        self.conv9 = nn.Sequential(nn.Conv2d(32, 64, (3, 3), 1, padding=[1, 1]),
                                   MaxFeatureMap2D(),
                                   nn.MaxPool2d((2, 1), (2, 1)))
        self.out = nn.Sequential(nn.Dropout(0.7),
                                 nn.Linear(20736, 128),
                                 MaxFeatureMap2D(),
                                 nn.Linear(64, self.enc_dim))

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        feat = torch.flatten(x, 1)
        feat = self.out(feat)

        return feat