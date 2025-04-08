import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 1. 多尺度注意力機制
class MultiScaleAttention(nn.Module):
    def __init__(self, feature_dim=1024, num_heads=4, kernel_sizes=[3, 5, 7, 11]):
        super(MultiScaleAttention, self).__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # **確保 feature_dim 可被 kernel_sizes 整除**
        assert feature_dim % len(kernel_sizes) == 0, f"Feature dimension {feature_dim} 無法被 {len(kernel_sizes)} 整除！"
        conv_out_dim = feature_dim // len(kernel_sizes)  # 確保不會出現浮點數問題

        # **多尺度卷積**
        self.scale_convs = nn.ModuleList([
            nn.Conv1d(in_channels=feature_dim, out_channels=conv_out_dim, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])

        # **多頭自注意力**
        self.self_attn = nn.MultiheadAttention(embed_dim=feature_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: [batch, seq_len, feature_dim]
        """
        # **轉換成 Conv1D 需要的格式**：[batch, feature_dim, seq_len]
        x_conv = x.permute(0, 2, 1)  # [batch, feature_dim, seq_len]
        # **多尺度卷積**
        scale_features = [F.relu(conv(x_conv)) for conv in self.scale_convs]

        # **拼接特徵**
        x_scaled = torch.cat(scale_features, dim=1)  # [batch, feature_dim, seq_len]

        # **轉換回 Transformer 需要的格式**
        x_scaled = x_scaled.permute(2, 0, 1)  # [seq_len, batch, feature_dim]

        # **多頭自注意力**
        attn_output, _ = self.self_attn(x_scaled, x_scaled, x_scaled)

        # **殘差與正規化**
        attn_output = attn_output.permute(1, 0, 2)  # [batch, seq_len, feature_dim]
        x = self.norm(x + self.dropout(attn_output))

        return x  # [batch, seq_len, feature_dim]

# 2. 多通道注意力機制
# 挪用 SE-Net 的概念，對多通道特徵進行注意力加權
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
        # x: [batch, num_channels, time]
        # 先對每一個頻率通道進行平均池化
        x_squeezed = self.avg_pool(x).squeeze(-1)  # [batch, num_channels]
        # 透過全連接層計算注意力權重
        attn_weights = self.fc(x_squeezed).unsqueeze(-1)  # [batch, num_channels, 1]
        # 最後做 Scale，將注意力權重乘回原始特徵
        return x * attn_weights  # [batch, num_channels, time]

# 3. 門控融合模塊（雙向調製）
class GatedFusion(nn.Module):
    def __init__(self, freq_dim, time_dim, hidden_dim):
        super(GatedFusion, self).__init__()
        # self.freq_to_gate = nn.Linear(freq_dim, time_dim)
        # self.time_to_gate = nn.Linear(time_dim, freq_dim)
        # self.fusion = nn.Linear(freq_dim + time_dim, hidden_dim)
        
    def forward(self, F_freq, F_time):
        # F_freq: [batch, freq_dim], F_time: [batch, time_dim]
        # 時域的門控
        G_time = torch.sigmoid(F_time)  # [batch, freq_dim]
        # 頻域的門控
        G_freq = torch.sigmoid(F_freq)  # [batch, time_dim]
        
        # 頻域特徵被時域生成的權重調節，決定哪些頻域特徵更重要。
        F_freq_mod = F_freq * G_time  # [batch, freq_dim]
        # 時域特徵被頻域生成的權重調節，決定哪些時域特徵更重要。
        F_time_mod = F_time * G_freq  # [batch, time_dim]
        F_pre_fused = torch.cat([F_freq_mod, F_time_mod], dim=-1)  # [batch, hidden_dim]
        return F_pre_fused

# 4. 多尺度專家模塊
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
    
# 5. SincConv 模塊
class SincConv(nn.Module):
    def __init__(self, out_channels, kernel_size, sample_rate):
        super(SincConv, self).__init__()
        self.out_channels = out_channels
        self.kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        self.sample_rate = sample_rate

        # 允許 f1 和 f2 在訓練時自由學習
        self.f1 = nn.Parameter(torch.linspace(30, sample_rate//2 - 50, out_channels))
        self.f2 = nn.Parameter(torch.linspace(50, sample_rate//2, out_channels))

        # 時間軸，避免 NaN
        self.t = torch.arange(-(self.kernel_size // 2), (self.kernel_size // 2) + 1).float()
        self.t = self.t / (self.sample_rate + 1e-6)  # 避免除以 0
        self.t = self.t.unsqueeze(0).repeat(self.out_channels, 1)

    def sinc(self, x):
        eps = 1e-6  # 避免 0/0
        return torch.where(torch.abs(x) < eps, torch.ones_like(x), torch.sin(x) / (x + eps))

    def forward(self, x):
        self.t = self.t.to(x.device)

        # **在 forward() 限制 f1 和 f2，而不是 __init__()**
        self.f1.data = torch.clamp(self.f1, min=30, max=self.sample_rate//2 - 50)
        self.f2.data = torch.clamp(self.f2, min=50, max=self.sample_rate//2)

        filters = []
        for i in range(self.out_channels):
            # 計算 Sinc 濾波器
            low = 2 * self.f1[i] * self.sinc(2 * np.pi * self.f1[i] * self.t[i])
            high = 2 * self.f2[i] * self.sinc(2 * np.pi * self.f2[i] * self.t[i])
            h = high - low

            # 窗函數
            window = 0.54 - 0.46 * torch.cos(2 * np.pi * torch.arange(self.kernel_size) / (self.kernel_size - 1))
            h = h * window.to(x.device)

            filters.append(h.unsqueeze(0).unsqueeze(0))

        filters = torch.cat(filters, dim=0).to(x.device)
        return F.conv1d(x, filters, padding=self.kernel_size // 2)
    
# 6. 完整模型（初步實驗版本）
class Detector(nn.Module):
    def __init__(self, sinc_filters=64, wav2vec_dim=1024, fusion_hidden_dim=128, num_classes=2,
                 local_num_experts=3, global_num_experts=3, top_k=1):
        super(Detector, self).__init__()
        # SincConv
        self.sinc_net = SincConv(out_channels=sinc_filters, kernel_size=251, sample_rate=16000)
        
        # 注意機制
        self.freq_attention = MultiChannelAttention(sinc_filters)
        self.time_attention = MultiScaleAttention(wav2vec_dim)
        
        # Local MoE：將 LCNN 輸出向量融合，輸入維度 lcnn_enc_dim，輸出 fusion_hidden_dim
        self.local_moe = SparseMoE(in_dim=sinc_filters, out_dim=fusion_hidden_dim,
                             num_experts=local_num_experts, top_k=top_k)
        # Global MoE：對 wav2vec 特徵做時間平均，輸入維度 wav2vec_dim，輸出 fusion_hidden_dim
        self.global_moe = SparseMoE(in_dim=wav2vec_dim, out_dim=fusion_hidden_dim,
                              num_experts=global_num_experts, top_k=top_k)

        # 門控融合
        self.fusion = GatedFusion(sinc_filters, wav2vec_dim, fusion_hidden_dim)
        
        
        # 分類器：輸入融合後向量，輸出 num_classes
        self.classifier = nn.Sequential(
            nn.Linear(fusion_hidden_dim * 2, fusion_hidden_dim),
            nn.ReLU(),
            nn.Linear(fusion_hidden_dim, num_classes)
        )

    def forward(self, wave, wav2vec_ft):
        # audio: [batch, 1, time]

        wave = wave.unsqueeze(1)  # 添加通道維度
        # 頻域分支
        sinc_out = F.relu(self.sinc_net(wave))  # [batch, sinc_filters, time]
        # print(sinc_out)
        F_freq = self.freq_attention(sinc_out)  # [batch, sinc_filters, time]
        # print(F_freq)
        F_freq = F_freq.mean(dim=-1)  # [batch, sinc_filters]
        # print(F_freq)
        local_out, local_gating = self.local_moe(F_freq)  # [B, fusion_hidden_dim]

        # 時域分支
        F_time = self.time_attention(wav2vec_ft)  # [batch, seq_len, wav2vec_dim]
        F_time = F_time.mean(dim=1)  # [batch, wav2vec_dim]
        global_out, global_gating = self.global_moe(F_time)  # [B, fusion_hidden_dim]
        
        print(local_out.shape, global_out.shape)
        # 門控融合
        F_pre_fused = self.fusion(local_out, global_out)  # [batch, fusion_hidden_dim]
        
        # 分類
        logits = self.classifier(F_pre_fused)  # [batch, num_classes]
        return logits, local_gating, global_gating