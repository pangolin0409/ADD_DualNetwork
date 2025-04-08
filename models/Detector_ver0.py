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
    def __init__(self, freq_dim, time_dim, fusion_dim, seq_len):
        super(GatedFusion, self).__init__()
        self.seq_len = seq_len  # 確保時間對齊
        self.freq_to_gate = nn.Linear(freq_dim, time_dim)
        self.time_to_gate = nn.Linear(time_dim, freq_dim)

        # 門控機制
        self.gate_freq = nn.Sequential(
            nn.Linear(freq_dim, freq_dim), 
            nn.Sigmoid()
        )
        self.gate_time = nn.Sequential(
            nn.Linear(time_dim, time_dim), 
            nn.Sigmoid()
        )

        # 融合層 (確保時間維度不變)
        self.fusion = nn.Linear(freq_dim + time_dim, fusion_dim)

    def forward(self, F_freq, F_time):
        """
        F_freq: [batch, freq_dim, time]
        F_time: [batch, time_dim, time]
        """
        # **確保頻域與時域的維度對齊**
        F_freq = F_freq.permute(0, 2, 1)  # [batch, time, freq_dim]
        F_time = F_time.permute(0, 2, 1)  # [batch, time, time_dim]

        # **門控調制 (Gating)**
        G_time = self.gate_time(F_time)  # 控制時域
        G_freq = self.gate_freq(F_freq)  # 控制頻域

        # **動態調制**
        F_freq_mod = F_freq * G_freq
        F_time_mod = F_time * G_time

        # **融合特徵**
        F_fused = torch.cat([F_freq_mod, F_time_mod], dim=-1)  # [batch, time, freq_dim + time_dim]
        F_fused = self.fusion(F_fused)  # [batch, time, fusion_dim]

        return F_fused.permute(0, 2, 1)  # [batch, fusion_dim, time]


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
    
# 6. 完整 MVMS-GFES 模型（初步實驗版本）
class Detector(nn.Module):
    def __init__(self, aasist_encoder, sinc_filters=64, wav2vec_dim=1024, fusion_hidden_dim=128, seq_len=201):
        super(Detector, self).__init__()
        self.seq_len = seq_len  # 設定時間長度
        # SincConv
        self.sinc_net = SincConv(out_channels=sinc_filters, kernel_size=251, sample_rate=16000)

        # 注意機制
        self.freq_attention = MultiChannelAttention(sinc_filters)
        self.time_attention = MultiScaleAttention(wav2vec_dim)

        # 門控融合 (確保時間對齊)
        self.fusion = GatedFusion(sinc_filters, wav2vec_dim, fusion_hidden_dim, seq_len=self.seq_len)

        # 分類器
        self.classifier = aasist_encoder
        self.fc = nn.Linear(160, 2)

        # **自適應池化來確保時間對齊**
        self.adaptive_pool = nn.AdaptiveAvgPool1d(output_size=self.seq_len)

    def forward(self, wave, wav2vec_ft):
        # audio: [batch, 1, time]
        wave = wave.unsqueeze(1)  # 添加通道維度
        # **頻域分支**
        sinc_out = F.relu(self.sinc_net(wave))  # [batch, sinc_filters, time]
        F_freq = self.freq_attention(sinc_out)  # [batch, sinc_filters, time]
        
        # **時域分支**
        F_time = self.time_attention(wav2vec_ft)  # [batch, seq_len, wav2vec_dim]
        print(F_time)
        F_time = F_time.permute(0, 2, 1)  # [batch, wav2vec_dim, seq_len]

        # **時間對齊：AdaptiveAvgPool**
        F_freq = self.adaptive_pool(F_freq)  # [batch, sinc_filters, seq_len]
        F_time = self.adaptive_pool(F_time)  # [batch, wav2vec_dim, seq_len]
        # **門控融合**
        F_fused = self.fusion(F_freq, F_time)  # [batch, fusion_hidden_dim, seq_len]
        # 加入通道維度
        F_fused = F_fused.unsqueeze(1)  # [batch, 1, fusion_dim, time]
        # **分類**
        logits = self.classifier(F_fused)  # [batch, num_classes, seq_len]
        logits = self.fc(logits)  # [batch, num_classes]
        
        return logits