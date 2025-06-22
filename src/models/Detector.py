import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Optional
import random
import math

class WaveformAugmentor:
    def __init__(self, manipulation_pool: Dict[str, Optional[torch.nn.Module]], 
                 activation_prob: float = 0.0, min_aug: int = 1, max_aug: int = 2, verbose: bool = False, target_len=64600):
        """
        Args:
            manipulation_pool: dict of augmentation name → module
            activation_prob: probability to activate augmentation per sample
            min_aug: minimum number of augmentations to apply
            max_aug: maximum number of augmentations to apply
            verbose: whether to print selected augmentations
        """
        self.pool = {k: v for k, v in manipulation_pool.items() if v is not None}
        self.activation_prob = activation_prob
        self.min_aug = min_aug
        self.max_aug = max_aug
        self.verbose = verbose
        self.target_len = target_len  # 目標長度，預設為64600

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        """
        Apply augmentation to a waveform with probability `activation_prob`.
        """
        if self.activation_prob <= 0 or len(self.pool) == 0:
            return wav

        if random.random() > self.activation_prob:
            return wav

        aug_count = random.randint(self.min_aug, self.max_aug)
        selected_keys = random.sample(list(self.pool.keys()), aug_count)

        if self.verbose:
            print(f"[AUG] Epoch Aug Prob: {self.activation_prob}, Selected: {selected_keys}")

        for key in selected_keys:
            aug = self.pool[key]
            try:
                wav = aug(wav)
            except Exception as e:
                if self.verbose:
                    print(f"[{key}] augmentation failed: {e}")
                continue

        wav = self.pad_or_clip(wav)
        return wav

    def pad_or_clip(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.shape[-1] < self.target_len:
            return F.pad(audio, (0, self.target_len - audio.shape[-1]))
        elif audio.shape[-1] > self.target_len:
            start = random.randint(0, audio.shape[-1] - self.target_len)
            return audio[..., start:start + self.target_len]
        return audio

    def update(self, epoch):
        if epoch < 1:
            self.activation_prob = 0.2
            self.min_aug = 1
            self.max_aug = 1
        elif epoch < 4:
            self.activation_prob = 0.4
            self.min_aug = 1
            self.max_aug = 1
        else:
            self.activation_prob = 0.6
            self.min_aug = 1
            self.max_aug = 2

###########################################
# Router 模組（包含 Data Augmentation）
###########################################
class LayerAwareRouter(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, router_temperature=2.0):
        super().__init__()
        self.temperature = router_temperature

        # 可以用一個大的共享 MLP，也可以每層獨立 MLP（先做共享）
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.GELU()
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, temp=None):
        # x: [B, L, D]
        if temp is None:
            temp = self.temperature

        B, L, D = x.shape
        x = self.fc1(x)           # [B, L, hidden_dim]
        x = self.relu(x)
        x = self.dropout(x)
        logits = self.fc2(x).squeeze(-1)  # [B, L]
        scores = self.sigmoid(logits / temp)  # αₗ (soft routing weights)
        return scores  # [B, L]

class TimeAttentionPool(nn.Module):
    def __init__(self, dim, heads=2):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True, dropout=0.1)
        # 使用 LayerNorm 來正規化輸入
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):  # x: [B, T, D]
        x_norm = self.norm(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)  # contextualize over time
        pooled = attn_out.mean(dim=1)  # [B, D]
        return pooled
    
###########################################
# Classifier 模組
###########################################
class Classifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=2):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, input_dim//2)
        self.act1 = nn.GELU()
        self.dropout1 = nn.Dropout(0.2)

        self.fc2 = nn.Linear(input_dim//2, input_dim//4)
        self.act2 = nn.GELU()
        self.dropout2 = nn.Dropout(0.1)

        self.fc3 = nn.Linear(input_dim//4, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

###########################################
# Detector 模組：結合 SparseMoE 與 Classifier
###########################################
class Detector(nn.Module):
    def __init__(self, encoder_dim=1024, num_experts=24, num_classes=2, max_temp=2.5, min_temp=1.0, start_alpha=0.2, end_alpha=0.8,warmup_epochs=10, processor=None, onnx_session=None):
        super().__init__()
        self.num_experts = num_experts
        self.processor = processor
        self.session = onnx_session
        self.max_temp = max_temp
        self.min_temp = min_temp
        self.start_alpha = start_alpha
        self.end_alpha = end_alpha
        self.warmup_epochs = warmup_epochs

        self.shared_mlp =  nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(encoder_dim // 2, encoder_dim // 2)
        )
        self.groups = 4
        self.mlp_alpha = nn.Parameter(torch.ones(self.num_experts))  # 24 層
        # 一個大MLP處理所有層
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(encoder_dim // 2, encoder_dim // 2)
            ) for _ in range(self.groups)
        ])

        # Attention pooling也是一個大的Linear
        self.time_attn_pool = TimeAttentionPool(encoder_dim // 2, heads=2)
        # Router
        self.router = LayerAwareRouter(encoder_dim // 2, router_temperature=max_temp)
        self.post_norm = nn.LayerNorm(encoder_dim//2)
        self.post_norm_final = nn.LayerNorm(encoder_dim//2)
        self.classifier = Classifier(input_dim=encoder_dim//2, num_classes=num_classes)

    def forward(self, wave, epoch=5, temp=None, alpha=None):
        temp = self.temp_schedule(epoch, temp=temp)
        alpha = self.blend_schedule(epoch, alpha=alpha)
    
        all_hidden_states = self.extract_features_from_onnx(wave)  
        layer_outputs = all_hidden_states[:, 1:]  # [B, 24, T, D]
        B, L, T, D = layer_outputs.shape

        out = []
        grouped_mlp_outputs = []
        shared_mlp_outputs = []
        group_size = L // self.groups

        for i in range(self.num_experts):
            group_idx = i // group_size
            shared_mlp_output = self.shared_mlp(layer_outputs[:, i])  # [B, T, D']
            group_mlp_output = self.mlps[group_idx](layer_outputs[:, i])  # [B, T, D']
            weighted_mlp_output = shared_mlp_output + self.mlp_alpha[i] * group_mlp_output  # [B, T, D']
            grouped_mlp_outputs.append(weighted_mlp_output)
            shared_mlp_outputs.append(shared_mlp_output)
            # Time pooling
            time_pooled = self.time_attn_pool(weighted_mlp_output)  # [B, D']
            out.append(time_pooled)

        linear_outputs = torch.stack(grouped_mlp_outputs, dim=1)  # [B, 24, T, D']
        shared_mlp_outputs = torch.stack(shared_mlp_outputs, dim=1)  # [B, 24, T, D']
        time_pooled = torch.stack(out, dim=1)  # [B, 24, D']
        
        # --- Branch 1: Layer-Aware Router ---
        uniform = torch.ones(B, self.num_experts, device=time_pooled.device) / self.num_experts
        learned = self.router(time_pooled, temp)  # [B, 24]
        routing_weights = alpha * learned + (1 - alpha) * uniform
        layer_fusion_feat = (linear_outputs * routing_weights.unsqueeze(-1).unsqueeze(-1)).sum(dim=1) # [B, T, D']
        layer_fusion_feat = self.post_norm(layer_fusion_feat)
        layer_fusion_feat = layer_fusion_feat.mean(dim=1)  # [B, D']

        # --- Branch 2: Time Context Pooling ---
        shared_mlp_outputs = shared_mlp_outputs.mean(dim=1)  # [B, T, D']
        time_context_feat = self.time_attn_pool(shared_mlp_outputs)  # [B, D']

        layer_fusion_feat = self.post_norm_final(layer_fusion_feat)
        time_context_feat = self.post_norm_final(time_context_feat)

        fused_all = layer_fusion_feat + time_context_feat  # [B, D']
        logits = self.classifier(fused_all)

        return logits, routing_weights, fused_all, time_pooled

    def temp_schedule(self, epoch, temp=None):
        if temp is not None:
            return temp
        progress = min(epoch / self.warmup_epochs, 1.0)
        return max(self.min_temp, self.max_temp - (self.max_temp - self.min_temp) * progress)

    def blend_schedule(self, epoch, alpha=None):
        if alpha is not None:
            return alpha
        progress = min(epoch / self.warmup_epochs, 1.0)
        return min(self.end_alpha, self.start_alpha + (self.end_alpha - self.start_alpha) * progress)

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