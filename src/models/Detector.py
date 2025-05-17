import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import wandb

class AdaptiveMarginManager:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.real_norm_sum = 0.0
        self.real_count = 0
        self.fake_norm_sum = 0.0
        self.fake_count = 0

        self.real_mean = 0.0
        self.fake_mean = 0.0
        self.margin = 0.0

    def update(self, features: torch.Tensor, labels: torch.Tensor):
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        real_feat = features[real_mask]
        fake_feat = features[fake_mask]

        if real_feat.shape[0] > 0:
            self.real_norm_sum += real_feat.norm(p=2, dim=1).sum().item()
            self.real_count += real_feat.shape[0]

        if fake_feat.shape[0] > 0:
            self.fake_norm_sum += fake_feat.norm(p=2, dim=1).sum().item()
            self.fake_count += fake_feat.shape[0]

    def compute_margin(self):
        if self.real_count > 0:
            self.real_mean = self.real_norm_sum / self.real_count
        if self.fake_count > 0:
            self.fake_mean = self.fake_norm_sum / self.fake_count

        if self.real_mean > 0 and self.fake_mean > 0:
            scale = (1 - self.alpha) + self.alpha * (self.fake_mean / self.real_mean)
            self.margin = scale * self.real_mean

    def update_margin(self, features: torch.Tensor, labels: torch.Tensor, momentum=0.9):
        real_mask = (labels == 0)
        fake_mask = (labels == 1)

        real_feat = features[real_mask]
        fake_feat = features[fake_mask]

        real_sum = real_feat.norm(p=2, dim=1).sum().item()
        fake_sum = fake_feat.norm(p=2, dim=1).sum().item()

        real_count = real_feat.shape[0]
        fake_count = fake_feat.shape[0]

        # 滑動平均更新 sum 和 count
        self.real_norm_sum = momentum * self.real_norm_sum + (1 - momentum) * real_sum
        self.fake_norm_sum = momentum * self.fake_norm_sum + (1 - momentum) * fake_sum
        self.real_count = momentum * self.real_count + (1 - momentum) * real_count
        self.fake_count = momentum * self.fake_count + (1 - momentum) * fake_count

        self.compute_margin()

    def get_margin(self, fallback=25.0):
        if self.margin > 0:
            return self.margin
        else:
            print(f"[MarginManager] ⚠️ Margin 未初始化，使用 fallback: {fallback}")
            return fallback

    def log_info(self, fallback=25.0):
        margin_value = self.margin if self.margin > 0 else fallback
        print(f"[MarginManager] real_mean={self.real_mean:.4f}, fake_mean={self.fake_mean:.4f}, margin={margin_value:.4f}")
        return {
            "margin/real_mean": self.real_mean,
            "margin/fake_mean": self.fake_mean,
            "margin/value": margin_value,
            "margin/fallback_used": self.margin == 0
        }


###########################################
# Router 模組（包含 Data Augmentation）
###########################################
class SigmoidRouter(nn.Module):
    def __init__(self, input_dim, num_experts, hidden_dim=128, router_temperature=2.0):
        super(SigmoidRouter, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_dim, num_experts)
        self.noise_linear = nn.Linear(input_dim, num_experts)
        self.router_temperature = router_temperature
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.dropout(self.relu(self.fc1(x)))  # (batch, hidden_dim)
        logits = self.fc2(out)  # (batch, num_experts)
        noise_scale = F.softplus(self.noise_linear(x))  # 保證是正值
        noise = torch.randn_like(logits) * noise_scale
        logits = logits + noise
        routing = self.sigmoid(logits / self.router_temperature)
        return routing

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
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(encoder_dim, encoder_dim // 2),
                nn.ReLU(),
                nn.Linear(encoder_dim // 2, encoder_dim // 2)
            ) for _ in range(24)
        ])

        # Attention pooling也是一個大的Linear
        self.attn_score = nn.Linear(encoder_dim//2, 1)

        # Router
        self.router = SigmoidRouter((encoder_dim*num_experts)//2, num_experts, router_temperature=router_temperature)

        self.router_temperature = router_temperature
        self.pre_norm = nn.LayerNorm(encoder_dim//2)
        self.classifier = Classifier(input_dim=encoder_dim//2, num_classes=num_classes)

    def forward(self, wave, is_aug=False, aug_mask=None, aug_method="additive_noise", router_mode="learned"):
       
        all_hidden_states = self.extract_features_from_onnx(wave)  
        layer_outputs = all_hidden_states[:, 1:]  # [B, 24, T, D]
        B, L, T, D = layer_outputs.shape

        # -- MLP處理
        out = []
        for i, mlp in enumerate(self.mlps):
            out.append(mlp(layer_outputs[:, i]))  # x: [B, 24, T, D] → x[:, i]: [B, T, D]
        x = torch.stack(out, dim=1)  # [B, 24, T, D']

        # -- Attention Pooling
        score = self.attn_score(x)  # (B, 24, T, 1)
        weight = torch.softmax(score, dim=2)  # (B, 24, T, 1)
        pooled = (x * weight).sum(dim=2)  # (B, 24, expert_dim)

        # -- Router
        router_input = pooled.reshape(B, -1)  # (B, 24*expert_dim)

        if router_mode == "uniform":
            routing_weights = torch.ones(B, self.num_experts, device=pooled.device) / self.num_experts
        else:
            routing_weights = self.router(router_input)  # (B, num_experts)

        # -- Fusion
        fused = (pooled * routing_weights.unsqueeze(-1)).sum(dim=1)  # (B, expert_dim)

        fused = self.apply_augmentation(fused, is_aug, aug_mask, aug_method)  # (B, expert_dim)
        fused = self.pre_norm(fused)
        logits = self.classifier(fused)

        return logits, routing_weights, fused, pooled

    def set_router_temperature(self, temperature):
        self.router.router_temperature = temperature

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