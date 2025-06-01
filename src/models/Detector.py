import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.cluster import KMeans
from typing import Dict, Optional
import random
class ExpertPrototypeBank:
    def __init__(self, feature_dim, momentum=0.01, device='cuda'):
        self.feature_dim = feature_dim
        self.momentum = momentum
        self.device = device
        self.prototypes = torch.zeros(2, feature_dim, device=device)

    def _compute_batch_proto(self, features, labels):
        # features: [B, D], labels: [B]
        proto_dict = {}
        for class_id in [0, 1]:
            mask = (labels == class_id)
            if mask.sum() == 0:
                continue
            feats = features[mask]
            proto_dict[class_id] = feats.mean(dim=0)  # [D]
        return proto_dict

    def initialize(self, features, labels):
        proto_dict = self._compute_batch_proto(features, labels)
        for class_id, proto in proto_dict.items():
            self.prototypes[class_id] = proto

    def update(self, features, labels):
        proto_dict = self._compute_batch_proto(features, labels)
        for class_id, new_proto in proto_dict.items():
            self.prototypes[class_id] = (
                (1 - self.momentum) * self.prototypes[class_id] +
                self.momentum * new_proto
            )

    def get(self):
        return self.prototypes

class RoutingPreprocessor(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        x = self.norm(x)
        x = self.proj(x)
        return x

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

    def forward(self, x, temp):
        if temp is None:
            temp = self.router_temperature
        out = self.dropout(self.relu(self.fc1(x)))  # (batch, hidden_dim)
        logits = self.fc2(out)  # (batch, num_experts)
        noise_scale = F.softplus(self.noise_linear(x))  # 保證是正值
        noise = torch.randn_like(logits) * noise_scale
        logits = logits + noise
        routing = self.sigmoid(logits / temp)
        return routing

###########################################
# Classifier 模組
###########################################
class Classifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.2)  # 0.2~0.3 是較穩的值
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.dropout(x)
        x = self.fc2(x)
        return x

class CosineClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
        nn.init.xavier_uniform_(self.weight)

        self.scale = nn.Parameter(torch.tensor(30.0)) 

    def forward(self, x):
        x = F.normalize(x, dim=-1)                  # (B, D)
        w = F.normalize(self.weight, dim=-1)        # (C, D)
        cos_sim = torch.matmul(x, w.t())            # (B, C)

        return self.scale * cos_sim                 # (B, C)

###########################################
# Detector 模組：結合 SparseMoE 與 Classifier
###########################################
class Detector(nn.Module):
    def __init__(self, encoder_dim=1024, num_experts=24, num_classes=2, max_temp=2.5, min_temp=1.0, warmup_epochs=10, processor=None, onnx_session=None):
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
        self.router = SigmoidRouter((encoder_dim*num_experts)//2, num_experts, router_temperature=max_temp)

        self.max_temp = max_temp
        self.min_temp = min_temp
        self.warmup_epochs = warmup_epochs
        self.pre_norm = nn.LayerNorm(encoder_dim//2)
        self.classifier = Classifier(input_dim=encoder_dim//2, num_classes=num_classes)

    def forward(self, wave, epoch=5, temp=None, alpha=None):
        temp = self.temp_schedule(epoch, temp=temp)
        alpha = self.blend_schedule(epoch, alpha=alpha)
    
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

        uniform = torch.ones(B, self.num_experts, device=pooled.device) / self.num_experts
        learned = self.router(router_input, temp)  # (B, num_experts)
        routing_weights = alpha * learned + (1 - alpha) * uniform

        # -- Fusion
        fused = (pooled * routing_weights.unsqueeze(-1)).sum(dim=1)  # (B, expert_dim)

        fused = self.pre_norm(fused)
        logits = self.classifier(fused)

        return logits, routing_weights, fused, pooled

    def temp_schedule(self, epoch, temp=None):
        if temp is not None:
            if epoch >= 5:
                return temp
        return self.max_temp - (self.max_temp - self.min_temp) * (epoch / self.warmup_epochs)
        
    def blend_schedule(self, epoch, start_alpha=0.1, end_alpha=0.55, alpha=None):
        if alpha is not None:
            if epoch >= 5:
                return alpha
        return start_alpha + (end_alpha - start_alpha) * (epoch / self.warmup_epochs)

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
    
    # def extract_features_from_onnx(self, waveform):
    #     """
    #     使用 ONNX 將 raw waveform 轉成 hidden states (25 layers)
    #     - waveform: (B, T) 的 torch.Tensor，已經正規化過的 audio 資料
    #     """
    #     if isinstance(waveform, torch.Tensor):
    #         waveform = waveform.cpu().numpy()
        
    #     input_values = waveform.astype(np.float16)
    #     attention_mask = np.ones_like(input_values, dtype=np.int64)  # 全部都是有效 token

    #     # ONNX forward
    #     outputs = self.session.run(None, {
    #         "input_values": input_values,
    #         "attention_mask": attention_mask
    #     })

    #     # (25, B, T, D) → (B, 25, T, D)
    #     all_hidden_states = np.array(outputs[1])
    #     all_hidden_states = np.transpose(all_hidden_states, (1, 0, 2, 3))

    #     return torch.tensor(all_hidden_states, dtype=torch.float32).to(
    #         torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     )
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