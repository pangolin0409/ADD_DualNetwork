import torch
import torch.nn as nn

class UnifiedSelfAttentionLayer(nn.Module):
    def __init__(self, input_dim: int, num_heads: int, num_layers: int):
        super(UnifiedSelfAttentionLayer, self).__init__()
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=input_dim, num_heads=num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.norm_layers = nn.ModuleList([nn.LayerNorm(input_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 多層自注意力
        for attn, norm in zip(self.attention_layers, self.norm_layers):
            attn_output, _ = attn(x, x, x)
            x = norm(attn_output + x)  # 殘差連接和歸一化
        return x

import torch
import torch.nn as nn

# Q-Former Connector
class QFormerConnector(nn.Module):
    def __init__(self, input_dim: int, query_dim: int, num_queries: int, num_heads: int, num_layers: int):
        super(QFormerConnector, self).__init__()
        self.num_queries = num_queries
        self.query_embeddings = nn.Parameter(torch.randn(num_queries, query_dim))

        # 使用多頭注意力進行融合
        self.attention = nn.MultiheadAttention(embed_dim=query_dim, num_heads=num_heads, batch_first=True)
        self.projection = nn.Linear(input_dim, query_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 投影到 query_dim
        x = self.projection(x)  # [batch_size, query_dim]
        
        # 將輸入擴展為與 query 嵌入匹配的形狀
        queries = self.query_embeddings.unsqueeze(0).expand(x.size(0), -1, -1)  # [batch_size, num_queries, query_dim]

        # 使用多頭注意力進行處理
        x, _ = self.attention(queries, x.unsqueeze(1), x.unsqueeze(1))  # Key 和 Value 來自輸入特徵
        return x

class ConnectorClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super(ConnectorClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)

# Router 定義
class Router(nn.Module):
    def __init__(self, input_dim: int, num_experts: int):
        super(Router, self).__init__()
        self.fc = nn.Linear(input_dim, num_experts)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor, k: int) -> tuple:
        routing_weights = self.softmax(self.fc(x))  # [batch_size, num_experts]
        top_k_weights, top_k_indices = torch.topk(routing_weights, k=k, dim=-1)  # 取 Top-K 的權重和索引
        return top_k_weights, top_k_indices


# Expert 定義
class Expert(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# MoE 定義
class SoftMoE(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int, output_dim: int):
        super(SoftMoE, self).__init__()
        self.router = Router(input_dim, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        routing_weights = self.router(x)  # [batch_size, num_experts]
        expert_outputs = torch.stack([expert(x) for expert in self.experts], dim=1)  # [batch_size, num_experts, output_dim]
        output = torch.sum(routing_weights.unsqueeze(-1) * expert_outputs, dim=1)  # 加權平均 [batch_size, output_dim]
        return output
    
class SparseMoE(nn.Module):
    def __init__(self, input_dim: int, num_experts: int, hidden_dim: int, output_dim: int, k: int):
        super(SparseMoE, self).__init__()
        self.router = Router(input_dim, num_experts)
        self.experts = nn.ModuleList([Expert(input_dim, hidden_dim, output_dim) for _ in range(num_experts)])
        self.k = k  # 每個樣本選擇的專家數量

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Router 的輸出
        top_k_weights, top_k_indices = self.router(x, k=self.k)  # [batch_size, k], [batch_size, k]
        
        # 構建批次輸出
        batch_size = x.size(0)
        expert_outputs = torch.zeros(batch_size, self.k, self.experts[0].fc2.out_features).to(x.device)  # [batch_size, k, output_dim]

        # 計算 Top-K 專家輸出
        for i in range(self.k):
            expert_idx = top_k_indices[:, i]  # [batch_size]
            for b in range(batch_size):
                expert_outputs[b, i, :] = self.experts[expert_idx[b]](x[b].unsqueeze(0))  # [1, output_dim]

        # 加權融合 Top-K 專家輸出
        weighted_outputs = (top_k_weights.unsqueeze(-1) * expert_outputs).sum(dim=1)  # [batch_size, output_dim]
        return weighted_outputs
