import torch
import torch.nn as nn
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