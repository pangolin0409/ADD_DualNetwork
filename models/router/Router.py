import torch
import torch.nn as nn
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
