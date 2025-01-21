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