import torch
import torch.nn as nn
from transformers import Blip2QFormerModel, Blip2Config

class QFormerConnector(nn.Module):
    def __init__(self, input_dim, num_queries=4):
        super().__init__()
        
        # 使用預設 Blip2 config（例如 "Salesforce/blip2-opt-2.7b"）
        config = Blip2Config.from_pretrained("Salesforce/blip2-opt-2.7b")
        self.qformer = Blip2QFormerModel(config.qformer_config)

        # Q-Former queries = 768
        hidden_size = config.qformer_config.hidden_size  # 通常=768
        self.query_tokens = nn.Parameter(torch.randn(1, num_queries, hidden_size))

        # 關鍵：要把 x 投影到 1408 (encoder_hidden_size) 
        encoder_size = config.qformer_config.encoder_hidden_size  #=1408
        self.projection = nn.Linear(input_dim, encoder_size)

    def forward(self, x, attention_mask=None):
        # x shape: [batch, seq_len, input_dim] or [batch, input_dim]
        # 投影成 [batch, seq_len, 1408]
        x = self.projection(x)

        # 若無 mask, 則全部有效
        if attention_mask is None:
            attention_mask = torch.ones(x.size(0), x.size(1), dtype=torch.long, device=x.device)

        # forward
        outputs = self.qformer(
            query_embeds=self.query_tokens.expand(x.size(0), -1, -1), 
            encoder_hidden_states=x, 
            encoder_attention_mask=attention_mask, 
            return_dict=True
        )
        pooled_output = outputs.last_hidden_state
        return pooled_output

class MLPConnector(nn.Module):
    def __init__(self, input_dim, output_dim=768):
        super(MLPConnector, self).__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    
    def forward(self, x):
        if x.dim() == 3:
            # [batch_size, seq_len, input_dim]
            # 如果需要投影
            if self.proj is not None:
                x = self.proj(x)  # -> [batch_size, seq_len, target_dim]
            # 不做 pool，保留序列給 Q-Former
            return x

        elif x.dim() == 2:
            # [batch_size, input_dim] (比如 rawnet3 輸出單向量)
            # 如果要讓 Q-Former 處理，也得讓它有 seq_len 維度 (seq_len=1)
            x = x.unsqueeze(1)  # -> [batch_size, 1, input_dim]
            if self.proj is not None:
                x = self.proj(x)  # -> [batch_size, 1, target_dim]
            return x

        return x


class ConvPoolingConnector(nn.Module):
    def __init__(self, input_dim, output_dim=768, kernel_size=3, pooling_size=2):
        super(ConvPoolingConnector, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, output_dim, kernel_size, padding=kernel_size//2),
            nn.ReLU(),
            nn.MaxPool1d(pooling_size),
            nn.Conv1d(output_dim, output_dim, kernel_size, padding=kernel_size//2),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)