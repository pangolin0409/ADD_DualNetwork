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
