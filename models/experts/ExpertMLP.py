import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel

class ExpertMLP(nn.Module):
    def __init__(self, model_name="gpt2", model_dir="./pretrained_models/gpt2_local"):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=model_dir)
        # 載入 GPT2Model，但我們只用其中的 blocks, ln_f 等
        self.llm = AutoModel.from_pretrained(model_name, cache_dir=model_dir, config=self.config)

        # 冷凍 GPT-2 參數
        for param in self.llm.parameters():
            param.requires_grad = False

        # GPT-2 的 block 層 (list of GPT2Block)
        self.blocks = self.llm.h
        # 最後 layernorm
        self.final_ln = self.llm.ln_f

        # 用於加權匯集各層 hidden state
        self.layer_weights = nn.Parameter(torch.ones(len(self.blocks) + 2))

        # 輸出線性投影
        self.output_projection = nn.Linear(self.config.hidden_size, self.config.hidden_size)

    def forward(self, hidden_states):
        """
        hidden_states: [batch, seq_len, hidden_dim]
        回傳: [batch, seq_len, hidden_dim]
        """

        all_hidden_states = [hidden_states]  # 保存每層輸出(含初始)

        # 直接以 3D 形式傳入 GPT2Block
        for block in self.blocks:
            # block.forward 回傳 (hidden_states, presents, ...)
            hidden_states = block(hidden_states)[0]  # 只取 hidden_states
            all_hidden_states.append(hidden_states)

        # 通過最後的 layernorm
        hidden_states = self.final_ln(hidden_states)
        all_hidden_states.append(hidden_states)

        # all_hidden_states: list of [b, s, h]
        layer_outputs = torch.stack(all_hidden_states, dim=0)  # shape [L, b, s, h]

        # 按照 layer_weights 做加權
        weights = self.layer_weights.softmax(dim=0)  # [L]
        weighted_output = torch.einsum("l,lbsh->bsh", weights, layer_outputs)  # [b, s, h]

        # 最後投影
        out = self.output_projection(weighted_output)  # [b, s, h]
        return out
