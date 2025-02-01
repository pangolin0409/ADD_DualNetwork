import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModel
from torch.nn import functional as F
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


class TDNNLayer(nn.Module):
    """
    基本 TDNN (Time Delay Neural Network) block
    dilation表示擴張卷積, kernel_size=3 or 5可再調整
    """
    def __init__(self, input_dim, output_dim, device, kernel_size=3, dilation=1):
        super().__init__()
        self.tdnn = nn.Conv1d(
            in_channels=input_dim,
            out_channels=output_dim,
            kernel_size=kernel_size,
            dilation=dilation,
        ).to(device)
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bn = nn.BatchNorm1d(output_dim)
        self.device = device
    def forward(self, x):
        # x shape: (B, T, F)
        # 動態調整 kernel_size，確保不超過 T
        kernel_size = min(self.kernel_size, x.size(1))
        
        # 重新初始化 Conv1d，確保 kernel_size 不超過 T
        if self.tdnn is None or self.tdnn.kernel_size[0] != kernel_size:
            self.tdnn = nn.Conv1d(
                in_channels=self.input_dim,
                out_channels=self.output_dim,
                kernel_size=kernel_size,
                dilation=self.dilation,
            ).to(self.device)

        # 需要先轉成 (B, F, T) 才能丟進 Conv1d
        x = x.transpose(1, 2)   # -> (B, F, T)
        x = self.tdnn(x)        # -> (B, output_dim, T')
        x = self.bn(x)
        x = F.relu(x)
        x = x.transpose(1, 2)   # -> (B, T', output_dim)
        return x

class AttentionPooling(nn.Module):
    """
    單頭注意力加權，將 (B, T, D) 序列壓縮為 (B, D)
    """
    def __init__(self, input_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        # x shape: (B, T, D)
        # 計算注意力權重 alpha
        alpha = self.attention(x)          # (B, T, 1)
        alpha = F.softmax(alpha, dim=1)    # (B, T, 1)

        # 加權求和
        x = x * alpha                      # (B, T, D)
        x = x.sum(dim=1)                   # (B, D)
        return x

class ExpertTDNN(nn.Module):
    """
    專家: TDNN 多層 + Attention Pooling
    """
    def __init__(self, input_dim, hidden_dim, device):
        super().__init__()
        # 這裡示範2層TDNN，可自行增減
        self.tdnn1 = TDNNLayer(input_dim, hidden_dim, kernel_size=3, dilation=1, device=device).to(device)
        self.tdnn2 = TDNNLayer(hidden_dim, hidden_dim, kernel_size=3, dilation=2, device=device).to(device)
        self.attention = AttentionPooling(hidden_dim).to(device)

    def forward(self, x):
        # x shape: (B, T, input_dim)
        x = self.tdnn1(x)      # -> (B, T', hidden_dim)
        x = self.tdnn2(x)      # -> (B, T'', hidden_dim)
        x = self.attention(x)  # -> (B, hidden_dim)
        return x  # 取pooling後的向量做後續分類