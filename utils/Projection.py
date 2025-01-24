import torch
import torch.nn as nn

class Preprocessor(nn.Module):
    def __init__(self, modality, input_dim, target_dim):
        super(Preprocessor, self).__init__()
        self.modality = modality
        self.input_dim = input_dim
        self.target_dim = target_dim

        # (1) 不再做 AdaptiveAvgPool1d，把序列保留給 Q-Former
        # self.pool = nn.AdaptiveAvgPool1d(1)  # <-- 先註解/拿掉

        # (2) 若 input_dim != target_dim，做線性投影到 target_dim
        if input_dim != target_dim:
            self.proj = nn.Linear(input_dim, target_dim)
        else:
            self.proj = None

    def forward(self, x):
        """
        預期:
          - 如果 x.dim() == 3：表示 [batch_size, seq_len, input_dim]
            => 直接做線性投影 (若 self.proj 有定義)，得到 [batch_size, seq_len, target_dim]
          - 如果 x.dim() == 2：表示 [batch_size, input_dim] (single vector)
            => 看你要不要 unsqueeze(1) 成 [batch_size, 1, input_dim] 再投影
        """

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

        else:
            raise ValueError(f"Unsupported tensor shape {x.shape} in Preprocessor.")
