import torch.nn as nn
from transformers import AutoConfig, AutoModel

class ExpertMLP(nn.Module):
    """
    Expert MLP: Uses LLM's feed-forward layers (pre-trained).
    """
    def __init__(self, model_name="gpt2", model_dir="./pretrained_models/gpt2_local"):
        super(ExpertMLP, self).__init__()
        # Load LLM configuration and model
        self.config = AutoConfig.from_pretrained(model_name, cache_dir=model_dir)
        self.llm = AutoModel.from_pretrained(model_name, cache_dir=model_dir, config=self.config)

        # Extract the feed-forward layer (MLP) structure
        self.mlp_layers = nn.ModuleList([
            block.mlp for block in self.llm.h
        ])

        # Optionally freeze LLM weights if only using MLP
        for param in self.llm.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Args:
            x: Input features [batch_size, seq_len, hidden_dim].
        Returns:
            Projected features through LLM MLP.
        """
        # Pass through LLM's MLP layers
        for mlp in self.mlp_layers:
            x = mlp(x)
        return x