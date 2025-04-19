import os
import torch
from pathlib import Path
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from onnxruntime.quantization import quantize_dynamic, QuantType

class Wav2VecONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_values, attention_mask):
        output = self.model(input_values, attention_mask=attention_mask, output_hidden_states=True)
        return output.last_hidden_state, torch.stack(output.hidden_states)


# ==== 設定環境 ====
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
output_path = Path("./onnx_models")
output_path.mkdir(exist_ok=True)

# ==== 載入 Wav2Vec2 模型 ====
model_ckpt = './pretrained_models/wav2vec2-xls-r-300m'
processor = Wav2Vec2FeatureExtractor.from_pretrained(model_ckpt)
wav2vec2model = Wav2Vec2Model.from_pretrained(model_ckpt, output_hidden_states=True).to(device)
model = Wav2VecONNXWrapper(wav2vec2model).to(device)
model.eval()

# ==== 假輸入：模擬 1 秒音訊 ====
dummy_input = torch.randn(1, 16000).to(device)
dummy_mask = torch.ones(1, 16000, dtype=torch.long).to(device)

# ==== 匯出為 ONNX ====
onnx_path = output_path / "xlsr_base.onnx"
torch.onnx.export(
    model,
    args=(dummy_input, dummy_mask),
    f=str(onnx_path),
    input_names=["input_values", "attention_mask"],
    output_names=["last_hidden_state", "hidden_states"],
    opset_version=14,
    do_constant_folding=True,
    dynamic_axes={  # 讓 batch 維度為動態
        "input_values": {0: "batch_size", 1: "seq_len"},
        "attention_mask": {0: "batch_size", 1: "seq_len"}
    }
)
print("✅ ONNX model saved:", onnx_path)