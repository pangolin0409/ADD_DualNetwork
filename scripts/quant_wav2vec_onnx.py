from onnxconverter_common.float16 import *
from onnx import load_model, save_model
from pathlib import Path

fp32_model_path = Path("./onnx_models/xlsr_base.onnx")
fp16_model_path = Path("./onnx_models/xlsr_base_fp16.onnx")

# 讀取 ONNX 模型
model_fp32 = load_model(fp32_model_path)

# 轉換成 FP16，會自動處理敏感 layer（像 LayerNorm 用 FP32 保精度）
model_fp16 = convert_float_to_float16(model_fp32)

# 儲存新模型
save_model(model_fp16, fp16_model_path)

print("✅ FP16 模型已儲存到：", fp16_model_path)
