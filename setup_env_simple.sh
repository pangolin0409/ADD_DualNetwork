#!/bin/bash

# === 簡化版環境設置腳本 ===
# 使用 requirements.txt 安裝所有依賴

set -e  # 遇到錯誤時停止執行

echo "🚀 開始設置音訊深度偽造檢測項目環境（簡化版）..."

# === 檢查 conda 是否已安裝 ===
if ! command -v conda &> /dev/null; then
    echo "❌ Conda 未安裝，請先安裝 Anaconda 或 Miniconda"
    echo "下載地址: https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

echo "✅ Conda 已安裝"

# === 初始化 conda 環境 ===
echo "🔄 正在設置 conda 環境..."

# 移除現有環境（如果存在）
echo "刪除現有環境 'audio'（如果存在）..."
conda remove -n audio --all -y 2>/dev/null || true

# 創建新環境
echo "創建新的 conda 環境 'audio'..."
conda create -y -n audio python=3.10

# === 安裝 PyTorch ===
echo "📦 安裝 PyTorch..."
python -m pip install pip==23.0
pip install torch torchvision
pip install librosa==0.10.0
pip install soundfile==0.12.0
pip install fastapi==0.116.1
pip install python-dotenv==1.1.1
pip install uvicorn==0.35.0
pip install scipy==1.15.3
pip install python-multipart==0.0.20
pip install --editable ./fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install numpy==1.23.3
pip install mlflow==2.8.1

# === 驗證安裝 ===
echo "🔍 驗證安裝..."
conda run -n audio python -c "
import torch
import mlflow
import librosa
import soundfile
import numpy as np
import pandas as pd
print('✅ 所有主要套件安裝成功')
print(f'PyTorch 版本: {torch.__version__}')
print(f'MLflow 版本: {mlflow.__version__}')
print(f'NumPy 版本: {np.__version__}')
print(f'Pandas 版本: {pd.__version__}')
"

# === 顯示使用說明 ===
echo ""
echo "🎉 環境設置完成！"
echo "=================="
echo ""
echo "📋 使用說明："
echo "1. 激活環境: conda activate audio"
echo "2. 啟動 MLflow UI: python scripts/mlflow_example.py server"
echo "3. 運行訓練: python main.py --experiment train --config ./config/base.yml"
echo "4. 運行推理: python main.py --experiment inference --config ./config/base.yml"
echo ""
echo "🌐 MLflow UI: http://localhost:5000"
echo "📚 詳細指南: 查看 MLFLOW_GUIDE.md"
echo ""

echo "腳本執行完成！"
