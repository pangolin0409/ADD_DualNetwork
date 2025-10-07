@echo off
REM === 初始化 conda 環境 ===
CALL conda remove -n audio --all -y
CALL conda create -y -n audio python=3.10

REM === 啟用環境（一定要 call conda.bat）===
CALL conda activate audio

REM === 安裝套件 ===
python -m pip install pip==23.0
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install librosa==0.10.0
pip install soundfile==0.12.0
pip install fastapi==0.116.1
pip install python-dotenv==1.1.1
pip install uvicorn==0.35.0
pip install scipy==1.15.3
pip install python-multipart==0.0.20
pip install --editable ./fairseq-a54021305d6b3c4c5959ac9395135f63202db8f1
pip install numpy==1.23.3


PAUSE
