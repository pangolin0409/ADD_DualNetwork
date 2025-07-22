# LARTAAD
**L**ayer-**A**ware **R**outing and **T**emporal **A**ttention for **A**udio **D**eepfakes

這是一個專為音訊深度偽造檢測（Audio Deepfake Detection）設計的專案，採用了專家混合（Mixture of Experts, MoE）架構來提升檢測的準確性和效率。

## 環境設置

1.  **複製專案庫**
    ```bash
    git clone <your-repository-url>
    cd your-repository
    ```

2.  **安裝依賴套件**
    本專案所需的所有 Python 套件都記錄在 `requirements.txt` 中。先用 anacoda 建立名稱叫 audio 虛擬環境，
    之後請執行以下命令進行安裝：
    註: 因為 pytorch 必須額外安裝，因此用 bat 檔來執行
    ```bash
    ./scripts/install_all.bat
    ```

## 快速開始

### 1. 設定

所有的實驗設定都集中在 `config/` 目錄下。

-   `config/base.yml`: 基礎設定檔，包含了資料路徑、模型超參數等。

在開始訓練或推理之前，請根據您的環境和需求修改對應的設定檔，特別是資料集的路徑。

### 2. 模型訓練

訓練過程由 `main.py` 啟動。您需要指定設定檔和訓練模式。

-   **基本訓練指令**
    ```bash
    python main.py --config config/base.yml --experiment train --model_name "模型命名"

訓練完成後，模型的權重檔案 (checkpoints) 會自動儲存在 `checkpoints/` 目錄下對應的實驗資料夾中。

### 3. 模型推理

使用訓練好的模型進行推理，同樣是透過 `main.py`。您需要指定設定檔、推理模式以及要載入的模型權重路徑。

-   **基本推理指令**
    ```bash
    python main.py --config config/base.yml --mode inference --checkpoint_path "checkpoints/path/to/your/model.pth"
    ```
-   請將 `--checkpoint_path` 後的路徑替換為您實際儲存的模型檔案路徑。
-   `scripts/inference_all.bat` 提供了一個批量推理的範例。

推理結果和日誌將會儲存在 `logs/` 目錄下。

### 4. 訓練與模型推理懶人包
    inference_all.bat 可以直接訓練接著推理
-   **基本訓練指令**
    ```bash
    set MODEL=模型名稱

    python main.py --experiment train --model_name "%MODEL%"

    set TASKS="資料集名字"
    ```

## 專案結構

```
AudioFusion-MoE-DeepDetect/
├───checkpoints/      # 儲存訓練好的模型權重
├───config/           # 存放所有設定檔
├───data/             # (推測) 存放資料處理相關腳本
├───logs/             # 儲存訓練和推理的日誌
├───scripts/          # 存放常用的指令稿 (如: 訓練、推理)
├───src/              # 專案主要原始碼
│   ├───models/       # 模型架構定義
│   └───...
├───main.py           # 程式進入點
└───requirements.txt  # 專案依賴套件
```

## 注意事項

-   在執行任何腳本前，請確保您已根據 `config/` 中的設定準備好您的資料集。
-   `scripts/` 中的 `.bat` 檔案是為 Windows 環境設計的，如果您在 Linux 或 macOS 上運行，請將其轉換為對應的 `.sh` 腳本。
