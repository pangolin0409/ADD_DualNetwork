import argparse
import torch

def init():
    parser = argparse.ArgumentParser(description="Train Q-Former Connector for Audio Deepfake Detection")
    # 模型參數
    parser.add_argument("--input_dim", type=int, default=1024, help="Input feature dimension from RawNet2/Wav2Vec2")
    parser.add_argument("--query_dim", type=int, default=768, help="Projected feature dimension")
    parser.add_argument("--tdnn_hidden_dim", type=int, default=256, help="Projected feature dimension")
    parser.add_argument("--num_queries", type=int, default=4, help="Number of learnable queries")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--num_layers", type=int, default=4, help="Number of Transformer layers")
    parser.add_argument('--queue_size', type=int, default=6144, help="Queue size for negative samples")
    
    # 訓練參數
    parser.add_argument('-model_name', type=str, default='DAC')
    parser.add_argument('-nb_samp', type=int, default=64600)
    parser.add_argument('-alpha', type=float, default=.8)
    parser.add_argument('-beta', type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate for optimizer")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument('-nb_worker', type=int, default=12)
    parser.add_argument('-is_train_connectors', action='store_true', default=False)
    parser.add_argument('-is_train_experts', action='store_true', default=False)
    parser.add_argument('-is_train_classifier', action='store_true', default=False)

    # 預訓練模型路徑
    parser.add_argument("--rawnet3_path", type=str, default='./pretrained_models/rawnet3/rawnet3_weights.pt', help="Path to the RawNet3 model")
    parser.add_argument("--speechsplit_path", type=str, default='./pretrained_models/speechsplit/speechsplit_weights.ckpt', help="Path to the SpeechSplit model")
    parser.add_argument("--wav2vec_path", type=str, default='./pretrained_models/wav2vec2-xls-r-300m', help="Path to the wav2vec model")
    parser.add_argument("--llm_model_name", type=str, default="gpt2", help="LLM model name for ExpertMLP and Classifier")
    parser.add_argument("--llm_model_dir", type=str, default="./pretrained_models/gpt2_local", help="Path to the LLM model directory")
    
    # 新增資料集名稱參數
    parser.add_argument("--datasets", nargs='+', type=str, default=['DFADD', 'CodecFake', 'ASVspoof2021_DF'], help="List of dataset names, e.g., DFADD, CodecFake, ASVspoof2021_DF")
    
    args = parser.parse_args()
    return args