import argparse
import torch

def init():
    parser = argparse.ArgumentParser(description="Train Q-Former Connector for Audio Deepfake Detection")
    # 模型參數
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--encoder_dim', type=int, default=1024, help="Dimension of the encoder output")
    parser.add_argument('--routing_dim', type=int, default=4, help="Dimension of the routing network")
    parser.add_argument('--top_k', type=int, default=2)
    parser.add_argument('--expert_dim', type=int, default=128, help="Dimension of the routing network")
    parser.add_argument('--num_classes', type=int, default=2, help="Number of classes for classification")
    # 訓練參數
    parser.add_argument('-model_name', type=str, default='WA_ASV019')
    parser.add_argument('-nb_samp', type=int, default=64600)
    parser.add_argument('--weight_decay', type=float, default=0.0001, help="Weight decay for optimizer")
    parser.add_argument("--batch_size", type=int, default=48, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--num_epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument('-nb_worker', type=int, default=16)
    parser.add_argument("--aasist_config_path", type=str, default='./config/AASIST.conf', help="Path to the AASIST model config")
    parser.add_argument("--save_path", type=str, default='./checkpoints', help="Path to save the model checkpoints")
    
    parser.add_argument('--lambda_ce', type=float, default=1.0, help="Weight for cross entropy loss")
    parser.add_argument('--lambda_contrastive', type=float, default=0.5, help="Weight for contrastive loss")
    parser.add_argument('--lambda_consistency', type=float, default=0.1, help="Weight for consistency loss")
    parser.add_argument('--lambda_moe', type=float, default=0.3, help="Weight for mixture of experts loss")
    parser.add_argument('--lambda_unknown', type=float, default=0.1, help="Weight for unknown sample detection loss")
    parser.add_argument('--lambda_entropy', type=float, default=0.1, help="Weight for entropy loss")
    parser.add_argument('--router_aug_prob', type=float, default=0.7, help="Probability of router augmentation")
    parser.add_argument('--patience', type=int, default=5, help="Number of epochs to wait before early stopping")
    parser.add_argument('--unknown_cluster_interval', type=int, default=2, help="Interval for updating unknown sample clusters")
    parser.add_argument('--warmup_epochs', type=int, default=5, help="Number of epochs for warmup")
    parser.add_argument('--consistency_warmup', type=int, default=4, help="Number of epochs for consistency warmup")
    # 預訓練模型路徑
    parser.add_argument("--rawnet3_path", type=str, default='./pretrained_models/rawnet3/rawnet3_weights.pt', help="Path to the RawNet3 model")
    parser.add_argument("--speechsplit_path", type=str, default='./pretrained_models/speechsplit/speechsplit_weights.ckpt', help="Path to the SpeechSplit model")
    parser.add_argument("--wav2vec_path", type=str, default='./pretrained_models/wav2vec2-xls-r-300m', help="Path to the wav2vec model")
    
    # 新增資料集名稱參數
    parser.add_argument("--datasets", nargs='+', type=str, default=['Asvspoof2019_LA'], help="List of dataset names, e.g., DFADD, CodecFake, ASVspoof2021_DF")
    # parser.add_argument("--datasets", nargs='+', type=str, default=['DFADD', 'ASVspoof2021_DF', 'CodecFake', 'Asvspoof2019_LA'], help="List of dataset names, e.g., DFADD, CodecFake, ASVspoof2021_DF")
    
    args = parser.parse_args()
    return args