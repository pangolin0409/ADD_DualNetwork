from data.dataloader import RawAudio
from RawNet2.model_RawNet2 import RawNet2
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import os
from tqdm import tqdm
import argparse
import utils.eval_metrics as em
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from train_router import QFormerConnector, SparseMoE, ConnectorClassifier

def init():
    parser = argparse.ArgumentParser(description="Inference with Sparse MoE for Audio Deepfake Detection")

    # 模型參數
    parser.add_argument("--input_dim", type=int, default=1024, help="Input feature dimension from RawNet2/Wav2Vec2")
    parser.add_argument("--query_dim", type=int, default=512, help="Projected feature dimension")
    parser.add_argument("--num_queries", type=int, default=16, help="Number of learnable queries")
    parser.add_argument("--num_heads", type=int, default=4, help="Number of attention heads in Transformer")
    parser.add_argument("--num_layers", type=int, default=1, help="Number of Transformer layers")

    # Router 和 Experts 參數
    parser.add_argument("--num_experts", type=int, default=4, help="Number of experts")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension in each expert")
    parser.add_argument("--moe_output_dim", type=int, default=256, help="Output dimension of MoE")
    
    parser.add_argument('--model_folder', type=str, help="directory for pretrained model",
                        default='./models/sparse_moe/')
    parser.add_argument('-n', '--model_name', type=str, help="the name of the model",
                        required=False, default='sparse_moe_model')
    parser.add_argument("-t", "--task", type=str, help="which dataset you would like to score on",
                        required=False, default='19eval')
    parser.add_argument('-nb_samp', type = int, default = 64600)
    parser.add_argument('-nb_worker', type=int, default=8)
    parser.add_argument("--gpu", type=str, help="GPU index", default="0")
    parser.add_argument("--k", type=int, default=2, help="Top-K experts to select in Sparse MoE")
    
    #DNN args
    parser.add_argument('-m_first_conv', type = int, default = 251)
    parser.add_argument('-m_in_channels', type = int, default = 1)
    parser.add_argument('-m_filts', type = list, default = [128, [128,128], [128,256], [256,256]])
    parser.add_argument('-m_blocks', type = list, default = [2, 4])
    parser.add_argument('-m_nb_fc_att_node', type = list, default = [1])
    parser.add_argument('-m_nb_fc_node', type = int, default = 1024)
    parser.add_argument('-m_gru_node', type = int, default = 1024)
    parser.add_argument('-m_nb_gru_layer', type = int, default = 1)
    parser.add_argument('-m_nb_samp', type = int, default = 64600)
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    args.model = {}
    for k, v in vars(args).items():
        if k[:2] == 'm_':
            print(k, v)
            args.model[k[2:]] = v
    args.model['nb_classes'] = 2        
    return args

def load_model(args):
    # 初始化模型架構
    rawnet2 = RawNet2(args.model).to(args.device)
    
    wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-xls-r-300m").to(args.device)
    connector_rawnet = QFormerConnector(
        input_dim=args.input_dim, query_dim=args.query_dim, num_queries=args.num_queries,
        num_heads=args.num_heads, num_layers=args.num_layers
    ).to(args.device)
    connector_wav2vec = QFormerConnector(
        input_dim=args.input_dim, query_dim=args.query_dim, num_queries=args.num_queries,
        num_heads=args.num_heads, num_layers=args.num_layers
    ).to(args.device)
    moe = SparseMoE(input_dim=args.query_dim * 2, num_experts=args.num_experts, hidden_dim=args.hidden_dim, 
                output_dim=args.moe_output_dim, k=2).to(args.device)
    classifier = ConnectorClassifier(input_dim=args.moe_output_dim, num_classes=2).to(args.device)


    # 加載檢查點
    model_path = os.path.join('./checkpoints/best_checkpoint.pth')
    checkpoint = torch.load(model_path)
    rawnet2.load_state_dict(checkpoint["rawnet2"])
    connector_rawnet.load_state_dict(checkpoint["connector_rawnet"])
    connector_wav2vec.load_state_dict(checkpoint["connector_wav2vec"])
    model_path = os.path.join('./checkpoints/best_moe_checkpoint.pth')
    checkpoint = torch.load(model_path)
    moe.load_state_dict(checkpoint["moe"])
    classifier.load_state_dict(checkpoint["classifier"])

    return rawnet2, wav2vec_model, connector_rawnet, connector_wav2vec, moe, classifier

def test_on_desginated_datasets(task, models, device):
    rawnet2, wav2vec_model, connector_rawnet, connector_wav2vec, moe, classifier = models

    # 加載數據集
    test_set = RawAudio(
        path_to_database=f'../datasets/{task}',
        meta_csv='meta.csv',
        return_label=True,
        nb_samp=64600,
        part='test'
    )
    testDataLoader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8)

    # 用於保存結果
    score_loader = []
    label_loader = []

    # Wav2Vec2 處理器
    processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-xls-r-300m")

    for i, data_slice in enumerate(tqdm(testDataLoader)):
        waveforms, labels = data_slice
        waveforms = waveforms.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            # RawNet2 特徵提取
            rawnet_features = rawnet2(waveforms, is_test=True)

            # Wav2Vec2 特徵提取
            input_values = processor(waveforms, sampling_rate=16000, return_tensors="pt").input_values.to(device)
            input_values = input_values.squeeze(0)
            wav2vec_features = wav2vec_model(input_values).last_hidden_state.mean(dim=1)

            # Connector 處理
            rawnet_proj = connector_rawnet(rawnet_features)
            wav2vec_proj = connector_wav2vec(wav2vec_features)

            # 特徵拼接
            combined_features = torch.cat([rawnet_proj.mean(dim=1), wav2vec_proj.mean(dim=1)], dim=1)

            # MoE 處理
            moe_output = moe(combined_features)

            # 分類器輸出
            model_outputs = classifier(moe_output)
            score = F.softmax(model_outputs, dim=1)[:, 0]

            # 保存分數和標籤
            score_loader.append(score.item())
            label_loader.append(labels.item())

    scores = torch.tensor(score_loader).numpy()
    labels = torch.tensor(label_loader).numpy()

    # 計算 EER 和其他指標
    target_scores = scores[labels == 0]  # 正例 (bonafide)
    nontarget_scores = scores[labels == 1]  # 負例 (spoof)

    eer, frr, far, threshold = em.compute_eer(target_scores, nontarget_scores)
    print(f'Equal Error Rate (EER): {eer}, False Rejection Rate (FFR): {frr}, False Acceptance Rate (FAR): {far}, Threshold: {threshold}')


if __name__ == "__main__":
    args = init()
    models = load_model(args)
    test_on_desginated_datasets(args.task, models, args.device)
