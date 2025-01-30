import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset, Subset
from data.dataloader import RawAudio
from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
from models.rawnet3.RawNet3 import RawNet3
from models.rawnet3.RawNetBasicBlock import Bottle2neck
from models.speechsplit.model import Generator_3
from models.speechsplit.hparams import hparams
from models.experts.ExpertMLP import ExpertMLP, Simple_Expert
from models.connectors.Qformer import QFormerConnector, MLPConnector, ConvPoolingConnector
from models.classifier.AudioClassifier import HiddenStateLLMClassifier, CustomTokenLLMClassifier
from utils.Projection import Preprocessor
from config import init
from trainer import train_phase_1, validation_phase, downsample_data

def main(args):
    print(args.is_train_connectors)
    print(args.is_train_experts)
    print(args.is_train_classifier)
    device = args.device

    # 定義每個模態的輸入維度
    modality_input_dims = {
        "RawNet3": 256,
        "Wav2Vec2": 1024,
        "SpeechSplit_timbre_content": 2,
        "SpeechSplit_pitch": 64,
        "SpeechSplit_rhythm": 2
    }

    # 定義目標維度（連接器的輸入維度）
    target_query_dim = args.query_dim

    # 初始化 datasets 和 dataloaders
    training_sets = {
        "DFADD": RawAudio(
            path_to_database='../datasets/DFADD',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='train'
        ),
        "CodecFake": RawAudio(
            path_to_database='../datasets/CodecFake',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='train'
        ),
        "ASVspoof2021_DF": RawAudio(
            path_to_database='../datasets/ASVspoof2021_DF',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='train'
        )
    }

    training_set_list = []
    for name, training_set in training_sets.items():
        real_indices, spoof_indices = downsample_data(meta_path=f'../datasets/{name}/train/meta.csv', dataset_name=name, target_fake_ratio=2)
        real_subset = Subset(training_set, real_indices)
        spoof_subset = Subset(training_set, spoof_indices)
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        training_set_list.append(adjusted_set)

    final_training_set = ConcatDataset(training_set_list)
    train_dataloader = DataLoader(final_training_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.nb_worker)

    # 初始化 datasets 和 dataloaders
    validation_sets = {
        "DFADD": RawAudio(
            path_to_database='../datasets/DFADD',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='validation'
        ),
        "CodecFake": RawAudio(
            path_to_database='../datasets/CodecFake',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='validation'
        ),
        "ASVspoof2021_DF": RawAudio(
            path_to_database='../datasets/ASVspoof2021_DF',
            meta_csv='meta.csv',
            return_label=True,
            nb_samp=args.nb_samp,
            part='validation'
        )
    }

    validation_set_list = []
    for name, validation_set in validation_sets.items():
        real_indices, spoof_indices = downsample_data(meta_path=f'../datasets/{name}/validation/meta.csv', dataset_name=name, target_fake_ratio=2)
        real_subset = Subset(validation_set, real_indices)
        spoof_subset = Subset(validation_set, spoof_indices)
        adjusted_set = ConcatDataset([real_subset, spoof_subset])
        validation_set_list.append(adjusted_set)

    final_validation_set = ConcatDataset(validation_set_list)
    validation_dataloader = DataLoader(final_validation_set, batch_size=args.batch_size, shuffle=True, drop_last=False, num_workers=args.nb_worker)

    # Load encoders
    encoders = {
        "RawNet3": RawNet3(
            Bottle2neck,
            model_scale=8,
            context=True,
            summed=True,
            encoder_type="ECA",
            nOut=256,
            out_bn=False,
            sinc_stride=10,
            log_sinc=True,
            norm_sinc="mean",
            grad_mult=1.0,
        ).to(device),
        "Wav2Vec2": Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-xls-r-300m",
            cache_dir=args.wav2vec_path
        ).to(device),
        "SpeechSplit_timbre_content": Generator_3(hparams).to(device),
        "SpeechSplit_pitch": Generator_3(hparams).to(device),
        "SpeechSplit_rhythm": Generator_3(hparams).to(device),
    }

    # Load feature extractor
    wav2vec_extractor = Wav2Vec2FeatureExtractor.from_pretrained(
        "facebook/wav2vec2-xls-r-300m", cache_dir=args.wav2vec_path
    )
    
    # Load RawNet3 weights
    encoders['RawNet3'].load_state_dict(
        torch.load(
            args.rawnet3_path,
            map_location=device,
            weights_only=True
        )["model"]
    )

    # Load SpeechSplit weights
    checkpoint = torch.load(args.speechsplit_path, map_location=device, weights_only=True)
    encoders['SpeechSplit_timbre_content'].load_state_dict(checkpoint['model'])
    encoders['SpeechSplit_pitch'].load_state_dict(checkpoint['model'])
    encoders['SpeechSplit_rhythm'].load_state_dict(checkpoint['model'])

    for encoder in encoders.values():
        encoder.requires_grad_(False)  # Freeze all encoders by default

    # 初始化預處理層
    preprocessors = {
        modality: Preprocessor(modality, modality_input_dims[modality], target_query_dim)
        for modality in encoders.keys()
    }
    preprocessors = {k: v.to(device) for k, v in preprocessors.items()}

    # 初始化連接器
    connectors = {}
    for modality in encoders.keys():
        input_dim = modality_input_dims[modality]
        connector = MLPConnector(input_dim=input_dim, output_dim=target_query_dim).to(device)
        # 如果不是訓練連接器，則載入預訓練連接器
        if not args.is_train_connectors:
            checkpoint_path = f"./{args.model_name}/pretrained_models/connectors/best_{modality}_connector.pth"
            connector.load_state_dict(torch.load(checkpoint_path))
        connectors[modality] = connector

    # 初始化專家
    experts = {
        modality: ExpertMLP(
            model_name=args.llm_model_name,
            model_dir=args.llm_model_dir
        ).to(device)
        for modality in encoders.keys()
    }

    # experts = {
    #     modality: Simple_Expert(
    #         hidden_dim=512,
    #         output_dim=768
    #     ).to(device)
    #     for modality in encoders.keys()
    # }

    # 初始化單一分類器
    classifier = HiddenStateLLMClassifier(
        llm_name=args.llm_model_name,
        model_dir=args.llm_model_dir
    ).to(device)

    # 僅訓練連接器
    optimizer = optim.Adam(
        [param for connector in connectors.values() for param in connector.parameters()],
        lr=args.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    # 準備儲存目錄
    checkpoint_dir = os.path.join(args.model_name, "pretrained_models/connectors")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_eer = float('inf')

    # 訓練迴圈 (加上 alpha_align 用來控制對齊損失權重)
    for epoch in range(args.epochs):
        train_loss = train_phase_1(
            train_dataloader,
            encoders,
            wav2vec_extractor,
            preprocessors,
            connectors,
            experts,
            classifier,
            optimizer,
            criterion,
            device,
            alpha_align=args.alpha,
            is_train_connector=args.is_train_connectors,
            is_train_experts=args.is_train_experts,
            is_train_classifier=args.is_train_classifier
        )
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {train_loss:.4f}")

        # 在validation_set上做驗證
        val_loss, val_eer = validation_phase(
            validation_dataloader,
            encoders,
            wav2vec_extractor,
            preprocessors,
            connectors,
            experts,
            classifier,
            criterion,
            device
        )
        print(f"Epoch {epoch+1} - Validation Loss: {val_loss:.4f}, Validation EER: {val_eer:.4f}")

        # 每個 epoch 寫一次紀錄
        with open("training_log.csv", "a") as f:
            f.write(f"{epoch},{train_loss:.4f},{val_loss:.4f}, {val_eer:.4f}\n")

        # 更新學習率
        scheduler.step()
        print(f"Epoch {epoch+1}: Learning Rate = {optimizer.param_groups[0]['lr']:.6f}")

        # 保存檢查點
        # for modality, connector in connectors.items():
        #     torch.save(
        #         connector.state_dict(),
        #         os.path.join(checkpoint_dir, f"{modality}_connector_epoch_{epoch}.pth")
        #     )

        # 檢查是否是最佳 EER
        if val_eer < best_eer:
            best_eer = val_eer
            if args.is_train_connectors:
                for modality, connector in connectors.items():
                    torch.save(
                        connector.state_dict(),
                        os.path.join(checkpoint_dir, f"best_{modality}_connector.pth")
                    )
            if args.is_train_experts:
                for modality, expert in experts.items():
                    torch.save(
                        expert.state_dict(),
                        os.path.join(checkpoint_dir, f"best_{modality}_expert.pth")
                    )

    print("Training complete.")


if __name__ == '__main__':
    args = init()
    main(args)
