import os 
import torch
import torch.nn as nn
import torch.optim as optim
from config import init
from trainer import train, validation_phase, get_components
from models.queue.NegativeQueue import NegativeQueue
from load_datasets import load_datasets
def main(args):
    device = args.device

    # 定義每個模態的輸入維度
    modality_input_dims = {
        "RawNet3": 256,
        "Wav2Vec2": 1024,
        "SpeechSplit_timbre_content": 2,
        "SpeechSplit_pitch": 64,
        "SpeechSplit_rhythm": 2
    }

    # 載入資料集
    train_dataloader, validation_dataloader = load_datasets(args)
    # 載入模型組件
    encoders, wav2vec_extractor, connectors, experts, classifier = get_components(args=args, modality_input_dims=modality_input_dims)

    negative_queue = NegativeQueue(feature_dim=args.tdnn_hidden_dim, queue_size=args.queue_size)

    
    params_to_update = []
    if args.is_train_connectors:
        for conn in connectors.values():
            params_to_update += list(conn.parameters())

    if args.is_train_experts:
        for exp in experts.values():
            params_to_update += list(exp.parameters())

    if args.is_train_classifier:
        params_to_update += list(classifier.parameters())

    optimizer = optim.Adam(params_to_update, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

    criterion = nn.CrossEntropyLoss()

    # 準備儲存目錄
    connector_checkpoint_dir = os.path.join(args.model_name, "pretrained_models/connectors")
    os.makedirs(connector_checkpoint_dir, exist_ok=True)
    expert_checkpoint_dir = os.path.join(args.model_name, "pretrained_models/experts")
    os.makedirs(expert_checkpoint_dir, exist_ok=True)
    classifier_checkpoint_dir = os.path.join(args.model_name, "pretrained_models/classifier")
    os.makedirs(classifier_checkpoint_dir, exist_ok=True)
    best_eer = float('inf')

    # 訓練迴圈 (加上 alpha_align 用來控制對齊損失權重)
    for epoch in range(args.epochs):
        if args.is_train_connectors:
            alpha_align = 1.0
        else:
            alpha_align = .0
        if args.is_train_experts:
            alpha_contrast = 1.0
            alpha_length = 1.0
        else:
            alpha_contrast = .0
            alpha_length = .0

        train_loss = train(
            train_dataloader,
            encoders,
            wav2vec_extractor,
            connectors,
            experts,
            classifier,
            negative_queue=negative_queue,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            is_train_connector=args.is_train_connectors,
            is_train_expert=args.is_train_experts,
            is_train_classifier=args.is_train_classifier,
            alpha_align=alpha_align,
            alpha_contrast=alpha_contrast,
            alpha_length=alpha_length
        )
        print(f"Epoch {epoch + 1}/{args.epochs} - Loss: {train_loss:.4f}")

        # 在validation_set上做驗證
        val_loss, val_eer = validation_phase(
            validation_dataloader,
            encoders,
            wav2vec_extractor,
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
        #         os.path.join(expert_checkpoint_dir, f"{modality}_connector_epoch_{epoch}.pth")
        #     )
        # for modality, expert in experts.items():
        #     torch.save(
        #         expert.state_dict(),
        #         os.path.join(expert_checkpoint_dir, f"{modality}_expert_epoch_{epoch}.pth")
        #     )
        # torch.save(
        #     classifier.state_dict(),
        #     os.path.join(classifier_checkpoint_dir, f"classifier_epoch_{epoch}.pth")
        # )

        # 檢查是否是最佳 EER
        if val_eer < best_eer:
            best_eer = val_eer
            if args.is_train_connectors:
                for modality, connector in connectors.items():
                    torch.save(
                        connector.state_dict(),
                        os.path.join(connector_checkpoint_dir, f"best_{modality}_connector.pth")
                    )
            
            if args.is_train_experts:
                for modality, expert in experts.items():
                    torch.save(
                        expert.state_dict(),
                        os.path.join(expert_checkpoint_dir, f"best_{modality}_expert.pth")
                    )
            
            if args.is_train_classifier:
                torch.save(
                    classifier.state_dict(),
                    os.path.join(classifier_checkpoint_dir, f"best_classifier.pth")
                )

    print("Training complete.")


if __name__ == '__main__':
    args = init()
    main(args)
