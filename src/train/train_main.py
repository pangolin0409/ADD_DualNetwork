import os
import gc
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR, StepLR, SequentialLR
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
import mlflow
from src.models.Detector import Detector
from src.data.load_datasets import load_datasets
from src.utils.eval_metrics import compute_eer, calculate_metrics
from src.utils.common_utils import get_git_branch, send_discord
from src.utils.mlflow_utils import MLflowManager, log_training_artifacts, create_model_signature

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class DiscordCallback(pl.Callback):
    def __init__(self, webhook: str, model_name: str):
        super().__init__()
        self.webhook = webhook
        self.model_name = model_name

    def on_fit_start(self, trainer, pl_module):
        if self.webhook:
            send_discord(f"🔄 開始訓練（Lightning）：{self.model_name}", self.webhook)

    def on_fit_end(self, trainer, pl_module):
        if self.webhook:
            send_discord(f"✅ 訓練完成（Lightning）：{self.model_name}", self.webhook)

class LightningDetector(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.model = Detector(
            ssl_model_name=args.ssl_model_name,
            encoder_dim=args.encoder_dim,
            num_experts=args.num_experts,
            num_classes=args.num_classes,
            max_temp=args.max_temp,
            min_temp=args.min_temp,
            start_alpha=args.start_alpha,
            end_alpha=args.end_alpha,
            warmup_epochs=args.warmup_epochs,
            is_training=True
        )
        self.ce_loss = nn.CrossEntropyLoss()
        self.validation_step_outputs = []

        # 若不是微調則凍結 SSL backbone
        if not getattr(args, 'is_finetune', False):
            for param in self.model.ssl_model.model.parameters():
                param.requires_grad = False

    def forward(self, wave, epoch=None):
        if epoch is None:
            epoch = self.current_epoch
        return self.model(wave=wave, epoch=epoch)

    def compute_total_loss(self, logits, routing, labels):
        loss_ce = self.ce_loss(logits, labels)
        loss_limp = self.model.compute_limp_loss(routing)
        loss_load = self.model.compute_load_balance_loss(routing)
        total_loss = (
            self.args.lambda_ce * loss_ce +
            self.args.lambda_limp * loss_limp +
            self.args.lambda_load * loss_load
        )
        return total_loss, loss_ce, loss_limp, loss_load

    def training_step(self, batch, batch_idx):
        wave, labels, _ = batch
        labels = labels.to(self.device)
        wave = wave.to(self.device)
        logits, routing, _, _ = self.forward(wave)
        total_loss, loss_ce, loss_limp, loss_load = self.compute_total_loss(logits, routing, labels)

        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("loss/total", total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss/ce", loss_ce, on_step=True, on_epoch=True)
        self.log("loss/loss_limp", loss_limp, on_step=True, on_epoch=True)
        self.log("loss/loss_load", loss_load, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)
        return total_loss

    def validation_step(self, batch, batch_idx):
        wave, labels, _ = batch
        labels = labels.to(self.device)
        wave = wave.to(self.device)
        logits, routing, _, _ = self.forward(wave)
        total_loss, loss_ce, loss_limp, loss_load = self.compute_total_loss(logits, routing, labels)
        scores = F.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        acc = (preds == labels).float().mean()

        self.log("val_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)

        self.validation_step_outputs.append({
            'scores': scores.detach().cpu(),
            'labels': labels.detach().cpu()
        })
        return total_loss

    def on_validation_epoch_end(self):
        if not self.validation_step_outputs:
            return
        scores = torch.cat([o['scores'] for o in self.validation_step_outputs], dim=0).numpy()
        labels = torch.cat([o['labels'] for o in self.validation_step_outputs], dim=0).numpy()
        self.validation_step_outputs.clear()

        eer, frr, far, threshold = compute_eer(scores[labels == 1], scores[labels == 0])
        precision, recall, f1, cm = calculate_metrics(scores, labels, threshold)

        # 記錄主要 metrics（EER 越小越好）
        self.log("val_eer", torch.tensor(eer), prog_bar=True)
        self.log("val_precision", torch.tensor(precision))
        self.log("val_recall", torch.tensor(recall))
        self.log("val_f1", torch.tensor(f1))

        # 額外紀錄到 MLflow（如果有活動的 run）
        try:
            mlflow.log_metrics({
                "EER": float(eer),
                "FRR": float(frr),
                "FAR": float(far),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }, step=self.current_epoch)
        except Exception:
            pass

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)

        def linear_warmup(epoch):
            return min(0.1 + 0.9 * (epoch / self.args.warmup_epochs), 1.0)

        scheduler_warmup = LambdaLR(optimizer, linear_warmup)
        scheduler_step = StepLR(optimizer, step_size=5, gamma=0.7)
        scheduler = SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_step], milestones=[self.args.warmup_epochs])

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'epoch',
                'frequency': 1,
                'monitor': 'val_eer',
            }
        }

    # DataLoaders
    def train_dataloader(self):
        return load_datasets(
            sample_rate=self.args.nb_samp,
            batch_size=self.args.batch_size,
            dataset_names=self.args.train_datasets,
            worker_size=self.args.nb_worker,
            target_fake_ratio=1,
            part='train',
            is_downsample=False,
            args=self.args
        )

    def val_dataloader(self):
        return load_datasets(
            sample_rate=self.args.nb_samp,
            batch_size=self.args.batch_size,
            dataset_names=self.args.valid_datasets,
            worker_size=self.args.nb_worker,
            target_fake_ratio=6,
            part='test',
            is_downsample=False
        )

def train_model(args):
    set_seed(args.seed)

    # 準備 MLflow
    mlflow_manager = MLflowManager()
    run_name = f"{get_git_branch()}_{args.model_name}"
    with mlflow_manager.start_run(run_name=run_name):
        mlflow.pytorch.autolog()

        # 紀錄訓練參數
        mlflow_manager.log_params({
            'model_name': args.model_name,
            'ssl_model_name': args.ssl_model_name,
            'encoder_dim': args.encoder_dim,
            'num_experts': args.num_experts,
            'num_classes': args.num_classes,
            'lr': args.lr,
            'batch_size': args.batch_size,
            'num_epochs': args.num_epochs,
            'weight_decay': args.weight_decay,
            'patience': args.patience,
            'warmup_epochs': args.warmup_epochs,
            'max_temp': args.max_temp,
            'min_temp': args.min_temp,
            'start_alpha': args.start_alpha,
            'end_alpha': args.end_alpha,
            'lambda_ce': args.lambda_ce,
            'lambda_limp': args.lambda_limp,
            'lambda_load': args.lambda_load,
            'is_finetune': args.is_finetune,
            'seed': args.seed,
            'nb_samp': args.nb_samp,
            'train_datasets': str(args.train_datasets),
            'valid_datasets': str(args.valid_datasets),
            'aug_group': args.aug_group,
            'aug_prob': args.aug_prob
        })

        # 準備路徑
        os.makedirs(os.path.join(args.save_path, args.model_name), exist_ok=True)
        os.makedirs(os.path.join(args.log_path, args.model_name), exist_ok=True)

        # Lightning 模組
        lit_module = LightningDetector(args)

        # Callbacks
        checkpoint_cb = ModelCheckpoint(
            dirpath=os.path.join(args.save_path, args.model_name),
            filename='best-model-{epoch:02d}-{val_eer:.4f}',
            monitor='val_eer',
            mode='min',
            save_top_k=1,
            save_last=True
        )
        earlystop_cb = EarlyStopping(monitor='val_eer', mode='min', patience=args.patience)
        discord_cb = DiscordCallback(webhook=args.discord_webhook, model_name=args.model_name)

        # Logger（可用 CSVLogger；MLflow 由 autolog 負責）
        csv_logger = CSVLogger(save_dir=os.path.join(args.log_path, args.model_name), name='pl_logs')

        # Trainer
        accelerator = 'gpu' if torch.cuda.is_available() else 'cpu'
        devices = 1
        trainer = pl.Trainer(
            max_epochs=args.num_epochs,
            accelerator=accelerator,
            devices=devices,
            callbacks=[checkpoint_cb, earlystop_cb, discord_cb],
            logger=csv_logger,
            log_every_n_steps=50
        )

        trainer.fit(lit_module)

        # 記錄訓練相關文件
        log_training_artifacts(
            mlflow_manager,
            args,
            model_path=os.path.join(args.save_path, args.model_name),
            log_path=os.path.join(args.log_path, args.model_name)
        )

def main(args):
    train_model(args)

if __name__ == "__main__":
    from config.config import init
    args = init()
    main(args)
