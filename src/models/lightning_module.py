"""
PyTorch Lightning Module for Person Re-Identification
人员重识别 PyTorch Lightning 模块

将模型、损失函数、优化器等封装到 Lightning 模块中，
简化训练、验证和测试流程
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Dict, Any, Optional, Tuple
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

from .siamese_cnn import SiameseCNN


class ReIDLightningModule(pl.LightningModule):
    """
    Person Re-ID Lightning 模块

    功能:
    - 自动处理训练/验证/测试循环
    - 损失函数和指标计算
    - 优化器和学习率调度
    - TensorBoard/WandB 日志记录
    """

    def __init__(
        self,
        model: nn.Module,
        loss_type: str = "contrastive",
        loss_params: Optional[Dict[str, Any]] = None,
        optimizer_name: str = "sgd",
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        weight_decay: float = 0.00025,
        scheduler_name: str = "polynomial",
        scheduler_params: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化 Lightning 模块

        Args:
            model: PyTorch 模型
            loss_type: 损失函数类型 ('cross_entropy', 'contrastive', 'triplet')
            loss_params: 损失函数参数
            optimizer_name: 优化器名称 ('sgd', 'adam', 'adamw')
            learning_rate: 学习率
            momentum: SGD 动量
            weight_decay: L2 正则化
            scheduler_name: 学习率调度器 ('polynomial', 'step', 'cosine')
            scheduler_params: 调度器参数
        """
        super().__init__()

        self.model = model
        self.loss_type = loss_type
        self.loss_params = loss_params or {}
        self.optimizer_name = optimizer_name
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.scheduler_name = scheduler_name
        self.scheduler_params = scheduler_params or {}

        # 保存超参数
        self.save_hyperparameters(ignore=['model'])

        # 损失函数
        self.loss_fn = self._create_loss_function()

    def _create_loss_function(self) -> nn.Module:
        """创建损失函数"""
        if self.loss_type == "cross_entropy":
            return nn.CrossEntropyLoss(
                label_smoothing=self.loss_params.get("label_smoothing", 0.0)
            )

        elif self.loss_type == "contrastive":
            margin = self.loss_params.get("margin", 2.0)
            return ContrastiveLoss(margin=margin)

        elif self.loss_type == "triplet":
            margin = self.loss_params.get("margin", 0.3)
            return nn.TripletMarginLoss(margin=margin)

        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        前向传播

        Args:
            x1, x2: (B, C, H, W) 图像对

        Returns:
            output: 模型输出
        """
        return self.model(x1, x2)

    def training_step(self, batch, batch_idx):
        """训练步骤"""
        (x1, x2), labels = batch

        # 前向传播
        if self.loss_type == "cross_entropy":
            # 输出 logits: (B, 2)
            outputs = self(x1, x2)
            loss = self.loss_fn(outputs, labels)

            # 计算准确率
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean()

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        elif self.loss_type == "contrastive":
            # Contrastive loss 需要 embeddings
            emb1 = self.model.get_embedding(x1)
            emb2 = self.model.get_embedding(x2)
            # 修复: Dataset 返回 label=1 (same), 0 (different)
            # 但 ContrastiveLoss 期望 label=0 (same), 1 (different)
            # 需要反转标签: 1 - labels
            loss = self.loss_fn(emb1, emb2, 1 - labels.float())

            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        else:
            outputs = self(x1, x2)
            loss = self.loss_fn(outputs, labels)
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        (x1, x2), labels = batch

        if self.loss_type == "cross_entropy":
            outputs = self(x1, x2)
            loss = self.loss_fn(outputs, labels)
            preds = torch.argmax(outputs, dim=1)
            acc = (preds == labels).float().mean()

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        elif self.loss_type == "contrastive":
            emb1 = self.model.get_embedding(x1)
            emb2 = self.model.get_embedding(x2)
            # 修复: 反转标签以匹配 ContrastiveLoss 的约定
            loss = self.loss_fn(emb1, emb2, 1 - labels.float())

            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def test_step(self, batch, batch_idx):
        """测试步骤"""
        return self.validation_step(batch, batch_idx)

    def configure_optimizers(self):
        """配置优化器和学习率调度器"""
        # 创建优化器
        if self.optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.learning_rate,
                momentum=self.momentum,
                weight_decay=self.weight_decay,
                nesterov=True,
            )
        elif self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_name}")

        # 创建学习率调度器
        if self.scheduler_name == "polynomial":
            power = self.scheduler_params.get("power", 0.75)

            # 修复: 检查 datamodule 是否存在且不为 None
            # 当使用 Trainer.fit(model, train_dataloaders=...) 时，datamodule 存在但为 None
            if (hasattr(self.trainer, 'datamodule') and
                self.trainer.datamodule is not None and
                hasattr(self.trainer.datamodule, 'train_dataloader')):
                try:
                    num_batches = len(self.trainer.datamodule.train_dataloader())
                    max_steps = self.trainer.max_epochs * num_batches
                except (TypeError, AttributeError):
                    max_steps = 10000
            else:
                # 回退到默认值
                max_steps = 10000

            scheduler = PolynomialLR(
                optimizer,
                max_steps=max_steps,
                power=power,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        elif self.scheduler_name == "step":
            step_size = self.scheduler_params.get("step_size", 500)
            gamma = self.scheduler_params.get("gamma", 0.1)

            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=step_size,
                gamma=gamma,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        elif self.scheduler_name == "cosine":
            T_max = self.scheduler_params.get("T_max", 2000)
            eta_min = self.scheduler_params.get("eta_min", 1e-5)

            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=T_max,
                eta_min=eta_min,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        else:
            return optimizer


class ContrastiveLoss(nn.Module):
    """
    Contrastive Loss for Siamese Networks
    孪生网络对比损失

    L = (1-Y) * 0.5 * D^2 + Y * 0.5 * max(margin - D, 0)^2

    Where:
        Y = 0 for similar pairs, 1 for dissimilar pairs
        D = Euclidean distance between embeddings
    """

    def __init__(self, margin: float = 2.0):
        super().__init__()
        self.margin = margin

    def forward(
        self, emb1: torch.Tensor, emb2: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            emb1, emb2: (B, D) embeddings
            labels: (B,) 0=same person, 1=different person

        Returns:
            loss: scalar
        """
        # Euclidean distance
        distances = F.pairwise_distance(emb1, emb2, p=2)

        # Contrastive loss
        # Same person (label=0): minimize distance
        # Different person (label=1): maximize distance (up to margin)
        loss_same = (1 - labels) * torch.pow(distances, 2)
        loss_diff = labels * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)

        loss = 0.5 * (loss_same + loss_diff).mean()

        return loss


class PolynomialLR(LRScheduler):
    """
    Polynomial Learning Rate Scheduler
    多项式学习率调度器

    lr = initial_lr * (1 + gamma * step)^(-power)
    """

    def __init__(
        self,
        optimizer: Optimizer,
        max_steps: int,
        power: float = 0.75,
        last_epoch: int = -1,
    ):
        self.max_steps = max_steps
        self.power = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch == 0:
            return [base_lr for base_lr in self.base_lrs]

        gamma = self.last_epoch / self.max_steps
        return [base_lr * ((1 + 0.0001 * self.last_epoch) ** (-self.power)) for base_lr in self.base_lrs]


if __name__ == "__main__":
    # 测试 Lightning 模块
    print("Testing ReIDLightningModule...")

    # 创建模型
    from .siamese_cnn import create_siamese_cnn

    model = create_siamese_cnn(input_size=(160, 60))

    # 创建 Lightning 模块
    lightning_module = ReIDLightningModule(
        model=model,
        loss_type="cross_entropy",
        learning_rate=0.01,
    )

    print(f"\nLightning Module: {lightning_module.__class__.__name__}")
    print(f"Loss type: {lightning_module.loss_type}")
    print(f"Optimizer: {lightning_module.optimizer_name}")

    # 测试前向传播
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 160, 60)
    x2 = torch.randn(batch_size, 3, 160, 60)
    labels = torch.randint(0, 2, (batch_size,))

    batch = ((x1, x2), labels)

    # 测试训练步骤
    loss = lightning_module.training_step(batch, 0)
    print(f"\nTraining loss: {loss.item():.4f}")

    print("\n✅ Lightning module test passed!")
