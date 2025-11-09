"""
Training script for Person Re-Identification
训练脚本

Usage:
    python scripts/train.py --config config/cuhk03.yaml
"""

import argparse
from pathlib import Path
import yaml
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CUHK03Dataset, Market1501Dataset, create_transforms_from_config
from src.models import create_siamese_cnn, ReIDLightningModule
from src.utils.logger import setup_logger


def load_config(config_path: str) -> dict:
    """
    加载配置文件，支持 Hydra 风格的继承
    Load config file with Hydra-style inheritance support

    Args:
        config_path: 配置文件路径

    Returns:
        完整的配置字典（已合并继承）
    """
    config_path = Path(config_path)

    # 使用 OmegaConf 加载配置
    cfg = OmegaConf.load(config_path)

    # 检查是否有 defaults 继承
    if "defaults" in cfg:
        defaults = cfg.defaults
        base_configs = []

        # 加载所有基础配置
        for default in defaults:
            if isinstance(default, str):
                # 简单的字符串引用，如 "base"
                base_name = default
            elif isinstance(default, dict):
                # 字典格式，提取第一个键
                base_name = list(default.keys())[0]
            else:
                continue

            # 构建基础配置文件路径
            base_path = config_path.parent / f"{base_name}.yaml"
            if base_path.exists():
                base_cfg = OmegaConf.load(base_path)
                base_configs.append(base_cfg)

        # 合并配置：base -> child (child 覆盖 base)
        if base_configs:
            # 从最底层开始合并
            merged = base_configs[0]
            for base_cfg in base_configs[1:]:
                merged = OmegaConf.merge(merged, base_cfg)
            # 最后合并当前配置（覆盖基础配置）
            merged = OmegaConf.merge(merged, cfg)
            cfg = merged

    # 删除 defaults 键（不需要在运行时使用）
    if "defaults" in cfg:
        cfg = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(cfg, dict):
            cfg.pop("defaults", None)
    else:
        cfg = OmegaConf.to_container(cfg, resolve=True)

    return cfg


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReID model")
    parser.add_argument("--config", type=str, default="config/cuhk03.yaml", help="Config file")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config with inheritance support
    # 修复: 使用 OmegaConf 加载配置，支持 Hydra 风格的 defaults 继承
    config = load_config(args.config)

    # Setup logger
    logger = setup_logger("train", log_file="logs/train.log")
    logger.info(f"Config: {args.config}")

    # Create datasets
    dataset_name = config["dataset"]["name"]
    if dataset_name == "cuhk03":
        train_transform = create_transforms_from_config(config, mode="train")
        val_transform = create_transforms_from_config(config, mode="val")

        train_dataset = CUHK03Dataset(
            root=config["paths"]["data_root"],
            mode="train",
            transform=train_transform,
            return_pairs=True,
            create_if_not_exists=True,
        )

        val_dataset = CUHK03Dataset(
            root=config["paths"]["data_root"],
            mode="val",
            transform=val_transform,
            return_pairs=True,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Val set: {len(val_dataset)} samples")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"].get("val_batch_size", 200),
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    # Create model
    model = create_siamese_cnn(
        input_size=tuple(config["model"]["input_size"]),
        num_classes=config["model"]["num_classes"],
    )

    # Create Lightning module
    lightning_module = ReIDLightningModule(
        model=model,
        loss_type=config["loss"]["type"],
        loss_params=config["loss"].get(config["loss"]["type"], {}),
        optimizer_name=config["optimizer"]["name"],
        learning_rate=config["optimizer"]["lr"],
        momentum=config["optimizer"]["momentum"],
        weight_decay=config["optimizer"]["weight_decay"],
        scheduler_name=config["scheduler"]["name"],
        scheduler_params=config["scheduler"].get(config["scheduler"]["name"], {}),
    )

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=config["experiment"]["checkpoint_dir"],
        filename="{epoch}-{val_loss:.4f}",
        save_top_k=config["training"]["save_top_k"],
        monitor=config["training"]["monitor"],
        mode=config["training"]["mode"],
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    callbacks = [checkpoint_callback, lr_monitor]

    if config["training"]["early_stopping"]["enabled"]:
        early_stop = EarlyStopping(
            monitor=config["training"]["monitor"],
            patience=config["training"]["early_stopping"]["patience"],
            min_delta=config["training"]["early_stopping"]["min_delta"],
            mode=config["training"]["mode"],
        )
        callbacks.append(early_stop)

    # Logger
    tb_logger = TensorBoardLogger(
        save_dir=config["logging"]["tensorboard"]["log_dir"],
        name=config["experiment"]["name"],
    )

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["training"]["max_epochs"],
        accelerator=config["experiment"]["accelerator"],
        devices=args.gpus,
        precision=config["training"]["precision"],
        callbacks=callbacks,
        logger=tb_logger,
        log_every_n_steps=config["logging"]["log_every_n_steps"],
        gradient_clip_val=config["training"]["gradient_clip_val"],
    )

    # Train
    logger.info("Starting training...")
    trainer.fit(
        lightning_module,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        ckpt_path=args.resume,
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
