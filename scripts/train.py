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

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import CUHK03Dataset, Market1501Dataset, create_transforms_from_config
from src.models import create_siamese_cnn, ReIDLightningModule
from src.utils.logger import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="Train ReID model")
    parser.add_argument("--config", type=str, default="config/cuhk03.yaml", help="Config file")
    parser.add_argument("--gpus", type=int, default=1, help="Number of GPUs")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()

    # Load config
    with open(args.config) as f:
        config = yaml.safe_load(f)

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
