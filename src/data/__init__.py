"""
Data processing module for Person Re-Identification
数据处理模块

包含数据集类、数据增强和数据加载工具
"""

from .base_dataset import BaseReIDDataset, PairSamplingStrategy
from .cuhk03_dataset import CUHK03Dataset
from .market1501_dataset import Market1501Dataset
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_legacy_augmentation_transform,
    create_transforms_from_config,
)

__all__ = [
    # Base classes
    "BaseReIDDataset",
    "PairSamplingStrategy",
    # Datasets
    "CUHK03Dataset",
    "Market1501Dataset",
    # Transforms
    "get_train_transforms",
    "get_val_transforms",
    "get_legacy_augmentation_transform",
    "create_transforms_from_config",
]
