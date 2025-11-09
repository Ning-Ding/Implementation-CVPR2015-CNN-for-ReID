"""
Models module for Person Re-Identification
模型模块

包含 Siamese CNN 及其变体实现
"""

from .layers import (
    CrossInputNeighborhoodDifferences,
    PatchSummaryConv,
    TiedConvBlock,
)
from .siamese_cnn import SiameseCNN, create_siamese_cnn
from .lightning_module import ReIDLightningModule, ContrastiveLoss, PolynomialLR

__all__ = [
    # Layers
    "CrossInputNeighborhoodDifferences",
    "PatchSummaryConv",
    "TiedConvBlock",
    # Models
    "SiameseCNN",
    "create_siamese_cnn",
    # Lightning
    "ReIDLightningModule",
    "ContrastiveLoss",
    "PolynomialLR",
]
