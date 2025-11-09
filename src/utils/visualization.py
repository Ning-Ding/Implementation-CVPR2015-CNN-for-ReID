"""
Visualization utilities
可视化工具
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Optional, List


def plot_cmc_curve(
    cmc: np.ndarray,
    save_path: Optional[str] = None,
    title: str = "CMC Curve",
    max_rank: int = 50,
):
    """绘制 CMC 曲线"""
    plt.figure(figsize=(10, 6))

    ranks = np.arange(1, min(len(cmc), max_rank) + 1)
    plt.plot(ranks, cmc[:max_rank], linewidth=2, marker='o', markersize=4)

    plt.xlabel("Rank", fontsize=12)
    plt.ylabel("Matching Rate", fontsize=12)
    plt.title(title, fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xlim([1, max_rank])
    plt.ylim([0, 1.0])

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    save_path: Optional[str] = None,
):
    """绘制训练曲线"""
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    epochs = np.arange(1, len(train_losses) + 1)
    ax.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax.plot(epochs, val_losses, label='Val Loss', linewidth=2)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Training Curves", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
