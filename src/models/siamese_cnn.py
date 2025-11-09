"""
Siamese CNN for Person Re-Identification
用于人员重识别的孪生卷积神经网络

实现 CVPR 2015 论文: "An Improved Deep Learning Architecture for Person Re-Identification"

网络结构:
1. Tied Convolution Layers (权重共享)
2. Cross-Input Neighborhood Differences (交叉输入邻域差异)
3. Patch Summary Features (图块摘要特征)
4. Across-Patch Features (跨图块特征)
5. Fully Connected Layers (全连接层)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from .layers import (
    TiedConvBlock,
    CrossInputNeighborhoodDifferences,
    PatchSummaryConv,
)


class SiameseCNN(nn.Module):
    """
    原始 CVPR 2015 Siamese CNN 架构

    Input: 两张图像 (160, 60, 3)
    Output: 二分类概率 [不同人, 同一人]

    Architecture:
        1. Conv1: 20 filters, 5x5, MaxPool 2x2  ->  (78, 28, 20)
        2. Conv2: 25 filters, 5x5, MaxPool 2x2  ->  (37, 12, 25)
        3. Cross-Input Neighborhood Differences ->  (185, 60, 25)
        4. Patch Summary: Conv 5x5, stride=5   ->  (37, 12, 25)
        5. Across-Patch: Conv 3x3, MaxPool     ->  (17, 5, 25)
        6. Concat + FC: 500 -> 2
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (160, 60),
        num_classes: int = 2,
        weight_decay: float = 0.00025,
        dropout: float = 0.0,
    ):
        """
        初始化 Siamese CNN

        Args:
            input_size: 输入图像尺寸 (height, width)
            num_classes: 输出类别数 (默认2: 不同/相同)
            weight_decay: L2 正则化系数
            dropout: Dropout 概率 (0 表示不使用)
        """
        super().__init__()

        self.input_size = input_size
        self.num_classes = num_classes
        self.weight_decay = weight_decay

        # ===== Tied Convolution Blocks (权重共享) =====
        self.conv1 = TiedConvBlock(
            in_channels=3,
            out_channels=20,
            kernel_size=5,
            pool_size=2,
            weight_decay=weight_decay,
        )
        # Output: (B, 20, 78, 28)

        self.conv2 = TiedConvBlock(
            in_channels=20,
            out_channels=25,
            kernel_size=5,
            pool_size=2,
            weight_decay=weight_decay,
        )
        # Output: (B, 25, 37, 12)

        # ===== Cross-Input Neighborhood Differences =====
        self.cross_input = CrossInputNeighborhoodDifferences(neighborhood_size=5)
        # Output: (B, 25, 185, 60)

        # ===== Patch Summary Features =====
        self.patch_summary1 = PatchSummaryConv(
            in_channels=25,
            out_channels=25,
            kernel_size=5,
            stride=5,
            weight_decay=weight_decay,
        )
        # Output: (B, 25, 37, 12)

        self.patch_summary2 = PatchSummaryConv(
            in_channels=25,
            out_channels=25,
            kernel_size=5,
            stride=5,
            weight_decay=weight_decay,
        )

        # ===== Across-Patch Features =====
        self.across_patch1 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )
        # Output: (B, 25, 18, 6)

        self.across_patch2 = nn.Sequential(
            nn.Conv2d(25, 25, kernel_size=3, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
        )

        # ===== Higher-Order Relationships (Fully Connected) =====
        # 动态计算 flatten 后的特征维度（支持任意输入尺寸）
        # 使用 dummy forward pass 确定实际维度
        fc_input_dim, embedding_input_dim = self._compute_feature_dims(input_size)

        self.fc_input_dim = fc_input_dim
        self.embedding_input_dim = embedding_input_dim

        self.fc1 = nn.Linear(self.fc_input_dim, 500)
        self.relu_fc = nn.ReLU(inplace=True)

        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.fc2 = nn.Linear(500, num_classes)

        # ===== Embedding Projection for Contrastive Learning =====
        # 用于从单张图像提取 embedding (跳过 pair-wise 操作)
        # Input: flattened conv2 features
        # Output: (B, 500) embedding
        self.embedding_projection = nn.Linear(self.embedding_input_dim, 500)

        # Weight initialization for FC layers
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight, std=0.001)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.kaiming_normal_(self.embedding_projection.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.embedding_projection.bias, 0)

    def _compute_feature_dims(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        通过 dummy forward pass 计算特征维度

        这个方法运行一个 dummy 前向传播来确定：
        1. Pair-wise path (after concat): fc_input_dim
        2. Single-image path (after conv2): embedding_input_dim

        Args:
            input_size: (height, width) 输入图像尺寸

        Returns:
            (fc_input_dim, embedding_input_dim): 两个路径的 flatten 后维度
        """
        with torch.no_grad():
            # 创建 dummy 输入
            h, w = input_size
            dummy_x1 = torch.zeros(1, 3, h, w)
            dummy_x2 = torch.zeros(1, 3, h, w)

            # === Single-image path (for embedding) ===
            # Conv layers only
            feat = self.conv1(dummy_x1)
            feat = self.conv2(feat)
            embedding_input_dim = feat.numel()  # Total elements for single image

            # === Pair-wise path (for classification) ===
            # Full forward until concat
            feat1 = self.conv2(self.conv1(dummy_x1))
            feat2 = self.conv2(self.conv1(dummy_x2))

            # Cross-input
            cross1, cross2 = self.cross_input(feat1, feat2)

            # Patch summary
            patch1 = self.patch_summary1(cross1)
            patch2 = self.patch_summary2(cross2)

            # Across-patch
            across1 = self.across_patch1(patch1)
            across2 = self.across_patch2(patch2)

            # Concat and get total dimension
            combined = torch.cat([across1, across2], dim=1)
            fc_input_dim = combined.numel()  # Total elements after concat

        return fc_input_dim, embedding_input_dim

    def forward_once(self, x: torch.Tensor) -> torch.Tensor:
        """
        单分支前向传播 (Tied convolutions)

        Args:
            x: (B, 3, H, W)

        Returns:
            features: (B, 25, 37, 12)
        """
        x = self.conv1(x)  # (B, 20, 78, 28)
        x = self.conv2(x)  # (B, 25, 37, 12)
        return x

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播

        Args:
            x1, x2: (B, 3, H, W) 输入图像对

        Returns:
            output: (B, num_classes) 分类 logits (未经 softmax)
        """
        # Tied Convolutions
        feat1 = self.forward_once(x1)  # (B, 25, 37, 12)
        feat2 = self.forward_once(x2)

        # Cross-Input Neighborhood Differences
        cross1, cross2 = self.cross_input(feat1, feat2)  # (B, 25, 185, 60)

        # Patch Summary Features
        patch1 = self.patch_summary1(cross1)  # (B, 25, 37, 12)
        patch2 = self.patch_summary2(cross2)

        # Across-Patch Features
        across1 = self.across_patch1(patch1)  # (B, 25, 18, 6)
        across2 = self.across_patch2(patch2)

        # Concatenate
        combined = torch.cat([across1, across2], dim=1)  # (B, 50, 18, 6)

        # Flatten
        combined = combined.view(combined.size(0), -1)  # (B, 5400)

        # Fully Connected
        x = self.fc1(combined)  # (B, 500)
        x = self.relu_fc(x)

        if self.dropout is not None:
            x = self.dropout(x)

        x = self.fc2(x)  # (B, num_classes)

        return x

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取单张图像的特征向量 (用于 contrastive learning 和检索)

        NOTE: 对于 contrastive learning，我们需要从单张图像提取 embedding。
        原始架构设计用于处理图像对（需要 cross-input differences），
        因此我们使用单分支路径：
        1. Tied convolutions 提取特征 (conv1 + conv2)
        2. Flatten 卷积特征
        3. 通过 embedding_projection 层投影到 500 维

        这种方法避免了 cross-input layer 的 pair-wise 依赖，
        同时保持与 FC1 相同的输出维度。

        Args:
            x: (B, 3, H, W) 单张图像

        Returns:
            embedding: (B, 500) 特征向量
        """
        # Step 1: Tied convolutions 提取特征
        feat = self.forward_once(x)  # (B, 25, 37, 12)

        # Step 2: Flatten 卷积特征
        flattened = feat.view(feat.size(0), -1)  # (B, 25*37*12) = (B, 11100)

        # Step 3: 投影到 500 维 embedding 空间
        embedding = self.embedding_projection(flattened)  # (B, 500)
        embedding = self.relu_fc(embedding)

        return embedding


def create_siamese_cnn(
    input_size: Tuple[int, int] = (160, 60),
    num_classes: int = 2,
    pretrained: bool = False,
    **kwargs
) -> SiameseCNN:
    """
    创建 Siamese CNN 模型

    Args:
        input_size: 输入尺寸
        num_classes: 类别数
        pretrained: 是否加载预训练权重 (暂不支持)
        **kwargs: 其他参数

    Returns:
        model: SiameseCNN 实例
    """
    model = SiameseCNN(
        input_size=input_size,
        num_classes=num_classes,
        **kwargs
    )

    if pretrained:
        raise NotImplementedError("Pretrained weights not available yet")

    return model


if __name__ == "__main__":
    # 测试模型
    print("Testing SiameseCNN...")

    model = create_siamese_cnn(input_size=(160, 60), num_classes=2)

    # 打印模型结构
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 测试前向传播
    batch_size = 4
    x1 = torch.randn(batch_size, 3, 160, 60)
    x2 = torch.randn(batch_size, 3, 160, 60)

    print(f"\nInput shape: {x1.shape}")

    output = model(x1, x2)
    print(f"Output shape: {output.shape}")
    assert output.shape == (batch_size, 2), f"Expected shape ({batch_size}, 2), got {output.shape}"

    # 测试 embedding 提取
    embedding = model.get_embedding(x1)
    print(f"Embedding shape: {embedding.shape}")
    assert embedding.shape == (batch_size, 500), f"Expected shape ({batch_size}, 500), got {embedding.shape}"

    print("\n✅ SiameseCNN test passed!")

    # 计算 FLOPs 和参数量
    from torchinfo import summary

    print("\nModel Summary:")
    summary(
        model,
        input_data=[x1, x2],
        col_names=["input_size", "output_size", "num_params", "trainable"],
        depth=3,
        row_settings=["var_names"]
    )
