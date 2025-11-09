"""
Custom layers for Person Re-Identification
自定义层实现

包含 CVPR 2015 论文中的 Cross-Input Neighborhood Differences 层等创新结构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class CrossInputNeighborhoodDifferences(nn.Module):
    """
    Cross-Input Neighborhood Differences Layer
    交叉输入邻域差异层

    论文核心创新：计算两个输入特征图之间的局部空间关系

    原理:
    1. 对每个输入特征图，提取每个位置的 5x5 邻域
    2. 将邻域 upsampling 到与原始特征图相同的空间尺寸
    3. 计算两个输入之间的差异: output1 = upsampled_input1 - neighbor_input2
    4. 这样可以捕获两幅图像在局部空间上的差异模式

    参数:
        neighborhood_size: 邻域大小 (default: 5)

    输入:
        x1, x2: (B, C, H, W) 特征图对

    输出:
        y1, y2: (B, C, H*K, W*K) Upsampled 特征图
                K = neighborhood_size
    """

    def __init__(self, neighborhood_size: int = 5):
        super().__init__()
        self.neighborhood_size = neighborhood_size
        self.padding = neighborhood_size // 2  # 2 for size=5

    def _upsample_neighbors(self, x: torch.Tensor) -> torch.Tensor:
        """
        提取邻域并 upsample

        Args:
            x: (B, C, H, W)

        Returns:
            upsampled: (B, C, H*K, W*K) where K = neighborhood_size
        """
        B, C, H, W = x.shape
        K = self.neighborhood_size

        # Padding
        x_pad = F.pad(x, (self.padding,) * 4, mode='constant', value=0)
        # x_pad: (B, C, H+4, W+4)

        output_list = []

        # 遍历原始特征图的每个位置
        for i in range(H):
            row_list = []
            for j in range(W):
                # 提取 5x5 邻域
                neighborhood = x_pad[:, :, i:i+K, j:j+K]  # (B, C, K, K)
                row_list.append(neighborhood)

            # 水平拼接
            row = torch.cat(row_list, dim=3)  # (B, C, K, W*K)
            output_list.append(row)

        # 垂直拼接
        output = torch.cat(output_list, dim=2)  # (B, C, H*K, W*K)

        return output

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            x1, x2: (B, C, H, W) 输入特征图对

        Returns:
            y1, y2: (B, C, H*K, W*K) Cross-input 差异特征
        """
        # Upsample 当前特征
        x1_up = F.interpolate(
            x1,
            scale_factor=self.neighborhood_size,
            mode='nearest'
        )  # (B, C, H*K, W*K)

        x2_up = F.interpolate(
            x2,
            scale_factor=self.neighborhood_size,
            mode='nearest'
        )

        # 提取邻域并 upsample
        x1_neighbors = self._upsample_neighbors(x1)  # (B, C, H*K, W*K)
        x2_neighbors = self._upsample_neighbors(x2)

        # 计算差异 (negated neighbors)
        y1 = x1_up - x2_neighbors
        y2 = x2_up - x1_neighbors

        return y1, y2


class PatchSummaryConv(nn.Module):
    """
    Patch Summary Features Layer
    图块摘要特征层

    使用大步长卷积来汇总图块信息
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        stride: int = 5,
        weight_decay: float = 0.00025,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
        )
        self.relu = nn.ReLU(inplace=True)

        # Weight initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C', H', W')
        """
        return self.relu(self.conv(x))


class TiedConvBlock(nn.Module):
    """
    Tied (Shared) Convolutional Block
    权重共享卷积块

    用于 Siamese 网络中，两个分支共享相同的权重
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 5,
        pool_size: int = 2,
        weight_decay: float = 0.00025,
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=0,
        )
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

        # Weight initialization
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)

        Returns:
            out: (B, C', H', W')
        """
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        return x


if __name__ == "__main__":
    # 测试 Cross-Input Layer
    print("Testing CrossInputNeighborhoodDifferences...")

    cross_input = CrossInputNeighborhoodDifferences(neighborhood_size=5)

    # 创建测试输入
    x1 = torch.randn(2, 25, 37, 12)  # Batch=2, C=25, H=37, W=12
    x2 = torch.randn(2, 25, 37, 12)

    y1, y2 = cross_input(x1, x2)

    print(f"Input shape: {x1.shape}")
    print(f"Output shape: {y1.shape}")
    print(f"Expected shape: (2, 25, {37*5}, {12*5})")
    assert y1.shape == (2, 25, 185, 60), "Shape mismatch!"
    print("✅ Cross-Input layer test passed!")

    # 测试 PatchSummaryConv
    print("\nTesting PatchSummaryConv...")
    patch_conv = PatchSummaryConv(in_channels=25, out_channels=25, kernel_size=5, stride=5)
    y = patch_conv(y1)
    print(f"Input shape: {y1.shape}")
    print(f"Output shape: {y.shape}")
    print("✅ PatchSummaryConv test passed!")

    # 测试 TiedConvBlock
    print("\nTesting TiedConvBlock...")
    conv_block = TiedConvBlock(in_channels=3, out_channels=20, kernel_size=5, pool_size=2)
    x = torch.randn(2, 3, 160, 60)
    y = conv_block(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print("✅ TiedConvBlock test passed!")

    print("\n✅ All layer tests passed!")
