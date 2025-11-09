"""
Data augmentation and transformation utilities
数据增强和转换工具

使用 Albumentations 库提供高性能的数据增强
"""

from typing import Dict, Any, Optional
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2


def get_train_transforms(
    image_size: tuple[int, int] = (160, 60),
    shift_limit: float = 0.05,
    scale_limit: float = 0.05,
    rotate_limit: int = 5,
    brightness_contrast_p: float = 0.3,
    horizontal_flip_p: float = 0.5,
    normalize: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    获取训练数据增强变换

    Args:
        image_size: 目标图像尺寸 (height, width)
        shift_limit: 平移范围 (相对于图像尺寸的比例)
        scale_limit: 缩放范围
        rotate_limit: 旋转角度范围 (degrees)
        brightness_contrast_p: 亮度对比度调整概率
        horizontal_flip_p: 水平翻转概率
        normalize: 是否进行归一化
        mean: 归一化均值 (ImageNet stats)
        std: 归一化标准差 (ImageNet stats)

    Returns:
        Albumentations 变换组合

    Example:
        >>> transform = get_train_transforms(image_size=(160, 60))
        >>> augmented = transform(image=image)
        >>> tensor = augmented['image']  # PyTorch tensor
    """
    height, width = image_size

    transforms_list = [
        A.Resize(height, width, interpolation=cv2.INTER_LINEAR),
        A.HorizontalFlip(p=horizontal_flip_p),
        A.ShiftScaleRotate(
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=brightness_contrast_p,
        ),
    ]

    if normalize:
        transforms_list.append(
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
        )

    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_val_transforms(
    image_size: tuple[int, int] = (160, 60),
    normalize: bool = True,
    mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: tuple[float, float, float] = (0.229, 0.224, 0.225),
) -> A.Compose:
    """
    获取验证/测试数据变换（不包含数据增强）

    Args:
        image_size: 目标图像尺寸 (height, width)
        normalize: 是否进行归一化
        mean: 归一化均值
        std: 归一化标准差

    Returns:
        Albumentations 变换组合
    """
    height, width = image_size

    transforms_list = [
        A.Resize(height, width, interpolation=cv2.INTER_LINEAR),
    ]

    if normalize:
        transforms_list.append(
            A.Normalize(mean=mean, std=std, max_pixel_value=255.0)
        )

    transforms_list.append(ToTensorV2())

    return A.Compose(transforms_list)


def get_legacy_augmentation_transform(
    image_size: tuple[int, int] = (160, 60),
    padding_ratio: float = 0.05,
) -> A.Compose:
    """
    复现原始论文的数据增强方法

    原始方法：
    1. 在图像四周添加 5% padding
    2. 随机平移 (在 padding 范围内)
    3. 裁剪回原始尺寸

    Args:
        image_size: 目标图像尺寸 (height, width)
        padding_ratio: padding 比例

    Returns:
        Albumentations 变换组合

    Note:
        这个方法忠实复现了原始 CVPR 2015 论文的数据增强策略
        用于验证复现的准确性
    """
    height, width = image_size

    # 计算 padding 尺寸
    padding_h = int(height * padding_ratio)
    padding_w = int(width * padding_ratio)

    return A.Compose([
        A.Resize(height, width),
        # 添加 padding
        A.PadIfNeeded(
            min_height=height + 2 * padding_h,
            min_width=width + 2 * padding_w,
            border_mode=cv2.BORDER_CONSTANT,
            value=0,
        ),
        # 随机裁剪（模拟随机平移）
        A.RandomCrop(height=height, width=width, p=1.0),
        # 转换为 tensor（不归一化，保持 [0, 1] 范围）
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255.0),
        ToTensorV2(),
    ])


def create_transforms_from_config(
    config: Dict[str, Any],
    mode: str = "train",
) -> A.Compose:
    """
    从配置字典创建数据变换

    Args:
        config: 配置字典 (从 YAML 加载)
        mode: 模式 ('train', 'val', 'test')

    Returns:
        Albumentations 变换组合

    Example:
        >>> import yaml
        >>> with open('config/cuhk03.yaml') as f:
        >>>     config = yaml.safe_load(f)
        >>> transform = create_transforms_from_config(config, mode='train')
    """
    model_config = config.get("model", {})
    data_config = config.get("data", {})

    image_size = tuple(model_config.get("input_size", [160, 60]))

    if mode == "train":
        aug_config = data_config.get("augmentation", {})
        return get_train_transforms(
            image_size=image_size,
            shift_limit=aug_config.get("shift_limit", 0.05),
            scale_limit=aug_config.get("scale_limit", 0.05),
            rotate_limit=aug_config.get("rotate_limit", 5),
            brightness_contrast_p=aug_config.get("brightness_contrast", 0.3),
            horizontal_flip_p=aug_config.get("horizontal_flip", 0.5),
            normalize=True,
            mean=data_config.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]),
            std=data_config.get("normalize", {}).get("std", [0.229, 0.224, 0.225]),
        )
    else:
        return get_val_transforms(
            image_size=image_size,
            normalize=True,
            mean=data_config.get("normalize", {}).get("mean", [0.485, 0.456, 0.406]),
            std=data_config.get("normalize", {}).get("std", [0.229, 0.224, 0.225]),
        )


if __name__ == "__main__":
    # 测试数据增强
    import numpy as np
    from PIL import Image

    # 创建测试图像
    test_image = np.random.randint(0, 255, (160, 60, 3), dtype=np.uint8)

    # 测试训练变换
    train_transform = get_train_transforms()
    result = train_transform(image=test_image)
    print(f"Train transform output shape: {result['image'].shape}")
    print(f"Train transform output dtype: {result['image'].dtype}")
    print(f"Train transform output range: [{result['image'].min():.3f}, {result['image'].max():.3f}]")

    # 测试验证变换
    val_transform = get_val_transforms()
    result = val_transform(image=test_image)
    print(f"\nVal transform output shape: {result['image'].shape}")
    print(f"Val transform output dtype: {result['image'].dtype}")

    # 测试原始论文变换
    legacy_transform = get_legacy_augmentation_transform()
    result = legacy_transform(image=test_image)
    print(f"\nLegacy transform output shape: {result['image'].shape}")

    print("\n✅ All transforms working correctly!")
