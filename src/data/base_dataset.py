"""
Base dataset class for Person Re-Identification
人员重识别数据集基类

提供统一的数据集接口和通用功能
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Literal
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as A


class BaseReIDDataset(Dataset, ABC):
    """
    Person Re-ID 数据集抽象基类

    所有 ReID 数据集都应继承此类并实现抽象方法

    Attributes:
        root: 数据集根目录
        mode: 数据集模式 ('train', 'val', 'test')
        transform: 数据增强变换
        return_pairs: 是否返回图像对 (用于 Siamese 网络)
    """

    def __init__(
        self,
        root: str | Path,
        mode: Literal["train", "val", "test"] = "train",
        transform: Optional[A.Compose] = None,
        return_pairs: bool = True,
    ):
        """
        初始化数据集

        Args:
            root: 数据集根目录
            mode: 模式 ('train', 'val', 'test')
            transform: Albumentations 变换
            return_pairs: 是否返回图像对（Siamese 网络需要）
        """
        self.root = Path(root)
        self.mode = mode
        self.transform = transform
        self.return_pairs = return_pairs

        # 数据集统计信息
        self.num_identities = 0
        self.num_images = 0
        self.identity_to_images: Dict[int, list] = {}

        # 加载数据集
        self._load_dataset()

    @abstractmethod
    def _load_dataset(self):
        """
        加载数据集 (子类必须实现)

        应该设置:
            - self.num_identities
            - self.num_images
            - self.identity_to_images
            - 其他必要的数据结构
        """
        pass

    @abstractmethod
    def _get_single_item(self, index: int) -> Tuple[np.ndarray, int]:
        """
        获取单个图像及其身份标签

        Args:
            index: 图像索引

        Returns:
            (image, person_id): 图像数组和身份ID
        """
        pass

    def _get_positive_pair(self, person_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取正样本对（同一人的两张不同图像）

        Args:
            person_id: 人员身份ID

        Returns:
            (image1, image2): 两张图像

        Raises:
            ValueError: 如果该人的图像数量少于2张
        """
        images = self.identity_to_images.get(person_id, [])
        if len(images) < 2:
            raise ValueError(f"Person {person_id} has less than 2 images")

        idx1, idx2 = np.random.choice(len(images), 2, replace=False)
        image1 = self._load_image(images[idx1])
        image2 = self._load_image(images[idx2])

        return image1, image2

    def _get_negative_pair(self, person_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取负样本对（不同人的图像）

        Args:
            person_id: 参考人员ID

        Returns:
            (image1, image2): 两张不同人的图像
        """
        # 随机选择两个不同的人
        available_ids = [pid for pid in self.identity_to_images.keys() if pid != person_id]
        if len(available_ids) < 1:
            raise ValueError("Not enough identities for negative pairs")

        other_id = np.random.choice(available_ids)

        # 获取两个人的随机图像
        images1 = self.identity_to_images[person_id]
        images2 = self.identity_to_images[other_id]

        idx1 = np.random.choice(len(images1))
        idx2 = np.random.choice(len(images2))

        image1 = self._load_image(images1[idx1])
        image2 = self._load_image(images2[idx2])

        return image1, image2

    @abstractmethod
    def _load_image(self, image_id: Any) -> np.ndarray:
        """
        加载图像 (子类必须实现)

        Args:
            image_id: 图像标识符 (可以是路径、索引等)

        Returns:
            image: NumPy 数组 (H, W, C), RGB格式, uint8
        """
        pass

    def _apply_transform(self, image: np.ndarray) -> torch.Tensor:
        """
        应用数据增强变换

        Args:
            image: 输入图像 (H, W, C), RGB, uint8

        Returns:
            tensor: 变换后的图像 tensor (C, H, W)
        """
        if self.transform is not None:
            augmented = self.transform(image=image)
            return augmented["image"]
        else:
            # 默认转换：转为 float 并归一化到 [0, 1]
            image = image.astype(np.float32) / 255.0
            # HWC -> CHW
            image = np.transpose(image, (2, 0, 1))
            return torch.from_numpy(image)

    def __len__(self) -> int:
        """数据集大小"""
        return self.num_images

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, ...]:
        """
        获取数据项

        根据 self.return_pairs 决定返回格式:
        - return_pairs=True: ((img1, img2), label)  # label: 0=different, 1=same
        - return_pairs=False: (img, person_id)

        Args:
            index: 索引

        Returns:
            根据 return_pairs 返回不同格式
        """
        if not self.return_pairs:
            # 单图像模式
            image, person_id = self._get_single_item(index)
            image_tensor = self._apply_transform(image)
            return image_tensor, person_id
        else:
            # 配对模式
            # 随机决定生成正样本对还是负样本对
            is_positive = np.random.rand() > 0.5

            # 获取参考人员ID
            person_id = index % self.num_identities

            try:
                if is_positive:
                    image1, image2 = self._get_positive_pair(person_id)
                    label = 1  # 同一人
                else:
                    image1, image2 = self._get_negative_pair(person_id)
                    label = 0  # 不同人

                # 应用变换
                tensor1 = self._apply_transform(image1)
                tensor2 = self._apply_transform(image2)

                return (tensor1, tensor2), label

            except (ValueError, IndexError) as e:
                # 如果出错，返回一个负样本对
                print(f"Warning: Error generating pair for index {index}: {e}")
                person_id1 = index % self.num_identities
                person_id2 = (index + 1) % self.num_identities
                images1 = self.identity_to_images[person_id1]
                images2 = self.identity_to_images[person_id2]
                image1 = self._load_image(images1[0])
                image2 = self._load_image(images2[0])
                tensor1 = self._apply_transform(image1)
                tensor2 = self._apply_transform(image2)
                return (tensor1, tensor2), 0

    def get_stats(self) -> Dict[str, Any]:
        """
        获取数据集统计信息

        Returns:
            统计信息字典
        """
        return {
            "mode": self.mode,
            "num_identities": self.num_identities,
            "num_images": self.num_images,
            "avg_images_per_identity": self.num_images / max(self.num_identities, 1),
        }

    def __repr__(self) -> str:
        """字符串表示"""
        stats = self.get_stats()
        return (
            f"{self.__class__.__name__}(\n"
            f"  mode={stats['mode']},\n"
            f"  num_identities={stats['num_identities']},\n"
            f"  num_images={stats['num_images']},\n"
            f"  avg_images_per_id={stats['avg_images_per_identity']:.2f},\n"
            f"  return_pairs={self.return_pairs}\n"
            f")"
        )


class PairSamplingStrategy:
    """
    配对采样策略

    控制正负样本对的生成比例
    """

    def __init__(self, pattern: list[int] = [1, 0, 0]):
        """
        初始化采样策略

        Args:
            pattern: 采样模式列表
                     1 = 正样本对, 0 = 负样本对
                     例如: [1, 0, 0] 表示 1个正样本对, 2个负样本对循环
        """
        self.pattern = pattern
        self.index = 0

    def is_positive(self) -> bool:
        """
        判断当前应该采样正样本还是负样本

        Returns:
            True: 正样本, False: 负样本
        """
        result = bool(self.pattern[self.index])
        self.index = (self.index + 1) % len(self.pattern)
        return result

    def reset(self):
        """重置采样索引"""
        self.index = 0
