"""
CUHK03 Dataset for Person Re-Identification
CUHK03 人员重识别数据集

CUHK03 是最经典的 ReID 数据集之一，包含 1467 个身份，每个身份有多个视角的图像。

Dataset Structure / 数据集结构:
    - 原始文件: cuhk-03.mat (MATLAB格式)
    - 处理后: cuhk-03.hdf5 (按 identity 组织)
    - 索引文件: cuhk-03-index.hdf5 (train/val/test split)

Reference:
    Li et al., "DeepReID: Deep Filter Pairing Neural Network for Person Re-Identification", CVPR 2014
"""

from typing import Optional, Literal, Tuple
from pathlib import Path
import numpy as np
import h5py
from PIL import Image
import scipy.io

from .base_dataset import BaseReIDDataset


class CUHK03Dataset(BaseReIDDataset):
    """
    CUHK03 数据集

    特点:
    - 1467 个身份 (1360 labeled + 107 detected)
    - 5 个摄像头视角
    - 每个身份平均 10 张图像

    数据划分:
    - Train: 1260 identities
    - Val:   100 identities
    - Test:  100 identities
    """

    def __init__(
        self,
        root: str | Path,
        mode: Literal["train", "val", "test"] = "train",
        dataset_type: Literal["labeled", "detected"] = "labeled",
        transform: Optional = None,
        return_pairs: bool = True,
        create_if_not_exists: bool = False,
    ):
        """
        初始化 CUHK03 数据集

        Args:
            root: 数据集根目录
            mode: 数据集模式 ('train', 'val', 'test')
            dataset_type: 'labeled' (人工标注) 或 'detected' (DPM检测)
            transform: 数据增强变换
            return_pairs: 是否返回图像对
            create_if_not_exists: 如果处理后的文件不存在，是否自动创建
        """
        self.dataset_type = dataset_type
        self.create_if_not_exists = create_if_not_exists

        # 文件路径
        self.original_file = Path(root) / "cuhk-03.mat"
        self.processed_file = Path(root) / "cuhk-03.hdf5"
        self.index_file = Path(root) / "cuhk-03-index.hdf5"

        # 数据容器
        self.data_file: Optional[h5py.File] = None
        self.identity_indices = []  # 当前 mode 的 identity 索引列表

        super().__init__(root, mode, transform, return_pairs)

    def _load_dataset(self):
        """加载数据集"""
        # 检查并创建处理后的文件
        if not self.processed_file.exists():
            if self.create_if_not_exists and self.original_file.exists():
                print(f"Processing dataset from {self.original_file}...")
                self._create_processed_dataset()
            else:
                raise FileNotFoundError(
                    f"Processed dataset not found: {self.processed_file}\n"
                    f"Please download cuhk-03.mat or set create_if_not_exists=True"
                )

        # 检查并创建索引文件
        if not self.index_file.exists():
            if self.create_if_not_exists:
                print(f"Creating index file...")
                self._create_index_file()
            else:
                raise FileNotFoundError(
                    f"Index file not found: {self.index_file}\n"
                    f"Set create_if_not_exists=True to create it automatically"
                )

        # 加载索引
        with h5py.File(self.index_file, "r") as f:
            # 将 'val' 映射到 'valid'
            mode_key = "valid" if self.mode == "val" else self.mode
            if mode_key not in f:
                raise KeyError(f"Mode '{mode_key}' not found in index file")
            self.identity_indices = f[mode_key][:].tolist()

        # 打开数据文件（保持打开以提高性能）
        self.data_file = h5py.File(self.processed_file, "r")

        # 构建 identity_to_images 映射
        self.identity_to_images = {}
        self.num_images = 0

        for person_id in self.identity_indices:
            if str(person_id) in self.data_file:
                num_imgs = self.data_file[str(person_id)].shape[0]
                # 存储 (person_id, image_index) 元组
                self.identity_to_images[person_id] = list(range(num_imgs))
                self.num_images += num_imgs

        self.num_identities = len(self.identity_indices)

        print(f"Loaded CUHK03 {self.mode} set: {self.num_identities} identities, {self.num_images} images")

    def _create_processed_dataset(self):
        """
        从原始 .mat 文件创建处理后的 HDF5 文件

        原始文件结构:
        - f['labeled'][0][i]: 第 i 个摄像头
        - f['labeled'][0][i][j][k]: 第 i 个摄像头第 k 个人的第 j 张图像
        """
        with scipy.io.loadmat(str(self.original_file)) as mat_data:
            with h5py.File(self.processed_file, "w") as hdf5_file:
                labeled = mat_data[self.dataset_type]

                person_id = 0

                # 遍历摄像头 (0-2 包含 1360 个身份)
                for camera_idx in range(3):
                    camera_data = labeled[0][camera_idx]
                    num_persons = camera_data[0].size

                    for person_idx in range(num_persons):
                        images = []

                        # 获取该人的所有图像 (最多10张)
                        for img_idx in range(10):
                            try:
                                img_ref = camera_data[img_idx][person_idx]
                                img_data = mat_data[img_ref]

                                if img_data.ndim == 3:
                                    # 图像格式: (C, H, W) -> (H, W, C)
                                    img = np.transpose(img_data, (1, 2, 0))

                                    # Resize to standard size
                                    img_pil = Image.fromarray(img.astype(np.uint8))
                                    img_pil = img_pil.resize((60, 160), Image.BILINEAR)

                                    # 转换为 float32 并归一化到 [0, 1]
                                    img_array = np.array(img_pil, dtype=np.float32) / 255.0

                                    images.append(img_array)
                            except (IndexError, ValueError):
                                # 某些人可能没有10张图像
                                break

                        if len(images) > 0:
                            # 保存为 HDF5 dataset
                            hdf5_file.create_dataset(
                                str(person_id),
                                data=np.array(images, dtype=np.float32),
                                compression="gzip",
                            )
                            person_id += 1

                            if person_id % 100 == 0:
                                print(f"Processed {person_id} identities...")

                print(f"Dataset creation complete: {person_id} identities")

    def _create_index_file(self):
        """
        创建训练/验证/测试集的索引文件

        默认划分:
        - Test: 100 identities (随机选择)
        - Val: 100 identities (从剩余中随机选择)
        - Train: 1160 identities (剩余的)
        """
        total_identities = 1360  # CUHK03 labeled 总数

        # 设置随机种子以保证可复现
        np.random.seed(42)

        # 随机划分
        all_indices = np.arange(total_identities)
        np.random.shuffle(all_indices)

        test_indices = all_indices[:100]
        val_indices = all_indices[100:200]
        train_indices = all_indices[200:]

        with h5py.File(self.index_file, "w") as f:
            f.create_dataset("train", data=train_indices)
            f.create_dataset("valid", data=val_indices)
            f.create_dataset("test", data=test_indices)

        print(f"Index file created: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")

    def _load_image(self, image_id: int) -> np.ndarray:
        """
        加载图像

        Args:
            image_id: 图像索引（在该 identity 的图像列表中的索引）

        Returns:
            image: (H, W, C) NumPy 数组, RGB格式, float32, [0, 1]
        """
        # image_id 实际上是当前迭代中的 person_id
        # 我们需要重新设计这个逻辑
        pass

    def _get_single_item(self, index: int) -> Tuple[np.ndarray, int]:
        """
        获取单个图像及其身份标签

        Args:
            index: 全局图像索引

        Returns:
            (image, person_id): 图像和身份ID
        """
        # 找到对应的 identity 和图像索引
        cumsum = 0
        for person_id in self.identity_indices:
            num_imgs = len(self.identity_to_images[person_id])
            if index < cumsum + num_imgs:
                img_idx = index - cumsum
                image = self.data_file[str(person_id)][img_idx]
                # 转换为 uint8 以便数据增强
                image_uint8 = (image * 255).astype(np.uint8)
                return image_uint8, person_id
            cumsum += num_imgs

        raise IndexError(f"Index {index} out of range")

    def _load_image_by_person_and_index(self, person_id: int, img_idx: int) -> np.ndarray:
        """
        根据 person_id 和图像索引加载图像

        Args:
            person_id: 人员ID
            img_idx: 图像索引

        Returns:
            image: (H, W, C) NumPy 数组, RGB, uint8
        """
        image = self.data_file[str(person_id)][img_idx]
        # float32 [0,1] -> uint8 [0,255]
        return (image * 255).astype(np.uint8)

    def _get_positive_pair(self, person_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取正样本对（同一人的两张图像）"""
        images_indices = self.identity_to_images[person_id]
        if len(images_indices) < 2:
            raise ValueError(f"Person {person_id} has less than 2 images")

        idx1, idx2 = np.random.choice(len(images_indices), 2, replace=False)
        image1 = self._load_image_by_person_and_index(person_id, idx1)
        image2 = self._load_image_by_person_and_index(person_id, idx2)

        return image1, image2

    def _get_negative_pair(self, person_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """获取负样本对（不同人的图像）"""
        # 选择另一个不同的 identity
        other_id = np.random.choice(
            [pid for pid in self.identity_indices if pid != person_id]
        )

        # 各选一张图像
        idx1 = np.random.choice(self.identity_to_images[person_id])
        idx2 = np.random.choice(self.identity_to_images[other_id])

        image1 = self._load_image_by_person_and_index(person_id, idx1)
        image2 = self._load_image_by_person_and_index(other_id, idx2)

        return image1, image2

    def __del__(self):
        """析构函数：关闭HDF5文件"""
        if hasattr(self, "data_file") and self.data_file is not None:
            self.data_file.close()

    def __repr__(self) -> str:
        """字符串表示"""
        return (
            f"CUHK03Dataset(\n"
            f"  mode={self.mode},\n"
            f"  type={self.dataset_type},\n"
            f"  num_identities={self.num_identities},\n"
            f"  num_images={self.num_images},\n"
            f"  avg_images_per_id={self.num_images / max(self.num_identities, 1):.2f}\n"
            f")"
        )


if __name__ == "__main__":
    # 测试数据集加载
    from pathlib import Path

    # 假设数据在 data/cuhk03/ 目录
    dataset_root = Path("data/cuhk03")

    if dataset_root.exists():
        print("Testing CUHK03 Dataset...")

        # 测试训练集
        train_dataset = CUHK03Dataset(
            root=dataset_root,
            mode="train",
            return_pairs=True,
            create_if_not_exists=True,
        )

        print(f"\n{train_dataset}")
        print(f"\nDataset stats: {train_dataset.get_stats()}")

        # 测试获取一个样本
        (img1, img2), label = train_dataset[0]
        print(f"\nSample pair:")
        print(f"  Image 1 shape: {img1.shape}")
        print(f"  Image 2 shape: {img2.shape}")
        print(f"  Label: {label} ({'same person' if label == 1 else 'different person'})")

        print("\n✅ CUHK03 Dataset test passed!")
    else:
        print(f"Dataset root not found: {dataset_root}")
        print("Skipping dataset test.")
