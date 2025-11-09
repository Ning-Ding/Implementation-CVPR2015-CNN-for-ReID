"""
Market-1501 Dataset for Person Re-Identification
Market-1501 人员重识别数据集

Market-1501 是一个大规模的 ReID 数据集，包含从 6 个摄像头收集的数据。

Dataset Structure / 数据集结构:
    bounding_box_train/  - 训练集图像
    bounding_box_test/   - Gallery 图像
    query/               - Query 图像

Filename Format / 文件命名格式:
    XXXX_cY_sZ_NNNNNN.jpg
    - XXXX: Person ID (0000-1501)
    - cY: Camera ID (c1-c6)
    - sZ: Sequence number
    - NNNNNN: Frame number

Reference:
    Zheng et al., "Scalable Person Re-identification: A Benchmark", ICCV 2015
"""

from typing import Optional, Literal, Tuple, List
from pathlib import Path
import numpy as np
from PIL import Image
import h5py

from .base_dataset import BaseReIDDataset


class Market1501Dataset(BaseReIDDataset):
    """
    Market-1501 数据集

    特点:
    - 1501 个身份 (751 训练, 750 测试)
    - 6 个摄像头
    - 32,668 个标注框 (12,936 训练)

    数据划分:
    - Train: 751 identities, 12,936 images
    - Query: 750 identities, 3,368 images
    - Gallery: 750 identities, 19,732 images
    """

    def __init__(
        self,
        root: str | Path,
        mode: Literal["train", "query", "gallery"] = "train",
        transform: Optional = None,
        return_pairs: bool = True,
    ):
        """
        初始化 Market-1501 数据集

        Args:
            root: 数据集根目录
            mode: 'train', 'query', 或 'gallery'
            transform: 数据增强变换
            return_pairs: 是否返回图像对 (训练时使用)
        """
        self.image_list: List[Tuple[Path, int, int]] = []  # (path, person_id, camera_id)

        super().__init__(root, mode, transform, return_pairs)

    def _load_dataset(self):
        """加载数据集"""
        # 根据 mode 确定目录
        if self.mode == "train":
            img_dir = self.root / "bounding_box_train"
        elif self.mode == "query":
            img_dir = self.root / "query"
        elif self.mode == "gallery":
            img_dir = self.root / "bounding_box_test"
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        if not img_dir.exists():
            raise FileNotFoundError(
                f"Image directory not found: {img_dir}\n"
                f"Please download Market-1501 dataset and extract it to {self.root}"
            )

        # 解析图像文件名
        for img_path in sorted(img_dir.glob("*.jpg")):
            filename = img_path.stem  # 去掉 .jpg 后缀
            parts = filename.split("_")

            if len(parts) < 2:
                continue

            try:
                person_id = int(parts[0])
                camera_id = int(parts[1][1])  # 'c1' -> 1

                # 跳过 junk images (person_id = -1 或 0)
                if person_id <= 0:
                    continue

                # 跳过 distractors (person_id > 1501 在 gallery 中)
                if self.mode == "gallery" and person_id > 1501:
                    continue

                self.image_list.append((img_path, person_id, camera_id))

            except (ValueError, IndexError):
                # 跳过无效文件名
                continue

        # 构建 identity_to_images 映射
        self.identity_to_images = {}
        for idx, (path, person_id, camera_id) in enumerate(self.image_list):
            if person_id not in self.identity_to_images:
                self.identity_to_images[person_id] = []
            self.identity_to_images[person_id].append(idx)

        self.num_identities = len(self.identity_to_images)
        self.num_images = len(self.image_list)

        # 设置 identity_list 用于正确的索引映射
        # 修复: Market-1501 的 person_id 是 1-1501，不是连续的 0..n-1
        self.identity_list = list(self.identity_to_images.keys())

        print(
            f"Loaded Market-1501 {self.mode} set: "
            f"{self.num_identities} identities, {self.num_images} images"
        )

    def _load_image(self, image_idx: int) -> np.ndarray:
        """
        加载图像

        Args:
            image_idx: 图像在 image_list 中的索引

        Returns:
            image: (H, W, C) NumPy 数组, RGB, uint8
        """
        img_path, _, _ = self.image_list[image_idx]
        image = Image.open(img_path).convert("RGB")
        return np.array(image)

    def _get_single_item(self, index: int) -> Tuple[np.ndarray, int]:
        """
        获取单个图像及其身份标签

        Args:
            index: 索引

        Returns:
            (image, person_id): 图像和身份ID
        """
        img_path, person_id, _ = self.image_list[index % len(self.image_list)]
        image = Image.open(img_path).convert("RGB")
        return np.array(image), person_id

    def _get_positive_pair(self, person_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取正样本对 (同一人，不同摄像头)

        Args:
            person_id: 人员ID

        Returns:
            (image1, image2): 两张图像
        """
        image_indices = self.identity_to_images[person_id]

        if len(image_indices) < 2:
            # 如果只有一张图像，复制两次（数据增强会使它们不同）
            idx = image_indices[0]
            image = self._load_image(idx)
            return image.copy(), image.copy()

        # 尝试选择不同摄像头的图像
        idx1 = np.random.choice(image_indices)
        _, _, cam1 = self.image_list[idx1]

        # 找到不同摄像头的图像
        diff_cam_indices = [
            idx for idx in image_indices
            if self.image_list[idx][2] != cam1
        ]

        if len(diff_cam_indices) > 0:
            idx2 = np.random.choice(diff_cam_indices)
        else:
            # 如果只有一个摄像头，随机选择另一张
            idx2 = np.random.choice([idx for idx in image_indices if idx != idx1])

        image1 = self._load_image(idx1)
        image2 = self._load_image(idx2)

        return image1, image2

    def _get_negative_pair(self, person_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        获取负样本对 (不同人的图像)

        Args:
            person_id: 参考人员ID

        Returns:
            (image1, image2): 两张不同人的图像
        """
        # 选择当前人的一张图像
        image_indices = self.identity_to_images[person_id]
        idx1 = np.random.choice(image_indices)

        # 选择不同人的一张图像
        other_person_ids = [pid for pid in self.identity_to_images.keys() if pid != person_id]
        other_person_id = np.random.choice(other_person_ids)
        other_indices = self.identity_to_images[other_person_id]
        idx2 = np.random.choice(other_indices)

        image1 = self._load_image(idx1)
        image2 = self._load_image(idx2)

        return image1, image2

    def get_camera_ids(self) -> np.ndarray:
        """
        获取所有图像的摄像头ID

        Returns:
            camera_ids: shape (num_images,)
        """
        return np.array([cam for _, _, cam in self.image_list])

    def get_person_ids(self) -> np.ndarray:
        """
        获取所有图像的人员ID

        Returns:
            person_ids: shape (num_images,)
        """
        return np.array([pid for _, pid, _ in self.image_list])

    def __repr__(self) -> str:
        """字符串表示"""
        num_cameras = len(set(cam for _, _, cam in self.image_list))
        return (
            f"Market1501Dataset(\n"
            f"  mode={self.mode},\n"
            f"  num_identities={self.num_identities},\n"
            f"  num_images={self.num_images},\n"
            f"  num_cameras={num_cameras},\n"
            f"  avg_images_per_id={self.num_images / max(self.num_identities, 1):.2f}\n"
            f")"
        )


if __name__ == "__main__":
    # 测试数据集加载
    from pathlib import Path

    dataset_root = Path("data/market1501")

    if dataset_root.exists():
        print("Testing Market-1501 Dataset...")

        # 测试训练集
        train_dataset = Market1501Dataset(
            root=dataset_root,
            mode="train",
            return_pairs=True,
        )

        print(f"\n{train_dataset}")
        print(f"\nDataset stats: {train_dataset.get_stats()}")

        # 测试获取一个样本
        (img1, img2), label = train_dataset[0]
        print(f"\nSample pair:")
        print(f"  Image 1 shape: {img1.shape}")
        print(f"  Image 2 shape: {img2.shape}")
        print(f"  Label: {label} ({'same person' if label == 1 else 'different person'})")

        print("\n✅ Market-1501 Dataset test passed!")
    else:
        print(f"Dataset root not found: {dataset_root}")
        print("Skipping dataset test.")
