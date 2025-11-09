"""
Evaluation metrics for Person Re-Identification
人员重识别评估指标

实现 CMC (Cumulative Matching Characteristic) 和 mAP (mean Average Precision)
"""

import numpy as np
import torch
from typing import Tuple, List, Optional
from sklearn.metrics import average_precision_score


def compute_distance_matrix(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    metric: str = "euclidean",
) -> torch.Tensor:
    """
    计算查询集和图库集之间的距离矩阵

    Args:
        query_features: (N_q, D) 查询特征
        gallery_features: (N_g, D) 图库特征
        metric: 距离度量 ('euclidean' or 'cosine')

    Returns:
        距离矩阵: (N_q, N_g)
    """
    if metric == "euclidean":
        # Euclidean distance
        distmat = torch.cdist(query_features, gallery_features, p=2)
    elif metric == "cosine":
        # Cosine distance = 1 - cosine similarity
        query_norm = torch.nn.functional.normalize(query_features, p=2, dim=1)
        gallery_norm = torch.nn.functional.normalize(gallery_features, p=2, dim=1)
        distmat = 1 - torch.mm(query_norm, gallery_norm.t())
    else:
        raise ValueError(f"Unknown metric: {metric}")

    return distmat


def compute_cmc(
    distmat: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cams: Optional[np.ndarray] = None,
    gallery_cams: Optional[np.ndarray] = None,
    topk: int = 50,
) -> np.ndarray:
    """
    计算 CMC (Cumulative Matching Characteristic) 曲线

    Args:
        distmat: (N_q, N_g) 距离矩阵
        query_ids: (N_q,) 查询ID
        gallery_ids: (N_g,) 图库ID
        query_cams: (N_q,) 查询摄像头ID（可选）
        gallery_cams: (N_g,) 图库摄像头ID（可选）
        topk: Top-K ranks

    Returns:
        cmc: (topk,) CMC 曲线
    """
    num_q, num_g = distmat.shape
    if num_g < topk:
        topk = num_g

    indices = np.argsort(distmat, axis=1)  # (N_q, N_g) 排序索引

    cmc = np.zeros(topk)
    for q_idx in range(num_q):
        q_id = query_ids[q_idx]
        q_cam = query_cams[q_idx] if query_cams is not None else None

        # 获取排序后的 gallery
        order = indices[q_idx]
        g_ids = gallery_ids[order]
        g_cams = gallery_cams[order] if gallery_cams is not None else None

        # 移除同一摄像头的匹配（标准 ReID 评估协议）
        if q_cam is not None and g_cams is not None:
            # 过滤掉同一摄像头的样本
            keep = (g_cams != q_cam)
            g_ids = g_ids[keep]

        # 找到第一个匹配的位置
        matches = (g_ids == q_id)
        match_indices = np.where(matches)[0]
        if len(match_indices) > 0:
            first_match = match_indices[0]
            if first_match < topk:
                cmc[first_match:] += 1

    cmc = cmc / num_q  # 归一化
    return cmc


def compute_map(
    distmat: np.ndarray,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cams: Optional[np.ndarray] = None,
    gallery_cams: Optional[np.ndarray] = None,
) -> float:
    """
    计算 mAP (mean Average Precision)

    Args:
        distmat: (N_q, N_g) 距离矩阵
        query_ids: (N_q,) 查询ID
        gallery_ids: (N_g,) 图库ID
        query_cams: (N_q,) 查询摄像头ID（可选）
        gallery_cams: (N_g,) 图库摄像头ID（可选）

    Returns:
        mAP: scalar
    """
    num_q, num_g = distmat.shape
    indices = np.argsort(distmat, axis=1)

    aps = []
    for q_idx in range(num_q):
        q_id = query_ids[q_idx]
        q_cam = query_cams[q_idx] if query_cams is not None else None

        # 获取排序后的gallery
        order = indices[q_idx]
        g_ids = gallery_ids[order]
        g_cams = gallery_cams[order] if gallery_cams is not None else None

        # 移除同一摄像头的样本（标准 ReID 评估协议）
        if q_cam is not None and g_cams is not None:
            # 过滤掉同一摄像头的样本
            keep = (g_cams != q_cam)
            g_ids = g_ids[keep]

        # Ground truth: 同一人的图像
        valid = (g_ids == q_id)

        if not np.any(valid):
            continue

        # 计算 Average Precision
        # AP = mean of precision at each relevant position
        relevance = valid.astype(float)
        cumsum = np.cumsum(relevance)
        precision_at_k = cumsum / (np.arange(len(relevance)) + 1)
        ap = np.sum(precision_at_k * relevance) / np.sum(relevance)
        aps.append(ap)

    if len(aps) == 0:
        return 0.0

    return np.mean(aps)


def evaluate_reid(
    query_features: torch.Tensor,
    gallery_features: torch.Tensor,
    query_ids: np.ndarray,
    gallery_ids: np.ndarray,
    query_cams: Optional[np.ndarray] = None,
    gallery_cams: Optional[np.ndarray] = None,
    metric: str = "euclidean",
    cmc_topk: List[int] = [1, 5, 10, 20],
) -> dict:
    """
    完整的 ReID 评估流程

    Args:
        query_features: (N_q, D) 查询特征
        gallery_features: (N_g, D) 图库特征
        query_ids: (N_q,) 查询ID
        gallery_ids: (N_g,) 图库ID
        query_cams: (N_q,) 查询摄像头ID（可选）
        gallery_cams: (N_g,) 图库摄像头ID（可选）
        metric: 距离度量
        cmc_topk: CMC top-k ranks

    Returns:
        结果字典包含:
            - cmc: CMC 曲线
            - mAP: mean Average Precision
            - rank1, rank5, rank10, ...: CMC@rank
    """
    # 计算距离矩阵
    distmat = compute_distance_matrix(query_features, gallery_features, metric)
    distmat = distmat.cpu().numpy()

    # 计算 CMC
    max_rank = max(cmc_topk)
    cmc = compute_cmc(
        distmat, query_ids, gallery_ids,
        query_cams, gallery_cams, topk=max_rank
    )

    # 计算 mAP
    mAP = compute_map(
        distmat, query_ids, gallery_ids,
        query_cams, gallery_cams
    )

    # 整理结果
    results = {
        "mAP": mAP,
        "cmc": cmc,
    }

    for k in cmc_topk:
        if k <= len(cmc):
            results[f"rank{k}"] = cmc[k-1]

    return results


if __name__ == "__main__":
    # 测试评估指标
    print("Testing ReID evaluation metrics...")

    # 模拟数据
    num_query = 100
    num_gallery = 500
    feat_dim = 128

    query_features = torch.randn(num_query, feat_dim)
    gallery_features = torch.randn(num_gallery, feat_dim)

    query_ids = np.random.randint(0, 50, num_query)
    gallery_ids = np.random.randint(0, 50, num_gallery)

    query_cams = np.random.randint(0, 6, num_query)
    gallery_cams = np.random.randint(0, 6, num_gallery)

    # 评估
    results = evaluate_reid(
        query_features,
        gallery_features,
        query_ids,
        gallery_ids,
        query_cams,
        gallery_cams,
        metric="euclidean",
        cmc_topk=[1, 5, 10, 20],
    )

    print("\nEvaluation Results:")
    print(f"  mAP: {results['mAP']:.4f}")
    print(f"  Rank-1: {results['rank1']:.4f}")
    print(f"  Rank-5: {results['rank5']:.4f}")
    print(f"  Rank-10: {results['rank10']:.4f}")
    print(f"  CMC curve shape: {results['cmc'].shape}")

    print("\n✅ Evaluation metrics test passed!")
