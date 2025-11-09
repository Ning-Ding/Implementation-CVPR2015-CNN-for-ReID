"""
Evaluation module
评估模块
"""

from .metrics import (
    compute_distance_matrix,
    compute_cmc,
    compute_map,
    evaluate_reid,
)

__all__ = [
    "compute_distance_matrix",
    "compute_cmc",
    "compute_map",
    "evaluate_reid",
]
