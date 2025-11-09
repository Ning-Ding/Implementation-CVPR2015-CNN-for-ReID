"""
CVPR 2015 CNN for Person Re-Identification
PyTorch Implementation - Educational Refactoring

基于 CVPR 2015 论文《An Improved Deep Learning Architecture for Person Re-Identification》
的 PyTorch 现代化实现，面向教学和学习。
"""

__version__ = "2.0.0"
__author__ = "Ning Ding (Original), Refactored by AI"
__license__ = "MIT"

from src import config, data, models, training, evaluation, utils

__all__ = [
    "config",
    "data",
    "models",
    "training",
    "evaluation",
    "utils",
]
