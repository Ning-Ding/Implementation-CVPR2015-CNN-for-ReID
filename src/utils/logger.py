"""
Logging utilities
日志工具
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from rich.logging import RichHandler
from rich.console import Console


def setup_logger(
    name: str = "reid",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    use_rich: bool = True,
) -> logging.Logger:
    """
    设置日志记录器

    Args:
        name: Logger 名称
        log_file: 日志文件路径
        level: 日志级别
        use_rich: 是否使用 Rich 格式化

    Returns:
        logger: 配置好的 logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # 清除现有 handlers
    logger.handlers.clear()

    # Console handler
    if use_rich:
        console_handler = RichHandler(
            rich_tracebacks=True,
            markup=True,
            show_time=True,
            show_path=False,
        )
    else:
        console_handler = logging.StreamHandler(sys.stdout)

    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(level)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# 全局 logger 实例
_default_logger = None


def get_logger(name: str = "reid") -> logging.Logger:
    """获取全局 logger"""
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_logger(name)
    return _default_logger
