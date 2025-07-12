"""
集中管理日志工具函数，避免循环导入问题
"""

import logging
from typing import Optional

def get_logger(name: str) -> logging.Logger:
    """获取配置好的日志记录器

    Args:
        name: 日志器名称，通常使用__name__

    Returns:
        配置好的日志记录器实例
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # 避免重复添加handler
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger

def configure_logging(config: Optional[dict] = None) -> None:
    """全局配置日志系统

    Args:
        config: 日志配置字典，如果为None则使用默认配置
    """
    if config is None:
        config = {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }

    logging.basicConfig(**config)
