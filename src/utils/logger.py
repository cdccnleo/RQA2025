"""日志记录工具

提供统一的日志记录功能，用于项目中的日志管理。

函数:
- get_logger: 获取配置好的日志记录器
"""

import logging
from .logging_utils import setup_logging

def get_logger(name='rqa'):
    """获取配置好的日志记录器

    参数:
        name: 日志记录器名称

    返回:
        配置好的日志记录器
    """
    # 确保日志系统已初始化
    setup_logging()
    
    return logging.getLogger(name)

__all__ = ['get_logger']
