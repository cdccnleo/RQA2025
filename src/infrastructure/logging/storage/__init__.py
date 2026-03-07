
from .base import BaseStorage, MemoryStorage
"""
基础设施层 - 日志存储模块

提供各种日志存储的后端实现，包括文件存储、数据库存储、内存存储等。
"""

__all__ = [
    'BaseStorage',
    'MemoryStorage'
]
