"""
异步处理器 - 异步任务队列、Celery集成、任务监控
RQA2025量化交易系统异步处理器
"""

__version__ = "1.0.0"
__all__ = ['get_async_manager']

from .async_manager import AsyncManager

_default_manager = None

def get_async_manager():
    global _default_manager
    if _default_manager is None:
        _default_manager = AsyncManager()
    return _default_manager
