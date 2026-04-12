"""
分布式协调器 - 分布式锁、服务发现、配置同步
RQA2025量化交易系统分布式协调器
"""

__version__ = "1.0.0"
__all__ = ['get_distributed_manager']

from .distributed_manager import DistributedManager

_default_manager = None

def get_distributed_manager():
    global _default_manager
    if _default_manager is None:
        _default_manager = DistributedManager()
    return _default_manager
