"""
业务边界层 - 业务边界接口、服务间通信、路由管理
RQA2025量化交易系统业务边界层
"""

__version__ = "1.0.0"
__all__ = ['get_boundary_manager']

from .boundary_manager import BoundaryManager

_default_manager = None

def get_boundary_manager():
    global _default_manager
    if _default_manager is None:
        _default_manager = BoundaryManager()
    return _default_manager
