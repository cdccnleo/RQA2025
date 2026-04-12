"""
工具层 - 通用工具函数和辅助功能
RQA2025量化交易系统工具层
"""

__version__ = "1.0.0"
__all__ = ['get_utils_manager']

from .utils_manager import UtilsManager

_default_manager = None

def get_utils_manager():
    """获取默认管理器实例"""
    global _default_manager
    if _default_manager is None:
        _default_manager = UtilsManager()
    return _default_manager
