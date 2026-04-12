"""
移动端层 - 移动端API、推送通知、数据优化
RQA2025量化交易系统移动端层
"""

__version__ = "1.0.0"
__all__ = ['get_mobile_manager']

from .mobile_manager import MobileManager

_default_manager = None

def get_mobile_manager():
    global _default_manager
    if _default_manager is None:
        _default_manager = MobileManager()
    return _default_manager
