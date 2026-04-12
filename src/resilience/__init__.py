"""
弹性层 - 熔断、限流、重试等容错机制
RQA2025量化交易系统弹性层
"""

__version__ = "1.0.0"
__all__ = ['get_resilience_manager']

from .resilience_manager import ResilienceManager

_default_manager = None

def get_resilience_manager():
    """获取默认管理器实例"""
    global _default_manager
    if _default_manager is None:
        _default_manager = ResilienceManager()
    return _default_manager
