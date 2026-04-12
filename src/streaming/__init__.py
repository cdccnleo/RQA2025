"""
流处理层 - 实时数据流处理和分析
RQA2025量化交易系统流处理层
"""

__version__ = "1.0.0"
__all__ = ['get_streaming_manager']

from .streaming_manager import StreamingManager

_default_manager = None

def get_streaming_manager():
    """获取默认管理器实例"""
    global _default_manager
    if _default_manager is None:
        _default_manager = StreamingManager()
    return _default_manager
