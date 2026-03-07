"""
日志器模块
提供向后兼容的导入路径
"""

try:
    from ...logging.core.unified_logger import (
        UnifiedLogger,
        get_unified_logger,
        get_logger
    )
except ImportError:
    import logging
    
    class UnifiedLogger:
        def __init__(self, name: str = None):
            self.logger = logging.getLogger(name or __name__)
    
    def get_unified_logger(name: str = None):
        """获取统一日志器"""
        return UnifiedLogger(name)
    
    def get_logger(name: str = None):
        """获取日志器（别名）"""
        return get_unified_logger(name)

__all__ = ['UnifiedLogger', 'get_unified_logger', 'get_logger']

