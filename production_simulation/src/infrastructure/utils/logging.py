"""
日志工具模块（别名模块）
提供向后兼容的导入路径

实际实现在 logging/core/unified_logger.py 中

注意：本文件是 logging.py，不是 logging 包
如果要导入 logging.logger 子模块，应使用完整路径
"""

try:
    from ..logging.core.unified_logger import (
        UnifiedLogger,
        get_unified_logger,
        get_logger
    )
except ImportError:
    # 提供基础实现
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

