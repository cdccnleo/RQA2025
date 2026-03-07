"""
工具模块（顶层）
"""

# 导入logger
try:
    from .logger import get_logger, Logger
except ImportError:
    try:
        from src.infrastructure.utils.logging import get_logger
        Logger = None
    except ImportError:
        def get_logger(name=None):
            import logging
            return logging.getLogger(name or __name__)
        Logger = None

__all__ = ['get_logger', 'Logger']
