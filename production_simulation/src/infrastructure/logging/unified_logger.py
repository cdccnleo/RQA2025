"""
统一日志器模块（别名模块）
提供向后兼容的导入路径

实际实现在 core/unified_logger.py 中
"""

try:
    from .core.unified_logger import (
        UnifiedLogger,
        get_unified_logger,
        get_logger
    )
except ImportError:
    # 提供基础实现
    import logging as _logging

    class UnifiedLogger:
        def __init__(self, name: str = "unified"):
            self.name = name
            self._logger = _logging.getLogger(name)
            if not self._logger.handlers:
                self._logger.addHandler(_logging.NullHandler())
            self._logger.propagate = False

        def log(self, level, message: str, **kwargs) -> None:
            if isinstance(level, str):
                level_value = getattr(_logging, level.upper(), _logging.INFO)
            elif isinstance(level, int):
                level_value = level
            else:
                level_value = _logging.INFO
            self._logger.log(level_value, message)

        def info(self, message: str, **kwargs) -> None:
            self.log("INFO", message, **kwargs)

        def warning(self, message: str, **kwargs) -> None:
            self.log("WARNING", message, **kwargs)

        def error(self, message: str, **kwargs) -> None:
            self.log("ERROR", message, **kwargs)

    def get_unified_logger(name: str = None):
        """获取统一日志器"""
        return UnifiedLogger(name or "unified")

    def get_logger(name: str = None):
        """获取日志器（别名）"""
        return get_unified_logger(name)

__all__ = ['UnifiedLogger', 'get_unified_logger', 'get_logger']

