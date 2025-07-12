"""
配置错误处理模块
版本更新记录：
2024-04-02 v3.6.0 - 错误模块更新
    - 统一错误类型定义
    - 增强错误处理器
    - 完善异常上下文
2024-04-03 v3.6.1 - 异常类补充
    - 添加事件相关异常导出
"""

from .error_handler import ConfigErrorHandler
from .exceptions import (
    ConfigError,
    ConfigLoadError,
    ConfigValidationError,
    ConfigCacheError,
    ConfigWatcherError,
    VersionError,
    EventError,
    EventDeliveryError,
    EventProcessingError
)

__all__ = [
    'ConfigErrorHandler',
    'ConfigError',
    'ConfigLoadError',
    'ConfigValidationError',
    'ConfigCacheError',
    'ConfigWatcherError',
    'VersionError',
    'EventError',
    'EventDeliveryError',
    'EventProcessingError'
]
