"""配置管理模块异常定义

此模块定义了配置管理相关的所有异常类型，用于统一的错误处理。
"""

from enum import Enum, auto
from typing import Dict, Optional, Any
import time

class ConfigErrorType(Enum):
    """配置错误类型枚举"""
    LOAD_FAILURE = auto()      # 配置加载失败
    VALIDATION_FAILURE = auto() # 配置验证失败
    CACHE_FAILURE = auto()     # 缓存操作失败
    WATCHER_FAILURE = auto()   # 监控器失败
    ENCRYPTION_FAILURE = auto() # 加密操作失败
    PARSE_FAILURE = auto()     # 解析失败

class ConfigError(Exception):
    """配置管理基础异常类"""

    def __init__(
        self,
        error_type: ConfigErrorType,
        message: str,
        context: Optional[Dict[str, Any]] = None
    ):
        """初始化配置错误

        Args:
            error_type: 错误类型
            message: 错误消息
            context: 错误上下文信息
        """
        self.error_type = error_type
        self.message = message
        self.context = context or {}
        self.timestamp = time.time()
        super().__init__(message)

class ConfigLoadError(ConfigError):
    """配置加载错误"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigErrorType.LOAD_FAILURE, message, context)

class ConfigValidationError(ConfigError):
    """配置验证错误"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigErrorType.VALIDATION_FAILURE, message, context)

class ConfigCacheError(ConfigError):
    """配置缓存错误"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigErrorType.CACHE_FAILURE, message, context)

class ConfigWatcherError(ConfigError):
    """配置监控错误"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigErrorType.WATCHER_FAILURE, message, context)

class ConfigEncryptionError(ConfigError):
    """配置加密错误"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigErrorType.ENCRYPTION_FAILURE, message, context)

class ConfigParseError(ConfigError):
    """配置解析错误"""

    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigErrorType.PARSE_FAILURE, message, context)


class VersionError(ConfigError):
    """版本管理错误"""
    
    def __init__(self, message: str, version: str = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(ConfigErrorType.VALIDATION_FAILURE, message, context)
        self.version = version


class EventError(Exception):
    """基础事件异常类"""
    def __init__(self, message: str, event_type: str = None):
        super().__init__(message)
        self.event_type = event_type

class EventDeliveryError(EventError):
    """事件投递失败异常"""
    def __init__(self, message: str, event_type: str):
        super().__init__(f"事件投递失败: {message}", event_type)
        self.retryable = True

class EventProcessingError(EventError):
    """事件处理失败异常"""
    def __init__(self, message: str, event_type: str, handler: str):
        super().__init__(f"事件处理失败[{handler}]: {message}", event_type)
        self.handler = handler
