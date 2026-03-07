"""
日志核心接口模块

提供统一的日志接口定义
"""

import logging
import threading
from typing import Optional, Dict, Any, Union
from enum import Enum


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"
    
    @property
    def level(self) -> int:
        """获取logging级别值"""
        return getattr(logging, self.value)


class LogCategory(Enum):
    """日志类别"""
    SYSTEM = "system"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    ERROR = "error"
    AUDIT = "audit"
    DEBUG = "debug"
    DATABASE = "database"
    TRADING = "trading"
    RISK = "risk"
    GENERAL = "general"


class LogFormat(Enum):
    """日志格式"""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"
    SIMPLE = "simple"
    DETAILED = "detailed"


class ILogFormatter:
    """日志格式化器接口"""
    
    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return formatter.format(record)


class ILogHandler:
    """日志处理器接口"""
    
    def __init__(self):
        self.handler = logging.StreamHandler()
    
    def emit(self, record: logging.LogRecord):
        """发送日志"""
        self.handler.emit(record)
    
    def setFormatter(self, formatter):
        """设置格式化器"""
        self.handler.setFormatter(formatter)


def get_logger_pool(*_, **__):
    """
    获取 Logger 池单例实例

    该接口用于为监控与运维组件提供统一的 Logger 池访问入口，保持与架构设计一致，
    避免上层组件直接依赖具体实现。
    """
    try:
        from .logger_pool import LoggerPool
    except ImportError as exc:
        raise ImportError("Logger池实现不可用，请检查基础设施日志模块。") from exc
    
    # LoggerPool 本身为单例实现，直接返回即可。
    return LoggerPool.get_instance()


def get_logger(name: Optional[str] = None, **kwargs) -> logging.Logger:
    """
    获取日志记录器（通过LoggerPool，支持复用）
    
    Args:
        name: 日志记录器名称，默认为调用模块名
        **kwargs: 额外配置参数
    
    Returns:
        logging.Logger: 配置好的日志记录器
    """
    if name is None:
        name = __name__
    
    try:
        # 尝试从LoggerPool获取
        pool = get_logger_pool()
        logger_instance = pool.get_logger(name, **kwargs)
        
        # 如果返回的是UnifiedLogger实例，返回其内部的logging.Logger
        if hasattr(logger_instance, 'logger'):
            logger = logger_instance.logger
        # 如果是logging.Logger，直接返回
        elif isinstance(logger_instance, logging.Logger):
            logger = logger_instance
        # 降级：直接使用logging.getLogger
        else:
            logger = logging.getLogger(name)
    except Exception:
        # 降级：如果LoggerPool不可用，直接使用logging.getLogger
        logger = logging.getLogger(name)
    
    # 如果未配置，设置默认配置
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    return logger


class ILogger:
    """日志接口（向后兼容）"""
    
    def __init__(self, name: Optional[str] = None):
        self.logger = get_logger(name)
    
    def debug(self, message: str, **kwargs):
        """调试日志"""
        self.logger.debug(message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """信息日志"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """警告日志"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """错误日志"""
        self.logger.error(message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """严重错误日志"""
        self.logger.critical(message, **kwargs)


class BaseLogger(ILogger):
    """基础日志器（别名）"""

    def __init__(
        self,
        name: Optional[str] = None,
        level: Optional[Union[LogLevel, int, str]] = None,
    ):
        super().__init__(name)
        self.name = name or self.__class__.__name__
        self.logger = get_logger(self.name)
        self._logger = self.logger
        self._lock = threading.Lock()
        self.level: LogLevel = LogLevel.INFO
        self.set_level(level or LogLevel.INFO)

    def set_level(self, level: Union[LogLevel, int, str]) -> None:
        """设置日志级别"""
        if isinstance(level, LogLevel):
            self.level = level
            numeric_level = level.level
        elif isinstance(level, int):
            numeric_level = level
            for member in LogLevel:
                if member.level == level:
                    self.level = member
                    break
        else:  # str
            level_str = str(level).upper()
            numeric_level = getattr(logging, level_str, logging.INFO)
            self.level = LogLevel[level_str] if level_str in LogLevel.__members__ else LogLevel.INFO
        self.logger.setLevel(numeric_level)
        if hasattr(self, "_logger"):
            self._logger.setLevel(numeric_level)

    def get_level(self) -> LogLevel:
        """获取当前日志级别"""
        return self.level

    def _should_log(self, level: LogLevel) -> bool:
        hierarchy = {
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.CRITICAL: 50,
        }
        return hierarchy.get(level, 0) >= hierarchy.get(self.level, 0)

    def _format_message(self, message: str, kwargs: Dict[str, Any]) -> str:
        if not kwargs:
            return message
        try:
            extra = " ".join(f"{k}={v}" for k, v in kwargs.items())
            return f"{message} | {extra}"
        except Exception:
            return message

    def log(self, level: Union[str, int, LogLevel], message: str, **kwargs) -> None:
        """记录日志"""
        with self._lock:
            try:
                if isinstance(level, str):
                    level_str = level.upper()
                    if level_str not in LogLevel.__members__:
                        raise ValueError(f"Unsupported log level: {level}")
                    level_enum = LogLevel[level_str]
                elif isinstance(level, LogLevel):
                    level_enum = level
                else:  # int
                    level_enum = self._normalize_numeric_level(level)
            except ValueError as exc:
                raise Exception(f"日志记录失败: {exc}") from exc
            if self._should_log(level_enum):
                try:
                    formatted = self._format_message(message, kwargs)
                except Exception as exc:
                    raise Exception(f"日志记录失败: {exc}") from exc
                target_logger = getattr(self, "_logger", self.logger)
                method_name = level_enum.value.lower()
                log_method = getattr(target_logger, method_name, None) if not isinstance(level, int) else None
                try:
                    if callable(log_method):
                        log_method(formatted)
                    else:
                        target_logger.log(self._resolve_numeric_level(level_enum), formatted)
                except Exception as exc:
                    raise Exception(f"日志记录失败: {exc}") from exc

    def _normalize_numeric_level(self, value: int) -> LogLevel:
        for member in LogLevel:
            if member.level == value:
                return member
        return LogLevel.INFO

    def _resolve_numeric_level(self, level: LogLevel) -> int:
        return getattr(logging, level.value)

    def debug(self, message: str, **kwargs) -> None:
        self.log(LogLevel.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        self.log(LogLevel.INFO, message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        self.log(LogLevel.WARNING, message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        self.log(LogLevel.ERROR, message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        self.log(LogLevel.CRITICAL, message, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "level": self.level.value,
            "type": self.__class__.__name__,
        }


class UnifiedLogger(ILogger):
    """统一日志器"""
    
    def __init__(self, name: Optional[str] = None, category: Optional[str] = None):
        super().__init__(name)
        self.name = name or "UnifiedLogger"
        self.level = LogLevel.INFO
        self.category = category or "system"
    
    def log_with_category(self, level: str, message: str, category: Optional[str] = None, **kwargs):
        """带类别的日志"""
        cat = category or self.category
        self.logger.log(getattr(logging, level.upper()), f"[{cat}] {message}", **kwargs)


class PerformanceLogger(BaseLogger):
    """性能日志器"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "PerformanceLogger", LogLevel.INFO)

    def log_performance(self, operation: str, duration: float, **kwargs):
        """记录性能日志"""
        self.info(f"[PERF] {operation}: {duration:.4f}s", **kwargs)


class BusinessLogger(BaseLogger):
    """业务日志器"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "BusinessLogger", LogLevel.INFO)


class AuditLogger(BaseLogger):
    """审计日志器"""

    def __init__(self, name: Optional[str] = None):
        super().__init__(name or "AuditLogger", LogLevel.INFO)


__all__ = [
    'get_logger', 
    'get_logger_pool',
    'ILogger', 
    'LogLevel', 
    'BaseLogger',
    'UnifiedLogger',
    'BusinessLogger',
    'AuditLogger',
    'PerformanceLogger',
    'ILogFormatter',
    'ILogHandler',
    'LogCategory',
    'LogFormat'
]
