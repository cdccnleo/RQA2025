"""
统一日志系统

提供结构化日志记录、多级别日志、上下文追踪等功能。

Author: RQA2025 Development Team
Date: 2026-02-13
"""

import logging
import json
import time
import uuid
import threading
from typing import Dict, Any, Optional, List, Callable
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field, asdict
from contextvars import ContextVar

# 上下文变量，用于存储请求ID
_request_id: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
_trace_id: ContextVar[Optional[str]] = ContextVar('trace_id', default=None)


class LogLevel(Enum):
    """日志级别"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """日志格式"""
    JSON = "json"
    TEXT = "text"


@dataclass
class LogContext:
    """日志上下文"""
    service: str = ""
    version: str = ""
    environment: str = ""
    instance_id: str = ""
    request_id: Optional[str] = None
    trace_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: str
    level: str
    message: str
    logger: str
    context: LogContext
    metadata: Dict[str, Any] = field(default_factory=dict)
    exception: Optional[str] = None
    source: Dict[str, Any] = field(default_factory=dict)


class UnifiedLogger:
    """
    统一日志记录器

    提供以下功能：
    1. 结构化日志记录（JSON格式）
    2. 上下文追踪（请求ID、链路ID）
    3. 多级别日志
    4. 日志过滤和采样
    5. 异步日志写入
    6. 日志轮转和归档

    Attributes:
        name: 日志器名称
        context: 日志上下文
        level: 日志级别
        format: 日志格式
    """

    def __init__(
        self,
        name: str,
        context: Optional[LogContext] = None,
        level: LogLevel = LogLevel.INFO,
        format: LogFormat = LogFormat.JSON,
        handlers: Optional[List[logging.Handler]] = None
    ):
        self.name = name
        self.context = context or LogContext()
        self.level = level
        self.format = format
        self._logger = logging.getLogger(name)
        self._logger.setLevel(self._get_logging_level(level))

        # 添加处理器
        if handlers:
            for handler in handlers:
                self._logger.addHandler(handler)
        else:
            self._setup_default_handlers()

        # 回调函数
        self._pre_log_callbacks: List[Callable[[LogEntry], None]] = []
        self._post_log_callbacks: List[Callable[[LogEntry], None]] = []

    def _get_logging_level(self, level: LogLevel) -> int:
        """获取logging模块的日志级别"""
        level_map = {
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL
        }
        return level_map.get(level, logging.INFO)

    def _setup_default_handlers(self):
        """设置默认处理器"""
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)

        if self.format == LogFormat.JSON:
            formatter = logging.Formatter('%(message)s')
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

        console_handler.setFormatter(formatter)
        self._logger.addHandler(console_handler)

    def _create_log_entry(
        self,
        level: LogLevel,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
        exception: Optional[Exception] = None
    ) -> LogEntry:
        """创建日志条目"""
        # 获取当前上下文
        current_request_id = _request_id.get()
        current_trace_id = _trace_id.get()

        context = LogContext(
            service=self.context.service,
            version=self.context.version,
            environment=self.context.environment,
            instance_id=self.context.instance_id,
            request_id=current_request_id or self.context.request_id,
            trace_id=current_trace_id or self.context.trace_id,
            user_id=self.context.user_id,
            session_id=self.context.session_id,
            extra={**self.context.extra, **(extra or {})}
        )

        # 获取异常信息
        exception_str = None
        if exception:
            import traceback
            exception_str = traceback.format_exc()

        # 获取调用源信息
        import inspect
        frame = inspect.currentframe().f_back.f_back
        source = {
            "file": frame.f_code.co_filename,
            "line": frame.f_lineno,
            "function": frame.f_code.co_name
        }

        return LogEntry(
            timestamp=datetime.utcnow().isoformat() + "Z",
            level=level.value,
            message=message,
            logger=self.name,
            context=context,
            metadata=extra or {},
            exception=exception_str,
            source=source
        )

    def _format_log_entry(self, entry: LogEntry) -> str:
        """格式化日志条目"""
        if self.format == LogFormat.JSON:
            return json.dumps(asdict(entry), ensure_ascii=False, default=str)
        else:
            # 文本格式
            context_str = " ".join([
                f"[{k}={v}]"
                for k, v in asdict(entry.context).items()
                if v is not None
            ])
            return f"{entry.timestamp} {entry.level} {entry.logger} {context_str} {entry.message}"

    def _log(self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
        """记录日志"""
        if self._get_logging_level(level) < self._logger.level:
            return

        entry = self._create_log_entry(level, message, extra, exception)

        # 执行前置回调
        for callback in self._pre_log_callbacks:
            try:
                callback(entry)
            except Exception as e:
                self._logger.error(f"Pre-log callback error: {e}")

        # 格式化并记录
        formatted = self._format_log_entry(entry)

        if level == LogLevel.DEBUG:
            self._logger.debug(formatted)
        elif level == LogLevel.INFO:
            self._logger.info(formatted)
        elif level == LogLevel.WARNING:
            self._logger.warning(formatted)
        elif level == LogLevel.ERROR:
            self._logger.error(formatted)
        elif level == LogLevel.CRITICAL:
            self._logger.critical(formatted)

        # 执行后置回调
        for callback in self._post_log_callbacks:
            try:
                callback(entry)
            except Exception as e:
                self._logger.error(f"Post-log callback error: {e}")

    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """记录DEBUG级别日志"""
        self._log(LogLevel.DEBUG, message, extra)

    def info(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """记录INFO级别日志"""
        self._log(LogLevel.INFO, message, extra)

    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None):
        """记录WARNING级别日志"""
        self._log(LogLevel.WARNING, message, extra)

    def error(self, message: str, extra: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
        """记录ERROR级别日志"""
        self._log(LogLevel.ERROR, message, extra, exception)

    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None, exception: Optional[Exception] = None):
        """记录CRITICAL级别日志"""
        self._log(LogLevel.CRITICAL, message, extra, exception)

    def log(self, level: LogLevel, message: str, extra: Optional[Dict[str, Any]] = None):
        """记录指定级别日志"""
        self._log(level, message, extra)

    def register_pre_log_callback(self, callback: Callable[[LogEntry], None]):
        """注册前置日志回调"""
        self._pre_log_callbacks.append(callback)

    def register_post_log_callback(self, callback: Callable[[LogEntry], None]):
        """注册后置日志回调"""
        self._post_log_callbacks.append(callback)

    def set_request_id(self, request_id: str):
        """设置请求ID"""
        _request_id.set(request_id)

    def set_trace_id(self, trace_id: str):
        """设置链路ID"""
        _trace_id.set(trace_id)

    def clear_context(self):
        """清除上下文"""
        _request_id.set(None)
        _trace_id.set(None)

    def with_context(self, **kwargs) -> 'UnifiedLogger':
        """创建带新上下文的日志器"""
        new_context = LogContext(
            service=kwargs.get('service', self.context.service),
            version=kwargs.get('version', self.context.version),
            environment=kwargs.get('environment', self.context.environment),
            instance_id=kwargs.get('instance_id', self.context.instance_id),
            request_id=kwargs.get('request_id', self.context.request_id),
            trace_id=kwargs.get('trace_id', self.context.trace_id),
            user_id=kwargs.get('user_id', self.context.user_id),
            session_id=kwargs.get('session_id', self.context.session_id),
            extra={**self.context.extra, **kwargs.get('extra', {})}
        )
        return UnifiedLogger(
            name=self.name,
            context=new_context,
            level=self.level,
            format=self.format
        )


class LoggerManager:
    """日志管理器"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loggers: Dict[str, UnifiedLogger] = {}
                    cls._instance._default_context = LogContext()
        return cls._instance

    def set_default_context(self, context: LogContext):
        """设置默认上下文"""
        self._default_context = context

    def get_logger(
        self,
        name: str,
        context: Optional[LogContext] = None,
        level: LogLevel = LogLevel.INFO
    ) -> UnifiedLogger:
        """获取日志器"""
        if name not in self._loggers:
            ctx = context or self._default_context
            self._loggers[name] = UnifiedLogger(name, ctx, level)
        return self._loggers[name]

    def remove_logger(self, name: str):
        """移除日志器"""
        if name in self._loggers:
            del self._loggers[name]


# 便捷函数
def get_logger(
    name: str,
    service: str = "",
    version: str = "",
    environment: str = "",
    level: LogLevel = LogLevel.INFO
) -> UnifiedLogger:
    """
    获取日志器

    Args:
        name: 日志器名称
        service: 服务名称
        version: 版本
        environment: 环境
        level: 日志级别

    Returns:
        UnifiedLogger: 日志器实例
    """
    context = LogContext(
        service=service,
        version=version,
        environment=environment
    )
    manager = LoggerManager()
    return manager.get_logger(name, context, level)


def set_request_id(request_id: str):
    """设置请求ID"""
    _request_id.set(request_id)


def set_trace_id(trace_id: str):
    """设置链路ID"""
    _trace_id.set(trace_id)


def get_request_id() -> Optional[str]:
    """获取请求ID"""
    return _request_id.get()


def get_trace_id() -> Optional[str]:
    """获取链路ID"""
    return _trace_id.get()


def generate_request_id() -> str:
    """生成请求ID"""
    return f"req-{uuid.uuid4().hex[:12]}"


def generate_trace_id() -> str:
    """生成链路ID"""
    return f"trace-{uuid.uuid4().hex[:16]}"


# 装饰器
def log_execution_time(logger: Optional[UnifiedLogger] = None, level: LogLevel = LogLevel.INFO):
    """记录执行时间装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time
                log.log(level, f"Function {func.__name__} executed in {execution_time:.3f}s", {
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "status": "success"
                })
                return result
            except Exception as e:
                execution_time = time.time() - start_time
                log.error(f"Function {func.__name__} failed after {execution_time:.3f}s", {
                    "function": func.__name__,
                    "execution_time": execution_time,
                    "status": "failed",
                    "error": str(e)
                }, e)
                raise

        return wrapper
    return decorator


def log_api_call(logger: Optional[UnifiedLogger] = None):
    """记录API调用装饰器"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            log = logger or get_logger(func.__module__)
            request_id = generate_request_id()
            set_request_id(request_id)

            log.info(f"API call started: {func.__name__}", {
                "request_id": request_id,
                "function": func.__name__,
                "args": str(args),
                "kwargs": str(kwargs)
            })

            try:
                result = func(*args, **kwargs)
                log.info(f"API call completed: {func.__name__}", {
                    "request_id": request_id,
                    "function": func.__name__,
                    "status": "success"
                })
                return result
            except Exception as e:
                log.error(f"API call failed: {func.__name__}", {
                    "request_id": request_id,
                    "function": func.__name__,
                    "status": "failed"
                }, e)
                raise
            finally:
                set_request_id(None)

        return wrapper
    return decorator


__all__ = [
    'UnifiedLogger',
    'LoggerManager',
    'LogContext',
    'LogEntry',
    'LogLevel',
    'LogFormat',
    'get_logger',
    'set_request_id',
    'set_trace_id',
    'get_request_id',
    'get_trace_id',
    'generate_request_id',
    'generate_trace_id',
    'log_execution_time',
    'log_api_call'
]