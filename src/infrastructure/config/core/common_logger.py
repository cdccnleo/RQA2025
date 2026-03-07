"""
common_logger 模块

提供 common_logger 相关功能和接口。
"""

import json
import logging

import threading
import time

from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
#!/usr/bin/env python3
"""
通用日志工具

提供标准化的日志记录功能，支持结构化日志、性能监控日志和操作跟踪
减少日志代码的重复，提高日志的一致性和可维护性
"""


class LogLevel(Enum):
    """日志级别枚举"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(Enum):
    """日志格式枚举"""
    TEXT = "text"
    JSON = "json"
    STRUCTURED = "structured"


class OperationType(Enum):
    """操作类型枚举"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    UPDATE = "update"
    QUERY = "query"
    VALIDATE = "validate"
    MONITOR = "monitor"
    MAINTAIN = "maintain"


class LogContext:
    """日志上下文"""

    def __init__(self,
                 component: str = "",
                 operation: str = "",
                 operation_type: Optional[OperationType] = None,
                 user_id: Optional[str] = None,
                 session_id: Optional[str] = None,
                 request_id: Optional[str] = None,
                 parameters: Optional[Dict[str, Any]] = None):
        """初始化日志上下文

        Args:
            component: 组件名称
            operation: 操作名称
            operation_type: 操作类型
            user_id: 用户ID
            session_id: 会话ID
            request_id: 请求ID
            parameters: 操作参数
        """
        self.component = component
        self.operation = operation
        self.operation_type = operation_type
        self.user_id = user_id
        self.session_id = session_id
        self.request_id = request_id
        self.parameters = parameters or {}
        self.start_time = time.time()
        self.end_time = None
        self.duration = None

    def complete(self, success: bool = True, result: Any = None, error: Optional[str] = None):
        """完成操作，记录结束时间

        Args:
            success: 操作是否成功
            result: 操作结果
            error: 错误信息
        """
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.success = success
        self.result = result
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'component': self.component,
            'operation': self.operation,
            'operation_type': self.operation_type.value if self.operation_type else None,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'request_id': self.request_id,
            'parameters': self.parameters,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'success': getattr(self, 'success', None),
            'result': getattr(self, 'result', None),
            'error': getattr(self, 'error', None)
        }


class StructuredLogger:
    """结构化日志记录器

    提供标准化的日志记录功能，支持多种日志格式和上下文跟踪
    """

    def __init__(self,
                 name: str,
                 level: LogLevel = LogLevel.INFO,
                 format_type: LogFormat = LogFormat.STRUCTURED,
                 include_timestamp: bool = True,
                 include_thread_id: bool = True):
        """初始化结构化日志记录器

        Args:
            name: 日志记录器名称
            level: 日志级别
            format_type: 日志格式
            include_timestamp: 是否包含时间戳
            include_thread_id: 是否包含线程ID
        """
        self.name = name
        self.level = level
        self.format_type = format_type
        self.include_timestamp = include_timestamp
        self.include_thread_id = include_thread_id

        # 获取或创建日志记录器
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.value))

        # 如果没有处理器，添加一个
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = self._get_formatter()
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)

    def _get_formatter(self) -> logging.Formatter:
        """获取格式化器"""
        if self.format_type == LogFormat.JSON:
            return JSONFormatter(include_timestamp=self.include_timestamp,
                                 include_thread_id=self.include_thread_id)
        elif self.format_type == LogFormat.STRUCTURED:
            return StructuredFormatter(include_timestamp=self.include_timestamp,
                                       include_thread_id=self.include_thread_id)
        else:
            return TextFormatter(include_timestamp=self.include_timestamp,
                                 include_thread_id=self.include_thread_id)

    def _log(self, level: LogLevel, message: str, context: Optional[LogContext] = None,
             extra: Optional[Dict[str, Any]] = None):
        """内部日志记录方法

        Args:
            level: 日志级别
            message: 日志消息
            context: 日志上下文
            extra: 额外信息
        """
        log_method = getattr(self.logger, level.value.lower())

        # 准备日志数据
        log_data = {
            'message': message,
            'level': level.value,
            'logger': self.name
        }

        if context:
            log_data['context'] = context.to_dict()

        if extra:
            log_data.update(extra)

        if self.include_thread_id:
            log_data['thread_id'] = threading.get_ident()

        if self.include_timestamp:
            log_data['timestamp'] = datetime.now().isoformat()

        log_method(message, extra={'structured_data': log_data})

    def debug(self, message: str, context: Optional[LogContext] = None, **extra):
        """记录调试日志"""
        self._log(LogLevel.DEBUG, message, context, extra)

    def info(self, message: str, context: Optional[LogContext] = None, **extra):
        """记录信息日志"""
        self._log(LogLevel.INFO, message, context, extra)

    def warning(self, message: str, context: Optional[LogContext] = None, **extra):
        """记录警告日志"""
        self._log(LogLevel.WARNING, message, context, extra)

    def error(self, message: str, context: Optional[LogContext] = None, **extra):
        """记录错误日志"""
        self._log(LogLevel.ERROR, message, context, extra)

    def critical(self, message: str, context: Optional[LogContext] = None, **extra):
        """记录严重错误日志"""
        self._log(LogLevel.CRITICAL, message, context, extra)

    def log_operation(self, context: LogContext, success: bool = True,
                      result: Any = None, error: Optional[str] = None):
        """记录操作日志

        Args:
            context: 操作上下文
            success: 操作是否成功
            result: 操作结果
            error: 错误信息
        """
        context.complete(success=success, result=result, error=error)

        level = LogLevel.INFO if success else LogLevel.ERROR
        message = f"操作{'成功' if success else '失败'}: {context.operation}"

        if error:
            message += f" - {error}"

        self._log(level, message, context)

    def log_performance(self, operation: str, duration: float,
                        threshold: Optional[float] = None):
        """记录性能日志

        Args:
            operation: 操作名称
            duration: 执行时间(秒)
            threshold: 性能阈值(秒)
        """
        context = LogContext(component=self.name, operation=operation)
        context.complete(success=True, result={'duration': duration})

        level = LogLevel.WARNING if threshold and duration > threshold else LogLevel.DEBUG
        message = f"性能监控: {operation} 耗时 {duration:.3f}秒"

        if threshold and duration > threshold:
            message += f" (超过阈值 {threshold:.3f}秒)"

        extra = {
            'performance': {
                'operation': operation,
                'duration': duration,
                'threshold': threshold,
                'exceeded': threshold and duration > threshold
            }
        }

        self._log(level, message, context, extra)


class TextFormatter(logging.Formatter):
    """文本格式化器"""

    def __init__(self, include_timestamp: bool = True, include_thread_id: bool = True):
        """初始化文本格式化器"""
        fmt_parts = []
        if include_timestamp:
            fmt_parts.append('%(asctime)s')
        if include_thread_id:
            fmt_parts.append('Thread-%(thread)d')
        fmt_parts.extend(['%(levelname)s', '%(name)s', '%(message)s'])

        fmt = ' | '.join(fmt_parts)
        super().__init__(fmt)


class JSONFormatter(logging.Formatter):
    """JSON格式化器"""

    def __init__(self, include_timestamp: bool = True, include_thread_id: bool = True):
        """初始化JSON格式化器"""
        self.include_timestamp = include_timestamp
        self.include_thread_id = include_thread_id

    def format(self, record):
        """格式化日志记录"""
        log_data = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage()
        }

        if self.include_thread_id:
            log_data['thread_id'] = record.thread

        # 添加结构化数据
        if hasattr(record, 'structured_data'):
            log_data.update(record.structured_data)

        return json.dumps(log_data, ensure_ascii=False)


class StructuredFormatter(logging.Formatter):
    """结构化格式化器"""

    def __init__(self, include_timestamp: bool = True, include_thread_id: bool = True):
        """初始化结构化格式化器"""
        self.include_timestamp = include_timestamp
        self.include_thread_id = include_thread_id

    def format(self, record):
        """格式化日志记录"""
        parts = []

        if self.include_timestamp:
            parts.append(datetime.fromtimestamp(record.created).isoformat())

        if self.include_thread_id:
            parts.append(f"Thread-{record.thread}")

        parts.extend([
            record.levelname,
            record.name,
            record.getMessage()
        ])

        # 添加结构化数据
        if hasattr(record, 'structured_data'):
            structured = record.structured_data
            if 'context' in structured and structured['context']:
                context = structured['context']
                if context.get('component'):
                    parts.append(f"[{context['component']}]")
                if context.get('operation'):
                    parts.append(f"{context['operation']}")
                if context.get('duration'):
                    parts.append(f"({context['duration']:.3f}s)")

        return ' | '.join(parts)

# ==================== 便捷函数 ====================


def get_logger(name: str, level: LogLevel = LogLevel.INFO) -> StructuredLogger:
    """获取结构化日志记录器

    Args:
        name: 日志记录器名称
        level: 日志级别

    Returns:
        结构化日志记录器实例
    """
    return StructuredLogger(name=name, level=level)


def create_operation_context(component: str, operation: str,
                             operation_type: Optional[OperationType] = None) -> LogContext:
    """创建操作日志上下文

    Args:
        component: 组件名称
        operation: 操作名称
        operation_type: 操作类型

    Returns:
        日志上下文实例
    """
    return LogContext(component=component, operation=operation, operation_type=operation_type)

# ==================== 全局日志记录器 ====================


# 默认日志记录器
default_logger = StructuredLogger("infrastructure.config", LogLevel.INFO)

# 性能日志记录器
performance_logger = StructuredLogger("infrastructure.config.performance", LogLevel.DEBUG)

# 错误日志记录器
error_logger = StructuredLogger("infrastructure.config.error", LogLevel.ERROR)




