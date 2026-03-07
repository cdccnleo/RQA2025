"""
common_exception_handler 模块

提供 common_exception_handler 相关功能和接口。
"""

import logging
import traceback

import functools
import threading
import time

from enum import Enum
from typing import Any, Callable, Optional, Dict, List
#!/usr/bin/env python3
"""
通用异常处理工具

提供统一的异常处理机制，减少代码重复，提高错误处理的一致性
支持缓存管理和配置管理等多个模块的异常处理需求
"""

logger = logging.getLogger(__name__)


class ExceptionHandlingStrategy(Enum):
    """异常处理策略"""
    LOG_AND_RETURN_DEFAULT = "log_and_return_default"
    LOG_AND_RERAISE = "log_and_reraise"
    SILENT_RETURN_DEFAULT = "silent_return_default"
    COLLECT_AND_RETURN = "collect_and_return"


class LogLevel(Enum):
    """日志级别"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class ExceptionContext:
    """异常上下文信息"""

    def __init__(self,
                 operation: str = "",
                 component: str = "",
                 parameters: Optional[Dict[str, Any]] = None,
                 start_time: Optional[float] = None,
                 user_id: str = "",
                 session_id: str = "",
                 error_message: str = ""):
        """初始化异常上下文

        Args:
            operation: 操作名称
            component: 组件名称
            parameters: 操作参数
            start_time: 操作开始时间
            user_id: 用户ID
            session_id: 会话ID
        """
        # 确保参数类型正确
        self.operation = operation if operation is not None else ""
        self.component = component if component is not None else ""
        self.parameters = parameters or {}
        self.start_time = start_time or time.time()
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        self.user_id = user_id if user_id is not None else ""
        self.session_id = session_id if session_id is not None else ""
        self.error_message = error_message if error_message is not None else ""

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'operation': self.operation,
            'component': self.component,
            'parameters': self.parameters,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'duration': self.duration,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'error_message': self.error_message
        }


def handle_exceptions(strategy: ExceptionHandlingStrategy = ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
                      default_return: Any = None,
                      log_level: LogLevel = LogLevel.ERROR,
                      include_context: bool = True,
                      max_retries: int = 0,
                      retry_delay: float = 0.1):
    """
    通用异常处理装饰器

    提供灵活的异常处理策略，支持重试机制和详细的上下文信息记录。

    Args:
        strategy: 异常处理策略
        default_return: 异常时的默认返回值
        log_level: 日志级别
        include_context: 是否包含上下文信息
        max_retries: 最大重试次数
        retry_delay: 重试间隔(秒)

    Returns:
        装饰器函数

    Example:
        @handle_exceptions(
            strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
            default_return=None,
            log_level=LogLevel.WARNING
        )
        def risky_operation(self, param1, param2):
            # 可能抛出异常的操作
            return some_risky_call(param1, param2)
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # 准备上下文信息
            context = None
            if include_context:
                component = args[0].__class__.__name__ if args else "Unknown"
                context = ExceptionContext(
                    operation=func.__name__,
                    component=component,
                    parameters={**kwargs, 'args_count': len(args)},
                    start_time=time.time()
                )

            # 执行操作（带重试）
            last_exception = None
            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)

                    # 更新上下文的结束时间
                    if context:
                        context.end_time = time.time()
                        context.duration = context.end_time - context.start_time

                    return result

                except Exception as e:
                    last_exception = e

                    # 记录异常信息
                    if strategy != ExceptionHandlingStrategy.SILENT_RETURN_DEFAULT:
                        log_func = getattr(logger, log_level.name.lower(), logger.error)

                        context_info = ""
                        if context:
                            context_info = f" [{context.component}.{context.operation}]"

                        if attempt < max_retries:
                            log_func(
                                f"操作异常{context_info} (尝试 {attempt + 1}/{max_retries + 1}): {e}")
                            if retry_delay > 0:
                                time.sleep(retry_delay)
                        else:
                            log_func(f"操作异常{context_info}: {e}")

                    # 如果不是最后一次尝试，继续重试
                    if attempt < max_retries:
                        continue

                    # 处理异常策略
                    if strategy == ExceptionHandlingStrategy.LOG_AND_RERAISE:
                        raise
                    elif strategy == ExceptionHandlingStrategy.COLLECT_AND_RETURN:
                        return {
                            'success': False,
                            'error': str(e),
                            'exception': e,
                            'context': context.to_dict() if context else None
                        }
                    else:  # LOG_AND_RETURN_DEFAULT 或 SILENT_RETURN_DEFAULT
                        return default_return

            # 这行代码理论上不会到达，但为了完整性保留
            return default_return

        return wrapper
    return decorator

# ==================== 便捷装饰器 ====================


def handle_config_exceptions(default_return=None, log_level=LogLevel.WARNING):
    """配置管理异常处理装饰器"""
    return handle_exceptions(
        strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
        default_return=default_return,
        log_level=log_level,
        include_context=True
    )


def handle_cache_exceptions(default_return=None, log_level=LogLevel.ERROR):
    """缓存操作异常处理装饰器"""
    return handle_exceptions(
        strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
        default_return=default_return,
        log_level=log_level,
        include_context=True
    )


def handle_monitoring_exceptions(default_return=None, log_level=LogLevel.WARNING):
    """监控操作异常处理装饰器"""
    return handle_exceptions(
        strategy=ExceptionHandlingStrategy.SILENT_RETURN_DEFAULT,
        default_return=default_return,
        log_level=log_level,
        include_context=False
    )


def handle_validation_exceptions(default_return=None, log_level=LogLevel.ERROR):
    """验证操作异常处理装饰器"""
    return handle_exceptions(
        strategy=ExceptionHandlingStrategy.COLLECT_AND_RETURN,
        default_return=default_return,
        log_level=log_level,
        include_context=True
    )

# ==================== 异常收集器 ====================


class ExceptionCollector:
    """异常收集器

    用于在批量操作中收集和处理多个异常
    """

    def __init__(self, max_exceptions: int = 100):
        """初始化异常收集器

        Args:
            max_exceptions: 最大收集异常数量
        """
        self.exceptions: List[Dict[str, Any]] = []
        self.max_exceptions = max_exceptions
        self._lock = threading.Lock()

    def add_exception(self, exception: Exception, context: Optional[ExceptionContext] = None, tb_str: Optional[str] = None):
        """添加异常

        Args:
            exception: 异常对象
            context: 异常上下文
            tb_str: 异常追踪信息字符串，如果不提供则自动获取
        """
        if exception is None:
            return  # 不收集None异常

        with self._lock:
            if len(self.exceptions) >= self.max_exceptions:
                return  # 超过最大数量，丢弃新异常

            # 获取traceback信息
            if tb_str is None:
                tb_str = ''.join(traceback.format_exception(type(exception), exception, exception.__traceback__))

            exception_info = {
                'exception_type': type(exception).__name__,
                'message': str(exception),
                'context': context.to_dict() if context else None,
                'timestamp': time.time(),
                'traceback': tb_str
            }

            self.exceptions.append(exception_info)

    def get_exceptions(self, severity_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取收集的异常

        Args:
            severity_filter: 严重程度过滤器

        Returns:
            异常列表
        """
        with self._lock:
            if severity_filter:
                # 这里可以根据上下文中的严重程度进行过滤
                # 暂时返回所有异常
                return self.exceptions.copy()
            return self.exceptions.copy()

    def clear(self):
        """清空异常收集器"""
        with self._lock:
            self.exceptions.clear()

    def has_exceptions(self) -> bool:
        """检查是否有异常"""
        with self._lock:
            return len(self.exceptions) > 0

    def get_summary(self) -> Dict[str, Any]:
        """获取异常摘要"""
        with self._lock:
            by_type = {}
            for exc in self.exceptions:
                exc_type = exc['exception_type']
                by_type[exc_type] = by_type.get(exc_type, 0) + 1

            # 计算时间戳统计
            if self.exceptions:
                timestamps = [exc['timestamp'] for exc in self.exceptions]
                latest_timestamp = max(timestamps)
                earliest_timestamp = min(timestamps)
            else:
                latest_timestamp = None
                earliest_timestamp = None

            return {
                'total_count': len(self.exceptions),
                'by_type': by_type,
                'latest_timestamp': latest_timestamp,
                'earliest_timestamp': earliest_timestamp,
                'max_capacity': self.max_exceptions,
                'utilization_rate': len(self.exceptions) / self.max_exceptions if self.max_exceptions > 0 else 0
            }

# ==================== 全局异常收集器实例 ====================


global_exception_collector = ExceptionCollector(max_exceptions=1000)




