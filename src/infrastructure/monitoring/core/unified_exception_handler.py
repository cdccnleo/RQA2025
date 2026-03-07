#!/usr/bin/env python3
"""
RQA2025 基础设施层统一异常处理框架

提供统一的异常处理机制，支持多种策略、重试机制和上下文跟踪。
"""

from typing import Dict, Any, Optional, Callable, List, Type
from datetime import datetime
import logging
import traceback
import functools
import time

logger = logging.getLogger(__name__)


class MonitoringException(Exception):
    """
    监控系统基础异常类

    提供丰富的异常信息和上下文。
    """

    def __init__(self, message: str, error_code: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None,
                 cause: Optional[Exception] = None):
        """
        初始化监控异常

        Args:
            message: 异常消息
            error_code: 错误代码
            context: 异常上下文信息
            cause: 原始异常
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "MONITORING_ERROR"
        self.context = context or {}
        self.cause = cause
        self.timestamp = datetime.now()
        self.traceback = traceback.format_exc()

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典格式

        Returns:
            Dict[str, Any]: 异常信息字典
        """
        return {
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp.isoformat(),
            'traceback': self.traceback,
            'cause': str(self.cause) if self.cause else None
        }


class ValidationError(MonitoringException):
    """数据验证异常"""
    def __init__(self, message: str, field: Optional[str] = None, value: Any = None):
        super().__init__(
            message,
            error_code="VALIDATION_ERROR",
            context={'field': field, 'value': value}
        )


class ConfigurationError(MonitoringException):
    """配置异常"""
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message,
            error_code="CONFIGURATION_ERROR",
            context={'config_key': config_key}
        )


class ConnectionError(MonitoringException):
    """连接异常"""
    def __init__(self, message: str, host: Optional[str] = None, port: Optional[int] = None):
        super().__init__(
            message,
            error_code="CONNECTION_ERROR",
            context={'host': host, 'port': port}
        )


class DataPersistenceError(MonitoringException):
    """数据持久化异常"""
    def __init__(self, message: str, operation: Optional[str] = None, data_size: Optional[int] = None):
        super().__init__(
            message,
            error_code="DATA_PERSISTENCE_ERROR",
            context={'operation': operation, 'data_size': data_size}
        )


class AlertProcessingError(MonitoringException):
    """告警处理异常"""
    def __init__(self, message: str, alert_id: Optional[str] = None, rule_id: Optional[str] = None):
        super().__init__(
            message,
            error_code="ALERT_PROCESSING_ERROR",
            context={'alert_id': alert_id, 'rule_id': rule_id}
        )


class NotificationError(MonitoringException):
    """通知异常"""
    def __init__(self, message: str, channel: Optional[str] = None, recipient: Optional[str] = None):
        super().__init__(
            message,
            error_code="NOTIFICATION_ERROR",
            context={'channel': channel, 'recipient': recipient}
        )


class ExceptionHandlingStrategy:
    """
    异常处理策略基类

    定义异常处理的通用接口。
    """

    def handle(self, exception: Exception, context: Dict[str, Any]) -> Any:
        """
        处理异常

        Args:
            exception: 异常对象
            context: 处理上下文

        Returns:
            Any: 处理结果
        """
        raise NotImplementedError("子类必须实现handle方法")


class LogAndContinueStrategy(ExceptionHandlingStrategy):
    """记录日志并继续执行的策略"""

    def handle(self, exception: Exception, context: Dict[str, Any]) -> Any:
        """
        记录异常日志并返回默认值

        Args:
            exception: 异常对象
            context: 处理上下文

        Returns:
            Any: 默认返回值
        """
        operation = context.get('operation', 'unknown')
        logger.warning(f"操作 '{operation}' 失败，继续执行: {exception}")

        # 返回默认值
        return context.get('default_value', None)


class RetryStrategy(ExceptionHandlingStrategy):
    """重试策略"""

    def __init__(self, max_retries: int = 3, delay_seconds: float = 1.0,
                 backoff_factor: float = 2.0):
        """
        初始化重试策略

        Args:
            max_retries: 最大重试次数
            delay_seconds: 初始延迟秒数
            backoff_factor: 退避因子
        """
        self.max_retries = max_retries
        self.delay_seconds = delay_seconds
        self.backoff_factor = backoff_factor

    def handle(self, exception: Exception, context: Dict[str, Any]) -> Any:
        """
        执行重试逻辑

        Args:
            exception: 异常对象
            context: 处理上下文

        Returns:
            Any: 重试结果
        """
        operation = context.get('operation', 'unknown')
        func = context.get('func')
        args = context.get('args', ())
        kwargs = context.get('kwargs', {})

        if not func:
            raise exception

        for attempt in range(self.max_retries + 1):
            try:
                if attempt > 0:
                    delay = self.delay_seconds * (self.backoff_factor ** (attempt - 1))
                    logger.info(f"重试操作 '{operation}' (尝试 {attempt}/{self.max_retries})，延迟 {delay:.1f}秒")
                    time.sleep(delay)

                return func(*args, **kwargs)

            except Exception as e:
                if attempt == self.max_retries:
                    logger.error(f"操作 '{operation}' 在 {self.max_retries} 次重试后仍然失败")
                    raise e
                else:
                    logger.warning(f"操作 '{operation}' 失败 (尝试 {attempt + 1}/{self.max_retries + 1}): {e}")


class RaiseStrategy(ExceptionHandlingStrategy):
    """直接抛出异常的策略"""

    def handle(self, exception: Exception, context: Dict[str, Any]) -> Any:
        """
        直接重新抛出异常

        Args:
            exception: 异常对象
            context: 处理上下文

        Returns:
            Any: 不返回（会抛出异常）
        """
        operation = context.get('operation', 'unknown')
        logger.error(f"操作 '{operation}' 失败，抛出异常: {exception}")
        raise exception


class ExceptionHandler:
    """
    统一异常处理器

    提供灵活的异常处理机制，支持多种策略和上下文跟踪。
    """

    def __init__(self):
        """初始化异常处理器"""
        self.strategies: Dict[str, ExceptionHandlingStrategy] = {
            'log_and_continue': LogAndContinueStrategy(),
            'retry': RetryStrategy(),
            'raise': RaiseStrategy(),
        }

        # 默认策略映射
        self.default_strategies: Dict[Type[Exception], str] = {
            ValidationError: 'raise',
            ConfigurationError: 'raise',
            ConnectionError: 'retry',
            DataPersistenceError: 'log_and_continue',
            AlertProcessingError: 'log_and_continue',
            NotificationError: 'log_and_continue',
            Exception: 'log_and_continue',
        }

    def add_strategy(self, name: str, strategy: ExceptionHandlingStrategy):
        """
        添加异常处理策略

        Args:
            name: 策略名称
            strategy: 策略实例
        """
        self.strategies[name] = strategy

    def handle_exception(self, exception: Exception, operation: str = "unknown",
                        strategy: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Any:
        """
        处理异常

        Args:
            exception: 异常对象
            operation: 操作名称
            strategy: 处理策略，如果为None则使用默认策略
            context: 额外上下文信息

        Returns:
            Any: 处理结果
        """
        # 确定处理策略
        if strategy is None:
            strategy = self._get_default_strategy(exception)

        # 获取策略实例
        strategy_instance = self.strategies.get(strategy)
        if not strategy_instance:
            logger.error(f"未知的异常处理策略: {strategy}")
            raise exception

        # 构建处理上下文
        handle_context = {
            'operation': operation,
            'exception_type': type(exception).__name__,
            'timestamp': datetime.now().isoformat(),
        }

        if context:
            handle_context.update(context)

        # 执行处理
        try:
            return strategy_instance.handle(exception, handle_context)
        except Exception as e:
            logger.error(f"异常处理策略 '{strategy}' 执行失败: {e}")
            raise exception

    def _get_default_strategy(self, exception: Exception) -> str:
        """
        获取异常的默认处理策略

        Args:
            exception: 异常对象

        Returns:
            str: 策略名称
        """
        exception_type = type(exception)
        for exc_class, strategy in self.default_strategies.items():
            if issubclass(exception_type, exc_class):
                return strategy

        return 'log_and_continue'


def handle_monitoring_exception(operation: str = "unknown", strategy: Optional[str] = None,
                               context: Optional[Dict[str, Any]] = None):
    """
    监控异常处理装饰器

    Args:
        operation: 操作名称
        strategy: 处理策略
        context: 额外上下文

    Returns:
        Callable: 装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                handler = ExceptionHandler()
                return handler.handle_exception(e, operation, strategy, context)
        return wrapper
    return decorator


def with_exception_handling(operation: str = "unknown", strategy: Optional[str] = None,
                          context: Optional[Dict[str, Any]] = None):
    """
    异常处理上下文管理器

    Args:
        operation: 操作名称
        strategy: 处理策略
        context: 额外上下文

    Returns:
        ExceptionHandlingContext: 上下文管理器
    """
    return ExceptionHandlingContext(operation, strategy, context)


class ExceptionHandlingContext:
    """
    异常处理上下文管理器
    """

    def __init__(self, operation: str = "unknown", strategy: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        """
        初始化上下文管理器

        Args:
            operation: 操作名称
            strategy: 处理策略
            context: 额外上下文
        """
        self.operation = operation
        self.strategy = strategy
        self.context = context
        self.handler = ExceptionHandler()

    def __enter__(self):
        """进入上下文"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        if exc_val is not None:
            self.handler.handle_exception(exc_val, self.operation, self.strategy, self.context)
            return True  # 抑制异常


# 全局异常处理器实例
global_exception_handler = ExceptionHandler()


def handle_exception(exception: Exception, operation: str = "unknown",
                   strategy: Optional[str] = None, context: Optional[Dict[str, Any]] = None) -> Any:
    """
    全局异常处理函数

    Args:
        exception: 异常对象
        operation: 操作名称
        strategy: 处理策略
        context: 额外上下文

    Returns:
        Any: 处理结果
    """
    return global_exception_handler.handle_exception(exception, operation, strategy, context)
