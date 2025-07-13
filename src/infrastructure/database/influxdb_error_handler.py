"""InfluxDB异常处理策略"""
import time
import logging
from typing import Callable, Optional, TypeVar, Any
from functools import wraps
from influxdb_client.rest import ApiException
from src.infrastructure.error.error_handler import ErrorHandler, ErrorLevel

T = TypeVar('T')

class InfluxDBErrorHandler:
    """InfluxDB专用异常处理器"""

    def __init__(self, error_handler: ErrorHandler):
        self.error_handler = error_handler
        self.retry_config = {
            'max_attempts': 3,
            'delay': 1,
            'backoff': 2
        }
        self._circuit_open = False
        self._failures = 0
        self._last_failure = 0
        self.logger = logging.getLogger(__name__)

    def configure_retry(self, max_attempts: int = 3,
                       delay: float = 1, backoff: float = 2):
        """配置重试策略"""
        self.retry_config = {
            'max_attempts': max_attempts,
            'delay': delay,
            'backoff': backoff
        }

    def handle_connection_error(self, operation: str, exception: Exception):
        """处理连接错误"""
        self.error_handler.log(
            f"InfluxDB连接失败 - 操作: {operation}",
            exception,
            ErrorLevel.CRITICAL
        )
        # 触发连接恢复流程
        self._recover_connection()

    def handle_write_error(self, exception: Exception) -> None:
        """处理写入错误"""
        if hasattr(exception, 'status') and exception.status == 429:
            # 速率限制错误
            self.log_error(
                "写入操作",
                exception,
                "WARNING",
                "等待后重试"
            )
        else:
            # 其他写入错误
            self.log_error(
                "写入操作",
                exception,
                "ERROR",
                "写入失败"
            )
            if self.error_handler:
                self.error_handler.handle_error(exception)

    def handle_query_error(self, operation: str, exception: Exception):
        """处理查询错误"""
        self.error_handler.log(
            f"InfluxDB查询错误 - 操作: {operation}",
            exception,
            ErrorLevel.ERROR
        )

    def handle_management_error(self, operation: str, exception: Exception):
        """处理管理操作错误"""
        self.error_handler.log(
            f"InfluxDB管理操作错误 - 操作: {operation}",
            exception,
            ErrorLevel.WARNING
        )

    def retry_on_exception(self, func: Callable[..., T]) -> Callable[..., T]:
        """自动重试装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_exception = None
            delay = self.retry_config['delay']

            for attempt in range(1, self.retry_config['max_attempts'] + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    operation = f"{func.__name__} (尝试 {attempt}/{self.retry_config['max_attempts']})"

                    # 根据异常类型调用不同的处理方法
                    if isinstance(e, ConnectionError):
                        self.handle_connection_error(operation, e)
                        raise  # 连接错误直接抛出，不重试
                    elif "write" in func.__name__.lower():
                        self.handle_write_error(e)
                    elif "query" in func.__name__.lower():
                        self.handle_query_error(operation, e)
                    else:
                        self.handle_management_error(operation, e)

                    if attempt < self.retry_config['max_attempts']:
                        time.sleep(delay)
                        delay *= self.retry_config['backoff']

            raise last_exception if last_exception else RuntimeError("未知错误")
        return wrapper

    def fallback_on_exception(self, func: Callable[..., T]) -> Callable[..., T]:
        """异常时回退装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                self.error_handler.log(
                    f"InfluxDB操作降级 - 操作: {func.__name__}",
                    e,
                    ErrorLevel.WARNING
                )
                # 返回默认值或重新抛出异常
                return None
        return wrapper

    def _recover_connection(self):
        """连接恢复策略"""
        # 1. 尝试重新连接
        # 2. 如果失败，切换到备用实例
        # 3. 发送告警通知
        pass

    def circuit_breaker(self, func: Callable[..., T]) -> Callable[..., T]:
        """熔断器装饰器"""
        @wraps(func)
        def wrapper(*args, **kwargs) -> T:
            current_time = time.time()

            # 检查熔断状态
            if self._circuit_open:
                if current_time - self._last_failure > 60:  # 60秒恢复时间
                    # 尝试恢复
                    self._circuit_open = False
                else:
                    raise RuntimeError("Circuit is open")

            try:
                result = func(*args, **kwargs)
                # 成功调用重置失败计数
                self._failures = 0
                return result
            except Exception as e:
                self._failures += 1
                self._last_failure = current_time

                if self._failures >= 3:  # 3次失败后熔断
                    self._circuit_open = True
                    self.error_handler.log(
                        f"InfluxDB熔断器触发 - 操作: {func.__name__}",
                        e,
                        ErrorLevel.CRITICAL
                    )

                raise
        return wrapper

    @property
    def circuit_state(self):
        """获取熔断器状态"""
        return "open" if self._circuit_open else "closed"
