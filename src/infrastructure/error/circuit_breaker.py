from functools import wraps
from typing import Callable, Any
import time
import logging

logger = logging.getLogger(__name__)

class CircuitBreaker:
    """熔断器实现，用于防止级联故障"""

    def __init__(self,
                 failure_threshold: int = 3,
                 recovery_timeout: float = 30.0,
                 expected_exceptions: tuple = (Exception,)):
        """
        Args:
            failure_threshold: 触发熔断的连续失败次数
            recovery_timeout: 熔断后恢复时间(秒)
            expected_exceptions: 触发熔断的异常类型
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exceptions = expected_exceptions
        self.state = "closed"  # closed, open, half-open
        self.failure_count = 0
        self.last_failure_time = 0.0

    def __call__(self, func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            return self.call(func, *args, **kwargs)
        return wrapper

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行被包装的函数，应用熔断逻辑"""
        if self.state == "open":
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = "half-open"
                logger.warning("Circuit breaker transitioning to half-open state")
            else:
                raise CircuitBreakerError("Circuit breaker is open")

        try:
            result = func(*args, **kwargs)
            if self.state == "half-open":
                self.reset()
            return result
        except self.expected_exceptions as e:
            self.record_failure()
            raise

    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """execute方法作为call方法的别名"""
        return self.call(func, *args, **kwargs)

    def record_failure(self):
        """记录失败"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "open"
            logger.error("Circuit breaker tripped! State: open")

    def reset(self):
        """重置熔断器"""
        self.state = "closed"
        self.failure_count = 0
        self.last_failure_time = 0.0

    def is_open(self):
        """检查熔断器是否打开"""
        return self.state == "open"

    def is_closed(self):
        """检查熔断器是否关闭"""
        return self.state == "closed"

    def is_half_open(self):
        """检查熔断器是否半开"""
        return self.state == "half-open"

class CircuitBreakerError(Exception):
    """熔断器专用异常"""
    pass

# 默认熔断器实例
circuit_breaker = CircuitBreaker()
