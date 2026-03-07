
import secrets
import time
import random

from enum import Enum
from ..core.interfaces import IRetryPolicy
from typing import Optional, Callable, Any, Dict
"""
重试策略组件
"""


class RetryStrategy(Enum):
    """重试策略枚举"""
    FIXED = "fixed"           # 固定间隔
    EXPONENTIAL = "exponential"  # 指数增长
    LINEAR = "linear"         # 线性增长
    RANDOM = "random"         # 随机间隔


class RetryPolicy(IRetryPolicy):
    """重试策略管理器"""

    def __init__(self,
                 max_attempts: int = 3,
                 base_delay: float = 1.0,
                 max_delay: float = 60.0,
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                 jitter: bool = True,
                 backoff_factor: float = 2.0):
        """
        初始化重试策略

        Args:
            max_attempts: 最大尝试次数
            base_delay: 基础延迟时间
            max_delay: 最大延迟时间
            strategy: 重试策略
            jitter: 是否启用抖动
            backoff_factor: 退避因子
        """
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.strategy = strategy
        self.jitter = jitter
        self.backoff_factor = backoff_factor

    def execute(self, func: Callable, *args, **kwargs):
        """执行函数，应用重试策略"""
        last_exception: Optional[Exception] = None

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

            if attempt < self.max_attempts - 1 and last_exception is not None:
                delay = self.calculate_delay(attempt)
                time.sleep(delay)

        # 所有重试都失败了
        if last_exception is not None:
            raise last_exception
        else:
            # 这种情况理论上不应该发生，但为了类型检查安全
            raise Exception("Retry failed without capturing an exception")

    def calculate_delay(self, attempt: int) -> float:
        """计算延迟时间"""
        if self.strategy == RetryStrategy.FIXED:
            delay = self.base_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.base_delay * (self.backoff_factor ** attempt)
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.base_delay * (attempt + 1)
        elif self.strategy == RetryStrategy.RANDOM:
            delay = random.uniform(self.base_delay, self.max_delay)
        else:
            delay = self.base_delay

        # 应用抖动（仅当启用时）
        if self.jitter:
            delay = delay * (0.5 + random.random() * 0.5)

        # 限制最大延迟
        return min(delay, self.max_delay)

    def should_retry(self, attempt: int, exception: Exception) -> bool:
        """判断是否应该重试"""
        # 超过最大尝试次数
        if attempt >= self.max_attempts:
            return False

        # 不可重试的异常类型
        non_retryable_exceptions = (
            KeyboardInterrupt,
            SystemExit,
            MemoryError,
        )

        if isinstance(exception, non_retryable_exceptions):
            return False

        return True

    def get_retry_stats(self) -> Dict[str, Any]:
        """获取重试统计"""
        return {
            'max_attempts': self.max_attempts,
            'base_delay': self.base_delay,
            'max_delay': self.max_delay,
            'strategy': self.strategy.value,
            'jitter_enabled': self.jitter,
            'backoff_factor': self.backoff_factor
        }

    def reset_stats(self) -> None:
        """重置统计"""
        # RetryPolicy目前没有运行时统计需要重置
