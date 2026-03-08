"""
Retry Mechanism Module
重试机制模块

This module provides retry functionality for async operations
此模块为异步操作提供重试功能

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable, Type
from enum import Enum
import time
import secrets
import functools

logger = logging.getLogger(__name__)


class RetryStrategy(Enum):

    """Retry strategy enumeration"""
    FIXED = "fixed"              # Fixed delay between retries
    EXPONENTIAL = "exponential"  # Exponential backoff
    LINEAR = "linear"           # Linear increase in delay
    RANDOM = "random"           # Random delay within range


class RetryResult:

    """
    Retry Result Class
    重试结果类

    Contains information about retry attempts and final result
    包含重试尝试和最终结果的信息
    """

    def __init__(self,


                 success: bool,
                 result: Any = None,
                 attempts: int = 0,
                 total_delay: float = 0.0,
                 errors: Optional[List[Exception]] = None):
        """
        Initialize retry result
        初始化重试结果

        Args:
            success: Whether the operation ultimately succeeded
                    操作最终是否成功
            result: Final result if successful
                   如果成功则为最终结果
            attempts: Number of attempts made
                     尝试次数
            total_delay: Total delay time across all retries
                        所有重试的总延迟时间
            errors: List of exceptions encountered during retries
                   重试过程中遇到的异常列表
        """
        self.success = success
        self.result = result
        self.attempts = attempts
        self.total_delay = total_delay
        self.errors = errors or []

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary
        转换为字典

        Returns:
            dict: Result data as dictionary
                  结果数据字典
        """
        return {
            'success': self.success,
            'result': self.result,
            'attempts': self.attempts,
            'total_delay': self.total_delay,
            'error_count': len(self.errors),
            'last_error': str(self.errors[-1]) if self.errors else None
        }


class RetryConfig:

    """
    Retry Configuration Class
    重试配置类

    Configuration for retry behavior
    重试行为的配置
    """

    def __init__(self,


                 max_attempts: int = 3,
                 initial_delay: float = 1.0,
                 max_delay: float = 60.0,
                 backoff_factor: float = 2.0,
                 strategy: RetryStrategy = RetryStrategy.EXPONENTIAL,
                 jitter: bool = True,
                 retry_on: Optional[List[Type[Exception]]] = None):
        """
        Initialize retry configuration
        初始化重试配置

        Args:
            max_attempts: Maximum number of retry attempts
                         最大重试尝试次数
            initial_delay: Initial delay between retries (seconds)
                          重试之间的初始延迟（秒）
            max_delay: Maximum delay between retries (seconds)
                      重试之间的最大延迟（秒）
            backoff_factor: Backoff factor for exponential / linear strategies
                           指数 / 线性策略的退避因子
            strategy: Retry strategy to use
                     要使用的重试策略
            jitter: Whether to add random jitter to delays
                   是否为延迟添加随机抖动
            retry_on: List of exception types to retry on (None for all)
                     要重试的异常类型列表（None表示全部）
        """
        self.max_attempts = max_attempts
        self.initial_delay = initial_delay
        self.max_delay = max_delay
        self.backoff_factor = backoff_factor
        self.strategy = strategy
        self.jitter = jitter
        self.retry_on = retry_on or [Exception]

    def should_retry(self, exception: Exception) -> bool:
        """
        Determine if the exception should trigger a retry
        确定异常是否应该触发重试

        Args:
            exception: Exception that occurred
                      发生的异常

        Returns:
            bool: True if should retry, False otherwise
                  如果应该重试则返回True，否则返回False
        """
        return any(isinstance(exception, exc_type) for exc_type in self.retry_on)

    def calculate_delay(self, attempt: int) -> float:
        """
        Calculate delay for the given attempt
        计算给定尝试的延迟

        Args:
            attempt: Current attempt number (1 - based)
                    当前尝试次数（从1开始）

        Returns:
            float: Delay in seconds
                   延迟秒数
        """
        if self.strategy == RetryStrategy.FIXED:
            delay = self.initial_delay
        elif self.strategy == RetryStrategy.EXPONENTIAL:
            delay = self.initial_delay * (self.backoff_factor ** (attempt - 1))
        elif self.strategy == RetryStrategy.LINEAR:
            delay = self.initial_delay + (self.backoff_factor * (attempt - 1))
        elif self.strategy == RetryStrategy.RANDOM:
            delay = secrets.uniform(self.initial_delay, self.max_delay)
        else:
            delay = self.initial_delay

        # Apply maximum delay limit
        delay = min(delay, self.max_delay)

        # Add jitter if enabled
        if self.jitter and delay > 0:
            jitter_factor = secrets.uniform(0.5, 1.5)
            delay *= jitter_factor

        return delay


class RetryMechanism:

    """
    Retry Mechanism Class
    重试机制类

    Provides automatic retry functionality for operations
    为操作提供自动重试功能
    """

    def __init__(self, config: Optional[RetryConfig] = None):
        """
        Initialize retry mechanism
        初始化重试机制

        Args:
            config: Retry configuration (default config if None)
                   重试配置（如果为None则使用默认配置）
        """
        self.config = config or RetryConfig()
        self.retry_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_retries': 0,
            'average_attempts': 0.0
        }

        logger.info("Retry mechanism initialized")

    def execute_with_retry(self,


                           func: Callable,
                           *args,
                           config: Optional[RetryConfig] = None,
                           **kwargs) -> RetryResult:
        """
        Execute a function with retry logic
        使用重试逻辑执行函数

        Args:
            func: Function to execute
                 要执行的函数
            *args: Positional arguments for the function
                  函数的位置参数
            config: Retry configuration override (None to use instance config)
                   重试配置覆盖（None表示使用实例配置）
            **kwargs: Keyword arguments for the function
                     函数的关键字参数

        Returns:
            RetryResult: Result of the operation with retry information
                        包含重试信息的结果
        """
        retry_config = config or self.config
        errors = []
        total_delay = 0.0

        self.retry_stats['total_operations'] += 1

        for attempt in range(1, retry_config.max_attempts + 1):
            try:
                start_time = time.time()
                result = func(*args, **kwargs)
                execution_time = time.time() - start_time

                # Success
                self.retry_stats['successful_operations'] += 1
                if attempt > 1:
                    self.retry_stats['total_retries'] += (attempt - 1)

                return RetryResult(
                    success=True,
                    result=result,
                    attempts=attempt,
                    total_delay=total_delay
                )

            except Exception as e:
                errors.append(e)

                # Check if we should retry this exception
                if not retry_config.should_retry(e):
                    logger.debug(f"Exception {type(e).__name__} not configured for retry")
                    break

                # If this was the last attempt, don't wait
                if attempt == retry_config.max_attempts:
                    break

                # Calculate delay and wait
                delay = retry_config.calculate_delay(attempt)
                total_delay += delay

                logger.warning(f"Attempt {attempt} failed: {str(e)}. Retrying in {delay:.2f}s...")
                time.sleep(delay)

        # All attempts failed
        self.retry_stats['failed_operations'] += 1
        if retry_config.max_attempts > 1:
            self.retry_stats['total_retries'] += (retry_config.max_attempts - 1)

        return RetryResult(
            success=False,
            attempts=retry_config.max_attempts,
            total_delay=total_delay,
            errors=errors
        )

    def retry_async(self, func: Callable) -> Callable:
        """
        Decorator to add retry functionality to async functions
        为异步函数添加重试功能的装饰器

        Args:
            func: Function to decorate
                 要装饰的函数

        Returns:
            Decorated function with retry capability
            具有重试能力的装饰函数
        """
        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            return self.execute_with_retry(func, *args, **kwargs)

        return wrapper

    def get_retry_stats(self) -> Dict[str, Any]:
        """
        Get retry statistics
        获取重试统计信息

        Returns:
            dict: Retry statistics
                  重试统计信息
        """
        stats = self.retry_stats.copy()

        # Calculate derived statistics
        if stats['total_operations'] > 0:
            stats['success_rate'] = stats['successful_operations'] / stats['total_operations'] * 100
            if stats['successful_operations'] > 0:
                stats['average_attempts'] = (
                    stats['total_operations'] + stats['total_retries']) / stats['successful_operations']

        return stats

    def reset_stats(self) -> None:
        """
        Reset retry statistics
        重置重试统计信息
        """
        self.retry_stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'total_retries': 0,
            'average_attempts': 0.0
        }
        logger.info("Retry statistics reset")


def with_retry(config: Optional[RetryConfig] = None):
    """
    Decorator to add retry functionality to any function
    为任何函数添加重试功能的装饰器

    Args:
        config: Retry configuration
               重试配置

    Returns:
        Decorator function
        装饰器函数
    """

    def decorator(func: Callable) -> Callable:

        retry_mechanism = RetryMechanism(config)

        @functools.wraps(func)
        def wrapper(*args, **kwargs):

            return retry_mechanism.execute_with_retry(func, *args, **kwargs)

        return wrapper

    return decorator


# Global retry mechanism instance
# 全局重试机制实例
retry_mechanism = RetryMechanism()

__all__ = [
    'RetryStrategy',
    'RetryResult',
    'RetryConfig',
    'RetryMechanism',
    'retry_mechanism',
    'with_retry'
]
