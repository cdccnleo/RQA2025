#!/usr / bin / env python
# -*- coding: utf-8 -*-

"""
MiniQMT限流机制
实现多种限流策略，防止高频请求导致服务拒绝
"""

import time
import threading
import logging
from typing import Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum
from collections import deque

logger = logging.getLogger(__name__)


class RateLimitType(Enum):

    """限流类型"""
    FIXED_WINDOW = "fixed_window"      # 固定窗口
    SLIDING_WINDOW = "sliding_window"   # 滑动窗口
    LEAKY_BUCKET = "leaky_bucket"       # 漏桶算法
    TOKEN_BUCKET = "token_bucket"       # 令牌桶算法


class RateLimitStrategy(Enum):

    """限流策略"""
    REJECT = "reject"           # 直接拒绝
    QUEUE = "queue"             # 排队等待
    DEGRADE = "degrade"         # 降级处理
    RETRY = "retry"             # 重试机制


@dataclass
class RateLimitConfig:

    """限流配置"""
    limit_type: RateLimitType
    max_requests: int
    time_window: float
    approach: RateLimitStrategy = RateLimitStrategy.REJECT
    retry_delay: float = 1.0
    queue_timeout: float = 30.0
    burst_size: int = 10


class RateLimiter:

    """MiniQMT限流器"""

    def __init__(self, config: Dict[str, Any]):
        """
        初始化限流器

        Args:
            config: 限流配置
        """
        self.config = config
        self.limiters: Dict[str, Any] = {}
        self._lock = threading.RLock()

        # 统计信息
        self._stats = {
            'total_requests': 0,
            'limited_requests': 0,
            'rejected_requests': 0,
            'queued_requests': 0,
            'degraded_requests': 0,
            'retry_requests': 0
        }

    def create_limiter(self, key: str, config: RateLimitConfig) -> Any:
        """
        创建限流器

        Args:
            key: 限流键
            config: 限流配置

        Returns:
            限流器实例
        """
        with self._lock:
            if key in self.limiters:
                return self.limiters[key]

            if config.limit_type == RateLimitType.FIXED_WINDOW:
                limiter = FixedWindowLimiter(config)
            elif config.limit_type == RateLimitType.SLIDING_WINDOW:
                limiter = SlidingWindowLimiter(config)
            elif config.limit_type == RateLimitType.LEAKY_BUCKET:
                limiter = LeakyBucketLimiter(config)
            elif config.limit_type == RateLimitType.TOKEN_BUCKET:
                limiter = TokenBucketLimiter(config)
            else:
                raise ValueError(f"不支持的限流类型: {config.limit_type}")

            self.limiters[key] = limiter
            return limiter

    def is_allowed(self, key: str, request_id: str = None) -> bool:
        """
        检查请求是否允许

        Args:
            key: 限流键
            request_id: 请求ID

        Returns:
            是否允许请求
        """
        with self._lock:
            self._stats['total_requests'] += 1

            if key not in self.limiters:
                return True

            limiter = self.limiters[key]
            return limiter.is_allowed(request_id)

    def acquire(self, key: str, request_id: str = None, timeout: float = 10.0) -> bool:
        """
        获取请求许可

        Args:
            key: 限流键
            request_id: 请求ID
            timeout: 超时时间

        Returns:
            是否获取成功
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            if self.is_allowed(key, request_id):
                return True

            # 根据策略处理
            if key in self.limiters:
                limiter = self.limiters[key]
                strategy = limiter.config.approach

                if strategy == RateLimitStrategy.REJECT:
                    self._stats['rejected_requests'] += 1
                    logger.warning(f"请求被拒绝: {key}")
                    return False

                elif strategy == RateLimitStrategy.QUEUE:
                    self._stats['queued_requests'] += 1
                    time.sleep(0.1)  # 短暂等待
                    continue

                elif strategy == RateLimitStrategy.DEGRADE:
                    self._stats['degraded_requests'] += 1
                    logger.info(f"请求降级处理: {key}")
                    return True  # 允许但降级

                elif strategy == RateLimitStrategy.RETRY:
                    self._stats['retry_requests'] += 1
                    time.sleep(limiter.config.retry_delay)
                    continue

        self._stats['limited_requests'] += 1
        logger.warning(f"请求超时: {key}")
        return False

    def get_stats(self) -> Dict[str, Any]:
        """获取限流统计信息"""
        with self._lock:
            stats = self._stats.copy()
            for key, limiter in self.limiters.items():
                stats[f'{key}_limiter_stats'] = limiter.get_stats()
            return stats


class FixedWindowLimiter:

    """固定窗口限流器"""

    def __init__(self, config: RateLimitConfig):

        self.config = config
        self.current_window = int(time.time() / config.time_window)
        self.request_count = 0
        self._lock = threading.Lock()

    def is_allowed(self, request_id: str = None) -> bool:
        """检查是否允许请求"""
        with self._lock:
            current_time = time.time()
            window = int(current_time / self.config.time_window)

            if window != self.current_window:
                self.current_window = window
                self.request_count = 0

            if self.request_count < self.config.max_requests:
                self.request_count += 1
                return True

            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'current_window': self.current_window,
                'request_count': self.request_count,
                'max_requests': self.config.max_requests
            }


class SlidingWindowLimiter:

    """滑动窗口限流器"""

    def __init__(self, config: RateLimitConfig):

        self.config = config
        self.requests = deque()
        self._lock = threading.Lock()

    def is_allowed(self, request_id: str = None) -> bool:
        """检查是否允许请求"""
        with self._lock:
            current_time = time.time()

            # 清理过期请求
            while self.requests and current_time - self.requests[0] > self.config.time_window:
                self.requests.popleft()

            if len(self.requests) < self.config.max_requests:
                self.requests.append(current_time)
                return True

            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'current_requests': len(self.requests),
                'max_requests': self.config.max_requests,
                'window_size': self.config.time_window
            }


class LeakyBucketLimiter:

    """漏桶限流器"""

    def __init__(self, config: RateLimitConfig):

        self.config = config
        self.capacity = config.max_requests
        self.current_water = 0
        self.last_leak_time = time.time()
        self.leak_rate = config.max_requests / config.time_window
        self._lock = threading.Lock()

    def is_allowed(self, request_id: str = None) -> bool:
        """检查是否允许请求"""
        with self._lock:
            current_time = time.time()

            # 计算漏出的水量
            time_passed = current_time - self.last_leak_time
            leaked_water = time_passed * self.leak_rate

            self.current_water = max(0, self.current_water - leaked_water)
            self.last_leak_time = current_time

            if self.current_water < self.capacity:
                self.current_water += 1
                return True

            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'current_water': self.current_water,
                'capacity': self.capacity,
                'leak_rate': self.leak_rate
            }


class TokenBucketLimiter:

    """令牌桶限流器"""

    def __init__(self, config: RateLimitConfig):

        self.config = config
        self.capacity = config.max_requests
        self.tokens = config.max_requests  # 初始令牌数
        self.last_refill_time = time.time()
        self.refill_rate = config.max_requests / config.time_window
        self._lock = threading.Lock()

    def is_allowed(self, request_id: str = None) -> bool:
        """检查是否允许请求"""
        with self._lock:
            current_time = time.time()

            # 计算新增令牌
            time_passed = current_time - self.last_refill_time
            new_tokens = time_passed * self.refill_rate

            self.tokens = min(self.capacity, self.tokens + new_tokens)
            self.last_refill_time = current_time

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            return False

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'current_tokens': self.tokens,
                'capacity': self.capacity,
                'refill_rate': self.refill_rate
            }


class RateLimitDecorator:

    """限流装饰器"""

    def __init__(self, rate_limiter: RateLimiter, key: str, timeout: float = 10.0):

        self.rate_limiter = rate_limiter
        self.key = key
        self.timeout = timeout

    def __call__(self, func: Callable) -> Callable:

        def wrapper(*args, **kwargs):

            if self.rate_limiter.acquire(self.key, timeout=self.timeout):
                return func(*args, **kwargs)
            else:
                raise Exception(f"请求被限流: {self.key}")
        return wrapper


def rate_limit(limiter: RateLimiter, key: str, timeout: float = 10.0):
    """限流装饰器工厂函数"""
    return RateLimitDecorator(limiter, key, timeout)
