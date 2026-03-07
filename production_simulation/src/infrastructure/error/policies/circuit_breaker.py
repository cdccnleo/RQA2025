"""
circuit_breaker 模块

提供 circuit_breaker 相关功能和接口。
"""

import os

import threading
import time

from prometheus_client import Gauge, Counter, REGISTRY
from enum import Enum
from ..core.interfaces import ICircuitBreaker
from typing import Dict, Any, Callable
"""
熔断器组件
"""


class CircuitBreakerState(Enum):
    """熔断器状态枚举"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker(ICircuitBreaker):
    """熔断器实现"""

    def __init__(self,
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 success_threshold: int = 2,
                 name: str = "default"):
        """初始化熔断器"""
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        self.name = name

        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

        self._lock = threading.Lock()

        # Prometheus指标（如果可用）
        self.state_gauge = None
        self.failure_counter = None

        self._init_metrics()

    def _init_metrics(self) -> None:
        """初始化指标"""
        try:
            if os.name != 'nt':  # 非Windows系统
                try:
                    self.state_gauge = Gauge(
                        'circuit_breaker_state',
                        'Circuit breaker state',
                        ['name'],
                        registry=REGISTRY
                    )
                    self.failure_counter = Counter(
                        'circuit_breaker_failures_total',
                        'Total circuit breaker failures',
                        ['name'],
                        registry=REGISTRY
                    )
                except ImportError:
                    pass
        except Exception:
            # 指标初始化失败，跳过
            pass

    def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行函数，应用熔断器逻辑"""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.half_open()
            else:
                raise CircuitBreakerOpenException(f"Circuit breaker '{self.name}' is OPEN")

        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            self.record_failure()
            raise e

    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return False
        return time.time() - self.last_failure_time >= self.recovery_timeout

    def record_success(self) -> None:
        """记录成功"""
        with self._lock:
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.close()
            elif self.state == CircuitBreakerState.CLOSED:
                self.failure_count = 0

            self._update_metrics()

    def record_failure(self) -> None:
        """记录失败"""
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            # 在半开状态下，任何失败都会导致熔断器重新打开
            if self.state == CircuitBreakerState.HALF_OPEN or self.failure_count >= self.failure_threshold:
                self.open()

            self._update_metrics()

    def close(self) -> None:
        """关闭熔断器"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None

    def open(self) -> None:
        """打开熔断器"""
        self.state = CircuitBreakerState.OPEN

    def half_open(self) -> None:
        """设置为半开状态"""
        self.state = CircuitBreakerState.HALF_OPEN

    def is_closed(self) -> bool:
        """检查是否闭合"""
        return self.state == CircuitBreakerState.CLOSED

    def is_open(self) -> bool:
        """检查是否打开"""
        return self.state == CircuitBreakerState.OPEN

    def is_half_open(self) -> bool:
        """检查是否半开"""
        return self.state == CircuitBreakerState.HALF_OPEN

    def call_permitted(self) -> bool:
        """检查是否允许调用"""
        return self.state != CircuitBreakerState.OPEN

    def _update_metrics(self) -> None:
        """更新指标"""
        try:
            if self.state_gauge:
                state_value = {
                    CircuitBreakerState.CLOSED: 0,
                    CircuitBreakerState.OPEN: 1,
                    CircuitBreakerState.HALF_OPEN: 2
                }.get(self.state, -1)
                self.state_gauge.labels(name=self.name).set(state_value)

            if self.failure_counter and self.failure_count > 0:
                self.failure_counter.labels(name=self.name).inc()
        except Exception:
            # 指标更新失败，跳过
            pass

    def get_status(self) -> Dict[str, Any]:
        """获取状态"""
        with self._lock:
            return {
                'name': self.name,
                'state': self.state.value,
                'failure_count': self.failure_count,
                'success_count': self.success_count,
                'failure_threshold': self.failure_threshold,
                'success_threshold': self.success_threshold,
                'recovery_timeout': self.recovery_timeout,
                'last_failure_time': self.last_failure_time,
                'time_since_last_failure': time.time() - (self.last_failure_time or 0)
            }

    def reset(self) -> None:
        """重置熔断器"""
        with self._lock:
            self.close()
            self._update_metrics()

    def trip(self) -> None:
        """触发熔断"""
        self.open()
        self._update_metrics()


class CircuitBreakerOpenException(Exception):
    """熔断器开启异常"""
