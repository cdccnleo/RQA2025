"""
Circuit Breaker Module
熔断器模块

This module provides circuit breaker pattern implementation for async operations
此模块为异步操作提供熔断器模式实现

Author: RQA2025 Development Team
Date: 2025 - 01 - 28
"""

import logging
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime, timedelta
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):

    """Circuit breaker states"""
    CLOSED = "closed"        # Normal operation
    OPEN = "open"           # Circuit is open, failing fast
    HALF_OPEN = "half_open"  # Testing if service has recovered


class CircuitBreakerConfig:

    """
    Circuit Breaker Configuration Class
    熔断器配置类

    Configuration for circuit breaker behavior
    熔断器行为的配置
    """

    def __init__(self,


                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: Optional[List[type]] = None,
                 success_threshold: int = 3,
                 timeout: float = 30.0):
        """
        Initialize circuit breaker configuration
        初始化熔断器配置

        Args:
            failure_threshold: Number of consecutive failures to open circuit
                              打开电路的连续失败次数
            recovery_timeout: Time to wait before attempting recovery (seconds)
                             尝试恢复前等待的时间（秒）
            expected_exception: List of exceptions that count as failures
                               计为失败的异常列表
            success_threshold: Number of consecutive successes to close circuit in half - open state
                              在半开状态下关闭电路的连续成功次数
            timeout: Timeout for individual operations (seconds)
                    单个操作的超时时间（秒）
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception or [Exception]
        self.success_threshold = success_threshold
        self.timeout = timeout


class CircuitBreakerResult:

    """
    Circuit Breaker Result Class
    熔断器结果类

    Contains information about circuit breaker operation result
    包含熔断器操作结果的信息
    """

    def __init__(self,


                 success: bool,
                 result: Any = None,
                 circuit_state: CircuitBreakerState = CircuitBreakerState.CLOSED,
                 error: Optional[Exception] = None):
        """
        Initialize circuit breaker result
        初始化熔断器结果

        Args:
            success: Whether the operation succeeded
                    操作是否成功
            result: Result of the operation if successful
                   如果成功则为操作结果
            circuit_state: State of the circuit breaker after operation
                          操作后熔断器的状态
            error: Exception if operation failed
                  如果操作失败则为异常
        """
        self.success = success
        self.result = result
        self.circuit_state = circuit_state
        self.error = error
        self.timestamp = datetime.now()

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
            'circuit_state': self.circuit_state.value,
            'error': str(self.error) if self.error else None,
            'timestamp': self.timestamp.isoformat()
        }


class CircuitBreaker:

    """
    Circuit Breaker Class
    熔断器类

    Implements the circuit breaker pattern to prevent cascading failures
    实现熔断器模式以防止级联故障
    """

    def __init__(self,


                 name: str = "default_circuit_breaker",
                 config: Optional[CircuitBreakerConfig] = None):
        """
        Initialize circuit breaker
        初始化熔断器

        Args:
            name: Name of the circuit breaker
                熔断器的名称
            config: Circuit breaker configuration
                   熔断器配置
        """
        self.name = name
        self.config = config or CircuitBreakerConfig()

        # State management
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.next_attempt_time: Optional[datetime] = None

        # Statistics
        self.call_count = 0
        self.success_count_total = 0
        self.failure_count_total = 0

        # Thread safety
        self._lock = threading.Lock()

        logger.info(f"Circuit breaker {name} initialized")

    def call(self,


             func: Callable,
             *args,
             fallback: Optional[Callable] = None,
             **kwargs) -> CircuitBreakerResult:
        """
        Execute function through circuit breaker
        通过熔断器执行函数

        Args:
            func: Function to execute
                 要执行的函数
            *args: Positional arguments
                  位置参数
            fallback: Fallback function if circuit is open
                     如果电路打开则为后备函数
            **kwargs: Keyword arguments
                     关键字参数

        Returns:
            CircuitBreakerResult: Result of the operation
                                 操作结果
        """
        with self._lock:
            self.call_count += 1

            # Check if circuit should be closed (recovery timeout passed)
            if self.state == CircuitBreakerState.OPEN:
                if self.next_attempt_time and datetime.now() >= self.next_attempt_time:
                    self.state = CircuitBreakerState.HALF_OPEN
                    logger.info(f"Circuit breaker {self.name} moved to HALF_OPEN state")
                else:
                    # Circuit is open, use fallback or fail fast
                    if fallback:
                        try:
                            result = fallback(*args, **kwargs)
                            return CircuitBreakerResult(
                                success=True,
                                result=result,
                                circuit_state=self.state
                            )
                        except Exception as e:
                            logger.error(f"Fallback function failed for {self.name}: {str(e)}")

                    return CircuitBreakerResult(
                        success=False,
                        circuit_state=self.state,
                        error=Exception(f"Circuit breaker {self.name} is OPEN")
                    )

            # Execute the function
            try:
                if self.config.timeout > 0:
                    result = self._execute_with_timeout(func, args, kwargs, self.config.timeout)
                else:
                    result = func(*args, **kwargs)

                # Success
                self._handle_success()
                self.success_count_total += 1

                return CircuitBreakerResult(
                    success=True,
                    result=result,
                    circuit_state=self.state
                )

            except Exception as e:
                # Check if this exception should count as a failure
                if self._is_expected_exception(e):
                    self._handle_failure()
                    self.failure_count_total += 1

                    # If circuit just opened, try fallback
                    if self.state == CircuitBreakerState.OPEN and fallback:
                        try:
                            result = fallback(*args, **kwargs)
                            return CircuitBreakerResult(
                                success=True,
                                result=result,
                                circuit_state=self.state
                            )
                        except Exception as fallback_error:
                            logger.error(
                                f"Fallback function failed for {self.name}: {str(fallback_error)}")

                return CircuitBreakerResult(
                    success=False,
                    circuit_state=self.state,
                    error=e
                )

    def _execute_with_timeout(self,


                              func: Callable,
                              args: tuple,
                              kwargs: Dict[str, Any],
                              timeout: float) -> Any:
        """
        Execute function with timeout
        使用超时执行函数

        Args:
            func: Function to execute
                 要执行的函数
            args: Positional arguments
                 位置参数
            kwargs: Keyword arguments
                  关键字参数
            timeout: Timeout in seconds
                    超时秒数

        Returns:
            Function result
            函数结果

        Raises:
            TimeoutError: If execution exceeds timeout
                         如果执行超过超时
        """
        import signal

        def timeout_handler(signum, frame):

            raise TimeoutError(f"Function execution exceeded {timeout} seconds")

        # Set up signal handler
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(int(timeout))

        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Restore original handler
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

    def _is_expected_exception(self, exception: Exception) -> bool:
        """
        Check if exception is one that should trigger circuit breaker
        检查异常是否应该触发熔断器

        Args:
            exception: Exception to check
                      要检查的异常

        Returns:
            bool: True if exception should trigger breaker
                  如果异常应该触发断路器则返回True
        """
        return any(isinstance(exception, exc_type) for exc_type in self.config.expected_exception)

    def _handle_success(self) -> None:
        """Handle successful operation"""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.success_count += 1
            if self.success_count >= self.config.success_threshold:
                self._close_circuit()
        else:
            # Reset failure count on success in closed state
            self.failure_count = 0

    def _handle_failure(self) -> None:
        """Handle failed operation"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()

        if self.state == CircuitBreakerState.HALF_OPEN:
            # Failed in half - open state, go back to open
            self._open_circuit()
        elif self.state == CircuitBreakerState.CLOSED:
            if self.failure_count >= self.config.failure_threshold:
                self._open_circuit()

    def _open_circuit(self) -> None:
        """Open the circuit breaker"""
        self.state = CircuitBreakerState.OPEN
        self.next_attempt_time = datetime.now() + timedelta(seconds=self.config.recovery_timeout)
        logger.warning(
            f"Circuit breaker {self.name} opened due to {self.failure_count} consecutive failures")

    def _close_circuit(self) -> None:
        """Close the circuit breaker"""
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.next_attempt_time = None
        logger.info(f"Circuit breaker {self.name} closed - service recovered")

    def get_status(self) -> Dict[str, Any]:
        """
        Get circuit breaker status
        获取熔断器状态

        Returns:
            dict: Status information
                  状态信息
        """
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'call_count': self.call_count,
            'success_count_total': self.success_count_total,
            'failure_count_total': self.failure_count_total,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'next_attempt_time': self.next_attempt_time.isoformat() if self.next_attempt_time else None,
            'config': {
                'failure_threshold': self.config.failure_threshold,
                'recovery_timeout': self.config.recovery_timeout,
                'success_threshold': self.config.success_threshold,
                'timeout': self.config.timeout
            }
        }

    def force_open(self) -> None:
        """
        Force the circuit breaker to open
        强制熔断器打开
        """
        with self._lock:
            self._open_circuit()

    def force_close(self) -> None:
        """
        Force the circuit breaker to close
        强制熔断器关闭
        """
        with self._lock:
            self._close_circuit()

    def reset(self) -> None:
        """
        Reset the circuit breaker to initial state
        将熔断器重置为初始状态
        """
        with self._lock:
            self.state = CircuitBreakerState.CLOSED
            self.failure_count = 0
            self.success_count = 0
            self.call_count = 0
            self.success_count_total = 0
            self.failure_count_total = 0
            self.last_failure_time = None
            self.next_attempt_time = None
            logger.info(f"Circuit breaker {self.name} reset")


def circuit_breaker(name: str = "default",


                    config: Optional[CircuitBreakerConfig] = None):
    """
    Decorator to apply circuit breaker pattern to functions
    为函数应用熔断器模式的装饰器

    Args:
        name: Name of the circuit breaker
             熔断器的名称
        config: Circuit breaker configuration
               熔断器配置

    Returns:
        Decorator function
        装饰器函数
    """
    breaker = CircuitBreaker(name, config)

    def decorator(func: Callable) -> Callable:

        def wrapper(*args, **kwargs):

            result = breaker.call(func, *args, **kwargs)
            if result.success:
                return result.result
            else:
                raise result.error or Exception(f"Circuit breaker {name} is {breaker.state.value}")

        return wrapper

    return decorator


# Global circuit breaker instance
# 全局熔断器实例
circuit_breaker_instance = CircuitBreaker("global_circuit_breaker")

__all__ = [
    'CircuitBreakerState',
    'CircuitBreakerConfig',
    'CircuitBreakerResult',
    'CircuitBreaker',
    'circuit_breaker',
    'circuit_breaker_instance'
]
