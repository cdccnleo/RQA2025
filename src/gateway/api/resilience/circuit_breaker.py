"""熔断器模块"""

import logging
import threading
from typing import Callable, Any
from datetime import datetime

logger = logging.getLogger(__name__)


class CircuitBreaker:
    """熔断器
    
    实现熔断器模式，防止级联故障
    
    状态转换:
    - CLOSED: 正常状态，允许请求
    - OPEN: 熔断状态，拒绝请求
    - HALF_OPEN: 半开状态，尝试恢复
    """
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60,
                 success_threshold: int = 3):
        """初始化熔断器
        
        Args:
            failure_threshold: 失败阈值，超过此值触发熔断
            recovery_timeout: 恢复超时（秒），熔断后等待时间
            success_threshold: 成功阈值，半开状态需要的连续成功次数
        """
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.success_threshold = success_threshold
        
        self.failure_count = 0
        self.success_count = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.last_failure_time = None
        self.lock = threading.Lock()
    
    def call(self, func: Callable) -> Any:
        """执行带熔断器的函数
        
        Args:
            func: 要执行的函数
            
        Returns:
            函数执行结果
            
        Raises:
            Exception: 如果熔断器打开或函数执行失败
        """
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.success_count = 0
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func()
                self._on_success()
                return result
            except Exception as e:
                self._on_failure()
                raise e
    
    def _should_attempt_reset(self) -> bool:
        """检查是否应该尝试重置"""
        if self.last_failure_time is None:
            return True
        time_diff = datetime.now() - self.last_failure_time
        return time_diff.total_seconds() >= self.recovery_timeout
    
    def _on_success(self):
        """成功回调"""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
                logger.info("Circuit breaker reset to CLOSED")
    
    def _on_failure(self):
        """失败回调"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def record_failure(self):
        """记录失败"""
        with self.lock:
            self._on_failure()
    
    def record_success(self):
        """记录成功"""
        with self.lock:
            self._on_success()
    
    def can_attempt(self) -> bool:
        """检查是否可以尝试请求
        
        Returns:
            如果可以尝试返回True，否则返回False
        """
        with self.lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    logger.info("Circuit breaker attempting recovery - state changed to HALF_OPEN")
                    return True
                return False
            return True
    
    def get_state(self) -> str:
        """获取当前状态"""
        return self.state
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None
        }


__all__ = ['CircuitBreaker']

