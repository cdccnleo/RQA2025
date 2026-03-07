
import threading
import time

from .shared_interfaces import ILogger, StandardLogger
from collections import deque
from typing import List, Optional
"""
指标存储器

职责：专门负责存储和查询指标数据
"""


class MetricsStorage:
    """
    指标存储器

    职责：专门负责存储和查询指标数据
    """

    def __init__(self, max_history: int = 1000, retention_period: int = 3600,
                 logger: Optional[ILogger] = None):
        self.max_history = max_history
        self.retention_period = retention_period
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
        self.metrics_history: deque = deque(maxlen=max_history)
        self._lock = threading.Lock()

    def store_metrics(self, metrics) -> None:
        """存储性能指标"""
        with self._lock:
            self.metrics_history.append(metrics)
            self._cleanup_expired_data()

    def get_current_metrics(self):
        """获取最新指标"""
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, limit: int = 100) -> List:
        """获取指标历史"""
        with self._lock:
            return list(self.metrics_history)[-limit:]

    def get_metrics_in_time_range(self, start_time: float, end_time: float) -> List:
        """获取指定时间范围内的指标"""
        with self._lock:
            return [m for m in self.metrics_history if start_time <= m.timestamp <= end_time]

    def _cleanup_expired_data(self) -> None:
        """清理过期数据"""
        current_time = time.time()
        cutoff_time = current_time - self.retention_period

        # 移除过期数据
        while self.metrics_history and self.metrics_history[0].timestamp < cutoff_time:
            self.metrics_history.popleft()

    def get_storage_stats(self) -> dict:
        """获取存储统计信息"""
        with self._lock:
            current_time = time.time()
            return {
                'total_metrics': len(self.metrics_history),
                'max_capacity': self.max_history,
                'retention_period': self.retention_period,
                'oldest_metric_age': current_time - (self.metrics_history[0].timestamp if self.metrics_history else current_time),
                'utilization_percent': (len(self.metrics_history) / self.max_history) * 100
            }
