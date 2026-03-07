
import threading
import time

from ..alert_dataclasses import PerformanceMetrics
from ..shared_interfaces import ILogger, StandardLogger
from typing import List, Optional
"""
指标收集器

职责：收集、存储和查询性能指标
"""


class CachedMetricsCollector:
    """
    缓存指标收集器

    职责：收集、缓存、存储和查询性能指标
    """

    def __init__(self, performance_monitor, logger: Optional[ILogger] = None):
        self.performance_monitor = performance_monitor
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
        self._lock = threading.Lock()
        self._cache = {}
        self._cache_ttl = 300  # 5分钟缓存

    def get_current_metrics(self) -> PerformanceMetrics:
        """获取当前指标"""
        return self.collect_metrics()

    def collect_metrics(self) -> PerformanceMetrics:
        """收集性能指标"""
        with self._lock:
            try:
                # 检查缓存
                current_time = time.time()
                if 'metrics' in self._cache:
                    cache_time, cached_metrics = self._cache['metrics']
                    if current_time - cache_time < self._cache_ttl:
                        return cached_metrics

                # 收集新指标
                if hasattr(self.performance_monitor, 'collect_metrics'):
                    metrics = self.performance_monitor.collect_metrics()
                else:
                    # 默认指标收集
                    metrics = PerformanceMetrics(
                        cpu_usage=0.0,
                        memory_usage=0.0,
                        disk_usage=0.0,
                        network_io=0.0,
                        timestamp=current_time
                    )

                # 更新缓存
                self._cache['metrics'] = (current_time, metrics)
                return metrics

            except Exception as e:
                self.logger.log_error(f"收集指标失败: {e}")
                return PerformanceMetrics(
                    cpu_usage=0.0,
                    memory_usage=0.0,
                    disk_usage=0.0,
                    network_io=0.0,
                    timestamp=time.time()
                )

    def get_metrics_history(self, minutes: int = 60) -> List[PerformanceMetrics]:
        """获取指标历史"""
        try:
            if hasattr(self.performance_monitor, 'get_metrics_history'):
                end_time = time.time()
                start_time = end_time - minutes * 60
                return self.performance_monitor.get_metrics_history(start_time, end_time)
            else:
                return []
        except Exception as e:
            self.logger.log_error(f"获取指标历史失败: {e}")
            return []

    def get_average_metrics(self, minutes: int = 60) -> PerformanceMetrics:
        """获取平均指标"""
        history = self.get_metrics_history(minutes)
        if not history:
            return self.get_current_metrics()

        # 计算平均值
        total_cpu = sum(m.cpu_usage for m in history)
        total_memory = sum(m.memory_usage for m in history)
        total_disk = sum(m.disk_usage for m in history)
        total_network = sum(m.network_io for m in history)

        count = len(history)
        return PerformanceMetrics(
            cpu_usage=total_cpu / count,
            memory_usage=total_memory / count,
            disk_usage=total_disk / count,
            network_io=total_network / count,
            timestamp=time.time()
        )
