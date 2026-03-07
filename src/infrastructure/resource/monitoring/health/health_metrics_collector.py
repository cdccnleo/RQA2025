
import psutil
import time
from datetime import datetime

from ...models.alert_dataclasses import AlertPerformanceMetrics as PerformanceMetrics
from ...core.shared_interfaces import ILogger, StandardLogger
from typing import Dict, List, Optional, Any
"""
健康指标收集器

职责：专门负责收集系统健康相关的各种指标
"""


class HealthMetricsCollector:
    """
    健康指标收集器

    职责：收集系统健康监控所需的各种指标数据
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[ILogger] = None):
        self.config = config or {}
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")

        # 指标缓存
        self._metrics_cache: Optional[PerformanceMetrics] = None
        self._cache_timestamp: float = 0
        self._cache_ttl = self.config.get('metrics_cache_ttl', 60)  # 60秒缓存

    def collect_current_metrics(self) -> Optional[PerformanceMetrics]:
        """
        收集当前系统性能指标

        Returns:
            PerformanceMetrics: 当前性能指标，如果收集失败返回None
        """
        try:
            # 检查缓存是否有效
            if self._is_cache_valid():
                return self._metrics_cache

            current_time = time.time()
            
            # 收集各种指标
            cpu_percent = self._collect_cpu_metrics()
            memory_percent = self._collect_memory_metrics()
            disk_percent = self._collect_disk_metrics()
            network_io = self._collect_network_metrics()
            process_count = self._collect_process_metrics()
            thread_count = self._collect_thread_metrics()

            # 创建指标对象
            metrics = self._create_performance_metrics(
                current_time, cpu_percent, memory_percent, disk_percent,
                network_io, process_count, thread_count
            )

            # 更新缓存
            self._update_cache(metrics, current_time)

            self.logger.log_debug("成功收集系统性能指标")
            return metrics

        except Exception as e:
            self.logger.log_error(f"收集系统性能指标失败: {e}")
            return None

    def _is_cache_valid(self) -> bool:
        """检查缓存是否有效"""
        current_time = time.time()
        return (self._metrics_cache and
                current_time - self._cache_timestamp < self._cache_ttl)

    def _collect_cpu_metrics(self) -> float:
        """收集CPU指标"""
        try:
            return psutil.cpu_percent(interval=1)
        except Exception as e:
            self.logger.log_warning(f"收集CPU指标失败: {e}")
            return 0.0

    def _collect_memory_metrics(self) -> float:
        """收集内存指标"""
        try:
            memory = psutil.virtual_memory()
            return memory.percent
        except Exception as e:
            self.logger.log_warning(f"收集内存指标失败: {e}")
            return 0.0

    def _collect_disk_metrics(self) -> float:
        """收集磁盘指标"""
        try:
            disk = psutil.disk_usage('/')
            return disk.percent
        except Exception as e:
            self.logger.log_warning(f"收集磁盘指标失败: {e}")
            return 0.0

    def _collect_network_metrics(self) -> Dict[str, int]:
        """收集网络指标"""
        try:
            net_io = psutil.net_io_counters()
            return {
                'bytes_sent': net_io.bytes_sent,
                'bytes_recv': net_io.bytes_recv,
                'packets_sent': net_io.packets_sent,
                'packets_recv': net_io.packets_recv
            }
        except Exception as e:
            self.logger.log_warning(f"收集网络指标失败: {e}")
            return {'bytes_sent': 0, 'bytes_recv': 0,
                    'packets_sent': 0, 'packets_recv': 0}

    def _collect_process_metrics(self) -> int:
        """收集进程指标"""
        try:
            return len(psutil.pids())
        except Exception as e:
            self.logger.log_warning(f"收集进程指标失败: {e}")
            return 0

    def _collect_thread_metrics(self) -> int:
        """收集线程指标"""
        try:
            current_process = psutil.Process()
            return current_process.num_threads()
        except Exception as e:
            self.logger.log_warning(f"收集线程指标失败: {e}")
            return 0

    def _create_performance_metrics(self, timestamp: float, cpu_percent: float,
                                  memory_percent: float, disk_percent: float,
                                  network_io: Dict[str, int], process_count: int,
                                  thread_count: int) -> PerformanceMetrics:
        """创建性能指标对象"""
        return PerformanceMetrics(
            timestamp=datetime.fromtimestamp(timestamp),
            cpu_usage=cpu_percent,
            memory_usage=memory_percent,
            disk_usage=disk_percent,
            network_latency=0.0,  # 默认值
            test_execution_time=0.0,  # 默认值
            test_success_rate=1.0,  # 默认值
            active_threads=thread_count
        )

    def _update_cache(self, metrics: PerformanceMetrics, timestamp: float) -> None:
        """更新缓存"""
        self._metrics_cache = metrics
        self._cache_timestamp = timestamp

    def collect_historical_metrics(self, hours: int = 24) -> List[PerformanceMetrics]:
        """
        收集历史性能指标

        Args:
            hours: 收集过去多少小时的数据

        Returns:
            List[PerformanceMetrics]: 历史性能指标列表
        """
        # 注意：这个方法需要与实际的指标存储系统集成
        # 这里返回空列表作为占位符
        self.logger.log_warning("历史指标收集功能需要与指标存储系统集成")
        return []

    def get_metrics_summary(self, hours: int = 1) -> Dict[str, Any]:
        """
        获取指标汇总信息

        Args:
            hours: 汇总时间范围（小时）

        Returns:
            Dict[str, Any]: 指标汇总数据
        """
        try:
            current_metrics = self.collect_current_metrics()
            if not current_metrics:
                return self._get_empty_summary()

            # 这里可以扩展为更复杂的汇总逻辑
            return {
                'current': {
                    'cpu_percent': current_metrics.cpu_percent,
                    'memory_percent': current_metrics.memory_percent,
                    'disk_usage': current_metrics.disk_usage,
                    'process_count': current_metrics.process_count,
                    'thread_count': current_metrics.thread_count,
                    'timestamp': current_metrics.timestamp
                },
                'summary_period_hours': hours,
                'collection_timestamp': time.time()
            }

        except Exception as e:
            self.logger.log_error(f"获取指标汇总失败: {e}")
            return self._get_empty_summary()

    def _get_empty_summary(self) -> Dict[str, Any]:
        """获取空的汇总数据"""
        return {
            'current': {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_usage': 0.0,
                'process_count': 0,
                'thread_count': 0,
                'timestamp': time.time()
            },
            'summary_period_hours': 0,
            'collection_timestamp': time.time()
        }

    def clear_cache(self) -> None:
        """清除指标缓存"""
        self._metrics_cache = None
        self._cache_timestamp = 0
        self.logger.log_debug("已清除指标缓存")
