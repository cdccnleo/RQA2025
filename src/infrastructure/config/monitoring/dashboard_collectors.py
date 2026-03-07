
import psutil

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import logging
import time, threading
from datetime import datetime
from .dashboard_models import (
    MetricValue, ConfigOperationStats, MetricType
)

"""
监控面板数据收集器

实现各种指标数据的收集和管理
"""
logger = logging.getLogger(__name__)


class MetricsCollector(ABC):
    """指标收集器基类"""

    def __init__(self, collection_interval: int = 15):
        self.collection_interval = collection_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._metrics: Dict[str, MetricValue] = {}
        self._lock = threading.RLock()

    @abstractmethod
    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""

    @abstractmethod
    def collect_config_metrics(self) -> Dict[str, Any]:
        """收集配置指标"""

    def start_collection(self):
        """启动指标收集"""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._collection_loop, daemon=True)
        self._thread.start()
        logger.info("指标收集器已启动")

    def stop_collection(self):
        """停止指标收集"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("指标收集器已停止")

    def _collection_loop(self):
        """收集循环"""
        while self._running:
            try:
                self._collect_all_metrics()
                time.sleep(self.collection_interval)
            except KeyboardInterrupt:
                logger.info("收到中断信号，停止收集")
                self._running = False
                break
            except Exception as e:
                logger.error(f"指标收集失败: {e}")
                try:
                    time.sleep(self.collection_interval)
                except KeyboardInterrupt:
                    logger.info("收到中断信号，停止收集")
                    self._running = False
                    break

    def _collect_all_metrics(self):
        """收集所有指标"""
        with self._lock:
            # 系统指标
            system_metrics = self.collect_system_metrics()
            for name, value in system_metrics.items():
                self._metrics[name] = MetricValue(
                    name=name,
                    value=value,
                    type=MetricType.GAUGE,
                    timestamp=datetime.now()
                )

            # 配置指标
            config_metrics = self.collect_config_metrics()
            for name, value in config_metrics.items():
                self._metrics[name] = MetricValue(
                    name=name,
                    value=value,
                    type=MetricType.COUNTER,
                    timestamp=datetime.now()
                )

    def get_metric(self, name: str) -> Optional[MetricValue]:
        """获取指定指标"""
        with self._lock:
            return self._metrics.get(name)

    def get_all_metrics(self) -> Dict[str, MetricValue]:
        """获取所有指标"""
        with self._lock:
            return self._metrics.copy()

    def get_metrics_by_type(self, metric_type: MetricType) -> List[MetricValue]:
        """按类型获取指标"""
        with self._lock:
            return [m for m in self._metrics.values() if m.type == metric_type]

    def remove_metric(self, name: str) -> bool:
        """移除指定指标"""
        with self._lock:
            if name in self._metrics:
                del self._metrics[name]
                return True
            return False


class InMemoryMetricsCollector(MetricsCollector):
    """内存指标收集器实现"""

    def __init__(self, collection_interval: int = 15):
        super().__init__(collection_interval)
        self._operation_stats: Dict[str, ConfigOperationStats] = {}
        self._custom_metrics: Dict[str, Any] = {}

    def collect_system_metrics(self) -> Dict[str, Any]:
        """收集系统指标"""
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent

            # 磁盘使用率
            disk_usage = {}
            for partition in psutil.disk_partitions():
                try:
                    usage = psutil.disk_usage(partition.mountpoint)
                    disk_usage[partition.mountpoint] = usage.percent
                except Exception as e:
                    pass

            # 网络IO
            network_io = psutil.net_io_counters()
            network_stats = {
                'bytes_sent': network_io.bytes_sent,
                'bytes_recv': network_io.bytes_recv,
                'packets_sent': network_io.packets_sent,
                'packets_recv': network_io.packets_recv
            }

            # 负载平均值 (Unix系统)
            load_average = None
            try:
                load_average = psutil.getloadavg()
            except Exception as e:
                pass

            return {
                'system.cpu.usage': cpu_percent,
                'system.memory.usage': memory_percent,
                'system.disk.usage': disk_usage,
                'system.network.io': network_stats,
                'system.load.average': load_average
            }

        except Exception as e:
            logger.error(f"系统指标收集失败: {e}")
            return {}

    def collect_config_metrics(self) -> Dict[str, Any]:
        """收集配置指标"""
        metrics = {}

        # 操作统计指标
        for operation, stats in self._operation_stats.items():
            metrics[f'config.operations.{operation}.count'] = stats.count
            metrics[f'config.operations.{operation}.success_rate'] = stats.get_success_rate()
            metrics[f'config.operations.{operation}.avg_time'] = stats.avg_time

        # 自定义指标
        for name, value in self._custom_metrics.items():
            metrics[f'config.custom.{name}'] = value

        return metrics

    def record_operation(self, operation: str, duration: float, success: bool,
                         metadata: Optional[Dict[str, Any]] = None):
        """记录操作"""
        if operation not in self._operation_stats:
            self._operation_stats[operation] = ConfigOperationStats(operation=operation)

        self._operation_stats[operation].add_metric(duration, success)

        # 记录为指标
        with self._lock:
            self._metrics[f'config.operation.{operation}.duration'] = MetricValue(
                name=f'config.operation.{operation}.duration',
                value=duration,
                type=MetricType.GAUGE,
                timestamp=datetime.now(),
                metadata=metadata
            )

    def add_custom_metric(self, name: str, value: Any, metric_type: MetricType = MetricType.GAUGE):
        """添加自定义指标"""
        self._custom_metrics[name] = value

        with self._lock:
            self._metrics[f'config.custom.{name}'] = MetricValue(
                name=name,  # 保持原始名称
                value=value if isinstance(value, (int, float)) else 0,
                type=metric_type,
                timestamp=datetime.now()
            )

    def get_operation_stats(self, operation: str) -> Optional[ConfigOperationStats]:
        """获取操作统计"""
        return self._operation_stats.get(operation)

    def get_all_operation_stats(self) -> Dict[str, ConfigOperationStats]:
        """获取所有操作统计"""
        return self._operation_stats.copy()




