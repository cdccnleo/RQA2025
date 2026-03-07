"""
application_monitor 模块

提供 application_monitor 相关功能和接口。
"""

import logging
from typing import Dict, Any, Optional, List, Deque

import psutil
import threading
import time

from collections import deque
"""
RQA2025 Application Monitor

Application monitoring and metrics collection service.
"""

logger = logging.getLogger(__name__)


class ApplicationMonitor:

    """Application monitoring service for tracking application metrics and health."""

    def __init__(self, app_name: str = "RQA2025"):
        """
        Initialize application monitor.

        Args:
            app_name: Name of the application being monitored
        """
        self.app_name: str = app_name
        self.metrics: Dict[str, Dict[str, Any]] = {}
        self.start_time: float = time.time()
        self.logger: logging.Logger = logging.getLogger(f"{self.__class__.__name__}.{app_name}")

        # 性能监控相关
        self.performance_history: Deque[Dict[str, Any]] = deque(maxlen=1000)  # 存储最近1000个性能数据点
        self.alert_thresholds: Dict[str, float] = {
            'cpu_percent': 80.0,
            'memory_percent': 85.0,
            'disk_usage_percent': 90.0
        }
        self.monitoring_active: bool = False
        self.monitor_thread = None

        # 初始化系统指标收集
        self._collect_system_info()

    def record_metric(self, name: str, value: Any, tags: Optional[Dict[str, str]] = None):
        """
        Record a metric.

        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
        """
        metric_data = {
            'value': value,
            'tags': tags or {},
            'timestamp': time.time(),
            'app_name': self.app_name
        }

        self.metrics[name] = metric_data
        self.logger.info(f"Recorded metric: {name} = {value}")

    def get_metric(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get a metric by name.

        Args:
            name: Metric name

        Returns:
            Metric data or None if not found
        """
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Get all metrics.

        Returns:
            Dictionary of all metrics
        """
        return self.metrics.copy()

    def get_recent_metrics(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent metrics as a list.

        Args:
            limit: Maximum number of metrics to return

        Returns:
            List of recent metrics with name and value
        """
        recent_metrics = []
        for name, metric_data in self.metrics.items():
            recent_metrics.append({
                'name': name,
                'value': metric_data.get('value'),
                'timestamp': metric_data.get('timestamp', time.time()),
                'tags': metric_data.get('tags', {})
            })

        # Sort by timestamp (most recent first) and limit
        recent_metrics.sort(key=lambda x: x['timestamp'], reverse=True)
        return recent_metrics[:limit]

    def clear_metrics(self):
        """Clear all metrics."""
        self.metrics.clear()
        self.logger.info("Cleared all metrics")

    def get_uptime(self) -> float:
        """
        Get application uptime in seconds.

        Returns:
            Uptime in seconds
        """
        return time.time() - self.start_time

    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check.

        Returns:
            Health check results
        """
        return {
            'status': 'healthy',
            'uptime': self.get_uptime(),
            'metrics_count': len(self.metrics),
            'app_name': self.app_name,
            'timestamp': time.time()
        }

    def _collect_system_info(self):
        """收集系统基本信息"""
        try:
            self.record_metric('system.cpu_count', psutil.cpu_count())
            self.record_metric('system.cpu_count_logical', psutil.cpu_count(logical=True))

            memory = psutil.virtual_memory()
            self.record_metric('system.memory.total', memory.total)

            disk = psutil.disk_usage('/')
            self.record_metric('system.disk.total', disk.total)

        except Exception as e:
            self.logger.warning(f"Failed to collect system info: {e}")

    def collect_performance_metrics(self):
        """收集当前性能指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('performance.cpu_percent', cpu_percent, {'unit': 'percent'})

            # 内存使用情况
            memory = psutil.virtual_memory()
            self.record_metric('performance.memory.percent', memory.percent, {'unit': 'percent'})
            self.record_metric('performance.memory.used', memory.used, {'unit': 'bytes'})
            self.record_metric('performance.memory.available', memory.available, {'unit': 'bytes'})

            # 磁盘使用情况
            disk = psutil.disk_usage('/')
            self.record_metric('performance.disk.percent', disk.percent, {'unit': 'percent'})
            self.record_metric('performance.disk.used', disk.used, {'unit': 'bytes'})
            self.record_metric('performance.disk.free', disk.free, {'unit': 'bytes'})

            # 网络IO
            network = psutil.net_io_counters()
            if network:
                self.record_metric('performance.network.bytes_sent',
                                   network.bytes_sent, {'unit': 'bytes'})
                self.record_metric('performance.network.bytes_recv',
                                   network.bytes_recv, {'unit': 'bytes'})

            # 存储性能数据点
            performance_data = {
                'timestamp': time.time(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'disk_percent': disk.percent
            }
            self.performance_history.append(performance_data)

            # 检查告警阈值
            self._check_alerts(performance_data)

        except Exception as e:
            self.logger.error(f"Failed to collect performance metrics: {e}")

    def _check_alerts(self, performance_data: Dict[str, Any]):
        """检查性能告警"""
        alerts = []

        if performance_data['cpu_percent'] > self.alert_thresholds['cpu_percent']:
            alerts.append(f"CPU使用率过高: {performance_data['cpu_percent']:.1f}%")

        if performance_data['memory_percent'] > self.alert_thresholds['memory_percent']:
            alerts.append(f"内存使用率过高: {performance_data['memory_percent']:.1f}%")

        if performance_data['disk_percent'] > self.alert_thresholds['disk_usage_percent']:
            alerts.append(f"磁盘使用率过高: {performance_data['disk_percent']:.1f}%")

        for alert in alerts:
            self.logger.warning(f"性能告警: {alert}")
            self.record_metric('alert.performance', alert, {'type': 'performance'})

    def start_monitoring(self, interval: int = 60):
        """启动性能监控"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, args=(interval,))
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        self.logger.info(f"Started performance monitoring with {interval}s interval")

    def stop_monitoring(self):
        """停止性能监控"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        self.logger.info("Stopped performance monitoring")

    def _monitoring_loop(self, interval: int):
        """监控循环"""
        while self.monitoring_active:
            self.collect_performance_metrics()
            time.sleep(interval)

    def get_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """获取性能摘要"""
        cutoff_time = time.time() - (hours * 3600)
        recent_data = [data for data in self.performance_history if data['timestamp'] > cutoff_time]

        if not recent_data:
            return {'error': 'No performance data available'}

        summary = {
            'time_range': f"{hours} hours",
            'data_points': len(recent_data),
            'cpu_percent': {
                'avg': sum(d['cpu_percent'] for d in recent_data) / len(recent_data),
                'max': max(d['cpu_percent'] for d in recent_data),
                'min': min(d['cpu_percent'] for d in recent_data)
            },
            'memory_percent': {
                'avg': sum(d['memory_percent'] for d in recent_data) / len(recent_data),
                'max': max(d['memory_percent'] for d in recent_data),
                'min': min(d['memory_percent'] for d in recent_data)
            },
            'disk_percent': {
                'avg': sum(d['disk_percent'] for d in recent_data) / len(recent_data),
                'max': max(d['disk_percent'] for d in recent_data),
                'min': min(d['disk_percent'] for d in recent_data)
            }
        }

        return summary

    def set_alert_threshold(self, metric: str, threshold: float):
        """设置告警阈值"""
        self.alert_thresholds[metric] = threshold
        self.logger.info(f"Set alert threshold for {metric}: {threshold}")

    def get_alert_thresholds(self) -> Dict[str, float]:
        """获取告警阈值"""
        return self.alert_thresholds.copy()
