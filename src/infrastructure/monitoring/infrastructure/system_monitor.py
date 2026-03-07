
import psutil
import threading
import time

from dataclasses import dataclass
from typing import Dict, Optional, List
#!/usr/bin/env python3
"""
RQA2025 系统监控组件

提供系统层面的监控和指标收集功能。
"""


@dataclass
class SystemMetrics:
    """系统指标数据"""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_connections: int
    timestamp: float


class SystemMonitor:
    """系统监控器"""

    def __init__(self):
        self.metrics_history: List[SystemMetrics] = []
        self.max_history_size = 1000
        self._lock = threading.RLock()

    def collect_system_metrics(self) -> SystemMetrics:
        """
        收集系统指标

        Returns:
            SystemMetrics: 系统指标数据
        """
        # 默认值
        cpu_percent = 0.0
        memory_percent = 0.0
        disk_usage_percent = 0.0
        network_connections = 0

        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=0.1)
        except Exception:
            pass  # 使用默认值

        try:
            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
        except Exception:
            pass  # 使用默认值

        try:
            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_usage_percent = disk.percent
        except Exception:
            pass  # 使用默认值

        try:
            # 网络连接数
            network_connections = len(psutil.net_connections())
        except Exception:
            pass  # 使用默认值

        metrics = SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            disk_usage_percent=disk_usage_percent,
            network_connections=network_connections,
            timestamp=time.time()
        )

        with self._lock:
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history.pop(0)

        return metrics

    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """
        获取当前指标

        Returns:
            SystemMetrics: 最新的系统指标
        """
        with self._lock:
            return self.metrics_history[-1] if self.metrics_history else None

    def get_metrics_history(self, limit: int = 100) -> List[SystemMetrics]:
        """
        获取指标历史

        Args:
            limit: 返回的最大条目数

        Returns:
            List[SystemMetrics]: 指标历史列表
        """
        with self._lock:
            return self.metrics_history[-limit:]

    def get_average_metrics(self, time_window: int = 60) -> Dict[str, float]:
        """
        获取指定时间窗口内的平均指标

        Args:
            time_window: 时间窗口（秒）

        Returns:
            Dict[str, float]: 平均指标
        """
        current_time = time.time()
        cutoff_time = current_time - time_window

        with self._lock:
            recent_metrics = [
                m for m in self.metrics_history
                if m.timestamp >= cutoff_time
            ]

        if not recent_metrics:
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'disk_usage_percent': 0.0,
                'network_connections': 0
            }

        return {
            'cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'disk_usage_percent': sum(m.disk_usage_percent for m in recent_metrics) / len(recent_metrics),
            'network_connections': sum(m.network_connections for m in recent_metrics) / len(recent_metrics)
        }

    def check_thresholds(self, thresholds: Optional[Dict[str, float]] = None) -> Dict[str, bool]:
        """
        检查是否超过阈值

        Args:
            thresholds: 阈值字典

        Returns:
            Dict[str, bool]: 阈值检查结果
        """
        if thresholds is None:
            thresholds = {
                'cpu_percent': 80.0,
                'memory_percent': 85.0,
                'disk_usage_percent': 90.0
            }

        current = self.get_current_metrics()
        if not current:
            return {key: False for key in thresholds.keys()}

        return {
            'cpu_percent': current.cpu_percent > thresholds.get('cpu_percent', 80.0),
            'memory_percent': current.memory_percent > thresholds.get('memory_percent', 85.0),
            'disk_usage_percent': current.disk_usage_percent > thresholds.get('disk_usage_percent', 90.0)
        }

    def clear_history(self) -> None:
        """清空指标历史"""
        with self._lock:
            self.metrics_history.clear()

    def monitor_system(self) -> SystemMetrics:
        """
        兼容接口：执行一次系统监控。
        """
        return self.collect_system_metrics()