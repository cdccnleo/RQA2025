"""
storage_monitor 模块

提供 storage_monitor 相关功能和接口。
"""

import os

import psutil
import threading
import time

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Dict, Any, Optional, List
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 基础设施层 - 存储监控组件

提供存储系统的实时监控、性能指标收集和告警功能。
集成到统一监控系统中，支持Prometheus指标导出。
"""


class StorageMetric(Enum):
    """存储监控指标类型"""

    TOTAL_SIZE = "total_size"
    USED_SIZE = "used_size"
    FREE_SIZE = "free_size"
    USAGE_PERCENT = "usage_percent"
    READ_COUNT = "read_count"
    WRITE_COUNT = "write_count"
    READ_BYTES = "read_bytes"
    WRITE_BYTES = "write_bytes"
    READ_TIME = "read_time"
    WRITE_TIME = "write_time"
    ERROR_COUNT = "error_count"


@dataclass
class StorageStats:
    """存储统计数据"""

    mount_point: str
    total_size: int
    used_size: int
    free_size: int
    usage_percent: float
    read_count: int = 0
    write_count: int = 0
    read_bytes: int = 0
    write_bytes: int = 0
    read_time: float = 0.0
    write_time: float = 0.0
    error_count: int = 0
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return asdict(self)


class StorageMonitor:
    """存储监控器"""

    def __init__(self, monitor_interval: float = 60.0):
        """
        初始化存储监控器

        Args:
            monitor_interval: 监控间隔（秒）
        """
        self.monitor_interval = monitor_interval
        self._stats_history: List[StorageStats] = []
        self._max_history_size = 1000
        self._lock = threading.RLock()
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._mount_points: List[str] = []

        # 手动统计计数器
        self._manual_read_count = 0
        self._manual_write_count = 0
        self._manual_read_bytes = 0
        self._manual_write_bytes = 0
        self._manual_read_time = 0.0
        self._manual_write_time = 0.0
        self._manual_error_count = 0

        # 初始化挂载点
        self._init_mount_points()

    def _init_mount_points(self):
        """初始化挂载点列表"""
        try:
            partitions = psutil.disk_partitions(all=False)
            self._mount_points = [p.mountpoint for p in partitions if os.path.exists(p.mountpoint)]
        except Exception:
            # 如果psutil不可用，使用基本的挂载点
            self._mount_points = ['/'] if os.name == 'posix' else ['C:\\']

    def start_monitoring(self):
        """启动监控"""
        with self._lock:
            if self._monitoring:
                return

            self._monitoring = True
            self._monitor_thread = threading.Thread(
                target=self._monitor_loop,
                daemon=True,
                name="StorageMonitor"
            )
            self._monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        with self._lock:
            self._monitoring = False
            if self._monitor_thread and self._monitor_thread.is_alive():
                self._monitor_thread.join(timeout=5.0)

    def _monitor_loop(self):
        """监控循环"""
        while self._monitoring:
            try:
                self._collect_stats()
            except Exception as e:
                print(f"存储监控错误: {e}")

            time.sleep(self.monitor_interval)

    def _collect_stats(self):
        """收集存储统计信息"""
        for mount_point in self._mount_points:
            try:
                stat = os.statvfs(mount_point) if os.name == 'posix' else None
                usage = psutil.disk_usage(mount_point)

                stats = StorageStats(
                    mount_point=mount_point,
                    total_size=usage.total,
                    used_size=usage.used,
                    free_size=usage.free,
                    usage_percent=usage.percent,
                    read_count=self._manual_read_count,
                    write_count=self._manual_write_count,
                    read_bytes=self._manual_read_bytes,
                    write_bytes=self._manual_write_bytes,
                    read_time=self._manual_read_time,
                    write_time=self._manual_write_time,
                    error_count=self._manual_error_count
                )

                with self._lock:
                    self._stats_history.append(stats)
                    if len(self._stats_history) > self._max_history_size:
                        self._stats_history.pop(0)

            except Exception:
                # 如果某个挂载点无法访问，跳过
                continue

    def record_operation(self, operation_type: str, size: int = 0,
                         duration: float = 0.0, success: bool = True):
        """
        记录存储操作

        Args:
            operation_type: 操作类型 ('read' 或 'write')
            size: 操作数据大小（字节）
            duration: 操作持续时间（秒）
            success: 是否成功
        """
        with self._lock:
            if operation_type == 'read':
                self._manual_read_count += 1
                self._manual_read_bytes += size
                self._manual_read_time += duration
            elif operation_type == 'write':
                self._manual_write_count += 1
                self._manual_write_bytes += size
                self._manual_write_time += duration

            if not success:
                self._manual_error_count += 1
    
    def record_write(self, size: int = 0, duration: float = 0.0):
        """记录写入操作的便捷方法"""
        self.record_operation('write', size=size, duration=duration, success=True)
    
    def record_error(self, symbol: str = ""):
        """记录错误的便捷方法"""
        with self._lock:
            self._manual_error_count += 1

    def get_current_stats(self) -> List[StorageStats]:
        """获取当前统计信息"""
        with self._lock:
            return self._stats_history[-len(self._mount_points):] if self._stats_history else []

    def get_stats_history(self, limit: int = 100) -> List[StorageStats]:
        """获取统计历史"""
        with self._lock:
            return self._stats_history[-limit:]

    def get_aggregated_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        with self._lock:
            if not self._stats_history:
                return {}

            # 计算每个挂载点的最新统计
            latest_stats = {}
            for stat in reversed(self._stats_history):
                if stat.mount_point not in latest_stats:
                    latest_stats[stat.mount_point] = stat

            # 聚合所有挂载点
            total_size = sum(s.total_size for s in latest_stats.values())
            used_size = sum(s.used_size for s in latest_stats.values())
            free_size = sum(s.free_size for s in latest_stats.values())

            return {
                "total_mount_points": len(latest_stats),
                "total_size": total_size,
                "used_size": used_size,
                "free_size": free_size,
                "usage_percent": (used_size / total_size * 100) if total_size > 0 else 0,
                "total_read_count": sum(s.read_count for s in latest_stats.values()),
                "total_write_count": sum(s.write_count for s in latest_stats.values()),
                "total_read_bytes": sum(s.read_bytes for s in latest_stats.values()),
                "total_write_bytes": sum(s.write_bytes for s in latest_stats.values()),
                "total_errors": sum(s.error_count for s in latest_stats.values()),
                "mount_points": list(latest_stats.keys()),
                "timestamp": time.time()
            }

    def get_metrics_for_prometheus(self) -> str:
        """获取Prometheus格式的指标"""
        stats = self.get_aggregated_stats()
        if not stats:
            return ""

        prometheus_lines = [
            "# HELP storage_total_size Total storage size in bytes",
            "# TYPE storage_total_size gauge",
            f"storage_total_size {stats.get('total_size', 0)}",
            "",
            "# HELP storage_used_size Used storage size in bytes",
            "# TYPE storage_used_size gauge",
            f"storage_used_size {stats.get('used_size', 0)}",
            "",
            "# HELP storage_usage_percent Storage usage percentage",
            "# TYPE storage_usage_percent gauge",
            f"storage_usage_percent {stats.get('usage_percent', 0)}",
            "",
            "# HELP storage_read_count Total read operations",
            "# TYPE storage_read_count counter",
            f"storage_read_count {stats.get('total_read_count', 0)}",
            "",
            "# HELP storage_write_count Total write operations",
            "# TYPE storage_write_count counter",
            f"storage_write_count {stats.get('total_write_count', 0)}",
            "",
            "# HELP storage_error_count Total storage errors",
            "# TYPE storage_error_count counter",
            f"storage_error_count {stats.get('total_errors', 0)}",
        ]

        return "\n".join(prometheus_lines)

    def reset_stats(self):
        """重置统计信息"""
        with self._lock:
            self._stats_history.clear()
            self._manual_read_count = 0
            self._manual_write_count = 0
            self._manual_read_bytes = 0
            self._manual_write_bytes = 0
            self._manual_read_time = 0.0
            self._manual_write_time = 0.0
            self._manual_error_count = 0

    def __del__(self):
        """析构函数"""
        self.stop_monitoring()
