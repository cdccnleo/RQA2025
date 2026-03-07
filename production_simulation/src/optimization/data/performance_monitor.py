#!/usr/bin/env python3
"""
数据性能监控器

提供数据加载和处理性能的实时监控功能。
"""

# 使用基础设施层日志，避免依赖上层组件
try:
    from src.infrastructure.logging import get_infrastructure_logger
except ImportError:
    # 降级到标准logging
    import logging

    def get_infrastructure_logger(name):

        logger = logging.getLogger(name)
        logger.warning("无法导入基础设施层日志，使用标准logging")
        return logger

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import json

logger = get_infrastructure_logger('data_performance_monitor')


@dataclass
class PerformanceMetric:

    """性能指标数据类"""
    operation: str
    duration_ms: float
    timestamp: datetime
    success: bool
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:

    """系统指标数据类"""
    cpu_percent: float
    memory_percent: float
    memory_used_mb: float
    disk_io_read_mb: float
    disk_io_write_mb: float
    network_io_sent_mb: float
    network_io_recv_mb: float
    timestamp: datetime


@dataclass
class PerformanceAlert:

    """性能告警数据类"""
    alert_id: str
    alert_type: str
    severity: str
    message: str
    threshold: float
    current_value: float
    timestamp: datetime
    status: str = "active"


class DataPerformanceMonitor:

    """
    数据性能监控器

    提供数据加载和处理性能的实时监控功能。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化性能监控器

        Args:
            config: 监控配置
        """
        self.config = config or {}

        # 性能指标存储
        self.metrics: List[PerformanceMetric] = []
        self.system_metrics: List[SystemMetrics] = []

        # 告警配置
        self.alert_thresholds = {
            'load_time_ms': 5000,  # 5秒
            'memory_percent': 80,   # 80%
            'cpu_percent': 90,      # 90%
            'error_rate': 0.1       # 10%
        }

        # 告警列表
        self.alerts: List[PerformanceAlert] = []

        # 统计信息
        self.stats = {
            'total_operations': 0,
            'successful_operations': 0,
            'failed_operations': 0,
            'avg_load_time_ms': 0,
            'max_load_time_ms': 0,
            'min_load_time_ms': float('inf')
        }

        # 监控线程
        self.monitoring_thread = None
        self.stop_monitoring = False

        # 回调函数
        self.alert_callbacks: List[Callable] = []

        logger.info("DataPerformanceMonitor initialized")

    def start_monitoring(self, interval_seconds: int = 30):
        """
        开始系统监控

        Args:
            interval_seconds: 监控间隔（秒）
        """
        if self.monitoring_thread is not None:
            logger.warning("Monitoring is already running")
            return

        self.stop_monitoring = False
        self.monitoring_thread = threading.Thread(
            target=self._monitor_system,
            args=(interval_seconds,),
            daemon=True
        )
        self.monitoring_thread.start()
        logger.info(f"Started system monitoring with {interval_seconds}s interval")

    def stop_monitoring(self):
        """停止系统监控"""
        self.stop_monitoring = True
        if self.monitoring_thread:
            self.monitoring_thread.join()
            self.monitoring_thread = None
        logger.info("Stopped system monitoring")

    def _monitor_system(self, interval_seconds: int):
        """系统监控线程"""
        while not self.stop_monitoring:
            try:
                # 收集系统指标
                system_metrics = self._collect_system_metrics()
                self.system_metrics.append(system_metrics)

                # 检查系统告警
                self._check_system_alerts(system_metrics)

                # 清理旧数据
                self._cleanup_old_metrics()

                time.sleep(interval_seconds)

            except Exception as e:
                logger.error(f"System monitoring error: {str(e)}")
                time.sleep(interval_seconds)

    def _collect_system_metrics(self) -> SystemMetrics:
        """收集系统指标"""
        try:
            # CPU使用率
            cpu_percent = psutil.cpu_percent(interval=1)

            # 内存使用情况
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)

            # 磁盘IO
            disk_io = psutil.disk_io_counters()
            disk_io_read_mb = disk_io.read_bytes / (1024 * 1024) if disk_io else 0
            disk_io_write_mb = disk_io.write_bytes / (1024 * 1024) if disk_io else 0

            # 网络IO
            network_io = psutil.net_io_counters()
            network_io_sent_mb = network_io.bytes_sent / (1024 * 1024) if network_io else 0
            network_io_recv_mb = network_io.bytes_recv / (1024 * 1024) if network_io else 0

            return SystemMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_used_mb=memory_used_mb,
                disk_io_read_mb=disk_io_read_mb,
                disk_io_write_mb=disk_io_write_mb,
                network_io_sent_mb=network_io_sent_mb,
                network_io_recv_mb=network_io_recv_mb,
                timestamp=datetime.now()
            )

        except Exception as e:
            logger.error(f"Failed to collect system metrics: {str(e)}")
            return SystemMetrics(
                cpu_percent=0,
                memory_percent=0,
                memory_used_mb=0,
                disk_io_read_mb=0,
                disk_io_write_mb=0,
                network_io_sent_mb=0,
                network_io_recv_mb=0,
                timestamp=datetime.now()
            )

    def _check_system_alerts(self, metrics: SystemMetrics):
        """检查系统告警"""
        # CPU告警
        if metrics.cpu_percent > self.alert_thresholds['cpu_percent']:
            self._create_alert(
                'high_cpu_usage',
                'warning',
                f"CPU usage is high: {metrics.cpu_percent:.1f}%",
                self.alert_thresholds['cpu_percent'],
                metrics.cpu_percent
            )

        # 内存告警
        if metrics.memory_percent > self.alert_thresholds['memory_percent']:
            self._create_alert(
                'high_memory_usage',
                'warning',
                f"Memory usage is high: {metrics.memory_percent:.1f}%",
                self.alert_thresholds['memory_percent'],
                metrics.memory_percent
            )

    def _create_alert(self, alert_type: str, severity: str, message: str,


                      threshold: float, current_value: float):
        """创建告警"""
        alert = PerformanceAlert(
            alert_id=f"{alert_type}_{datetime.now().timestamp()}",
            alert_type=alert_type,
            severity=severity,
            message=message,
            threshold=threshold,
            current_value=current_value,
            timestamp=datetime.now()
        )

        self.alerts.append(alert)

        # 调用回调函数
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {str(e)}")

        logger.warning(f"Performance alert: {message}")

    def record_operation(self, operation: str, duration_ms: float, success: bool,


                         error_message: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None):
        """
        记录操作性能指标

        Args:
            operation: 操作名称
            duration_ms: 执行时间（毫秒）
            success: 是否成功
            error_message: 错误信息
            metadata: 元数据
        """
        metric = PerformanceMetric(
            operation=operation,
            duration_ms=duration_ms,
            timestamp=datetime.now(),
            success=success,
            error_message=error_message,
            metadata=metadata or {}
        )

        self.metrics.append(metric)

        # 更新统计信息
        self.stats['total_operations'] += 1
        if success:
            self.stats['successful_operations'] += 1
        else:
            self.stats['failed_operations'] += 1

        # 更新平均加载时间
        total_ops = self.stats['total_operations']
        current_avg = self.stats['avg_load_time_ms']
        self.stats['avg_load_time_ms'] = (
            (current_avg * (total_ops - 1) + duration_ms) / total_ops
        )

        # 更新最大 / 最小加载时间
        self.stats['max_load_time_ms'] = max(self.stats['max_load_time_ms'], duration_ms)
        self.stats['min_load_time_ms'] = min(self.stats['min_load_time_ms'], duration_ms)

        # 检查性能告警
        if duration_ms > self.alert_thresholds['load_time_ms']:
            self._create_alert(
                'slow_operation',
                'warning',
                f"Operation {operation} is slow: {duration_ms:.2f}ms",
                self.alert_thresholds['load_time_ms'],
                duration_ms
            )

        # 检查错误率告警
        error_rate = self.stats['failed_operations'] / self.stats['total_operations']
        if error_rate > self.alert_thresholds['error_rate']:
            self._create_alert(
                'high_error_rate',
                'error',
                f"High error rate: {error_rate:.2%}",
                self.alert_thresholds['error_rate'],
                error_rate
            )

    def get_performance_report(self, hours: int = 24) -> Dict[str, Any]:
        """
        获取性能报告

        Args:
            hours: 报告时间范围（小时）

        Returns:
            Dict[str, Any]: 性能报告
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)

        # 过滤时间范围内的指标
        recent_metrics = [
            m for m in self.metrics
            if m.timestamp >= cutoff_time
        ]

        recent_system_metrics = [
            m for m in self.system_metrics
            if m.timestamp >= cutoff_time
        ]

        # 按操作类型分组
        operation_stats = defaultdict(list)
        for metric in recent_metrics:
            operation_stats[metric.operation].append(metric.duration_ms)

        # 计算各操作的统计信息
        operation_summary = {}
        for operation, durations in operation_stats.items():
            operation_summary[operation] = {
                'count': len(durations),
                'avg_duration_ms': sum(durations) / len(durations),
                'max_duration_ms': max(durations),
                'min_duration_ms': min(durations),
                'success_rate': len([m for m in recent_metrics
                                     if m.operation == operation and m.success]) / len(durations)
            }

        # 系统指标统计
        if recent_system_metrics:
            avg_cpu = sum(m.cpu_percent for m in recent_system_metrics) / len(recent_system_metrics)
            avg_memory = sum(m.memory_percent for m in recent_system_metrics) / \
                len(recent_system_metrics)
            max_cpu = max(m.cpu_percent for m in recent_system_metrics)
            max_memory = max(m.memory_percent for m in recent_system_metrics)
        else:
            avg_cpu = avg_memory = max_cpu = max_memory = 0

        return {
            'period_hours': hours,
            'total_operations': len(recent_metrics),
            'successful_operations': len([m for m in recent_metrics if m.success]),
            'failed_operations': len([m for m in recent_metrics if not m.success]),
            'operation_summary': operation_summary,
            'system_metrics': {
                'avg_cpu_percent': avg_cpu,
                'avg_memory_percent': avg_memory,
                'max_cpu_percent': max_cpu,
                'max_memory_percent': max_memory
            },
            'active_alerts': len([a for a in self.alerts if a.status == "active"]),
            'overall_stats': self.stats
        }

    def add_alert_callback(self, callback: Callable[[PerformanceAlert], None]):
        """
        添加告警回调函数
        Args:
            callback: 回调函数
        """
        self.alert_callbacks.append(callback)

    def _cleanup_old_metrics(self, max_age_hours: int = 24):
        """清理旧的性能指标"""
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)

        # 清理性能指标
        self.metrics = [m for m in self.metrics if m.timestamp >= cutoff_time]

        # 清理系统指标
        self.system_metrics = [m for m in self.system_metrics if m.timestamp >= cutoff_time]

        # 清理已解决的告警
        self.alerts = [a for a in self.alerts
                       if a.status == "active" or a.timestamp >= cutoff_time]

    def export_metrics(self, filepath: str, format: str = "json"):
        """
        导出性能指标

        Args:
            filepath: 文件路径
            format: 导出格式（json / csv）
        """
        try:
            if format.lower() == "json":
                data = {
                    'metrics': [
                        {
                            'operation': m.operation,
                            'duration_ms': m.duration_ms,
                            'timestamp': m.timestamp.isoformat(),
                            'success': m.success,
                            'error_message': m.error_message,
                            'metadata': m.metadata
                        }
                        for m in self.metrics
                    ],
                    'system_metrics': [
                        {
                            'cpu_percent': m.cpu_percent,
                            'memory_percent': m.memory_percent,
                            'memory_used_mb': m.memory_used_mb,
                            'timestamp': m.timestamp.isoformat()
                        }
                        for m in self.system_metrics
                    ],
                    'alerts': [
                        {
                            'alert_id': a.alert_id,
                            'alert_type': a.alert_type,
                            'severity': a.severity,
                            'message': a.message,
                            'threshold': a.threshold,
                            'current_value': a.current_value,
                            'timestamp': a.timestamp.isoformat(),
                            'status': a.status
                        }
                        for a in self.alerts
                    ]
                }

                with open(filepath, 'w', encoding='utf - 8') as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)

            logger.info(f"Metrics exported to {filepath}")

        except Exception as e:
            logger.error(f"Failed to export metrics: {str(e)}")


# 性能监控装饰器

def monitor_performance(operation_name: Optional[str] = None):
    """
    性能监控装饰器

    Args:
        operation_name: 操作名称，如果为None则使用函数名
    """

    def decorator(func):

        def wrapper(*args, **kwargs):

            op_name = operation_name or func.__name__
            start_time = time.time()

            try:
                result = func(*args, **kwargs)
                duration_ms = (time.time() - start_time) * 1000

                # 记录成功操作
                if hasattr(args[0], 'performance_monitor'):
                    args[0].performance_monitor.record_operation(
                        op_name, duration_ms, True
                    )

                return result

            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000

                # 记录失败操作
                if hasattr(args[0], 'performance_monitor'):
                    args[0].performance_monitor.record_operation(
                        op_name, duration_ms, False, str(e)
                    )

                raise

        return wrapper
    return decorator


if __name__ == "__main__":
    # 测试代码
    monitor = DataPerformanceMonitor()

    # 开始监控
    monitor.start_monitoring(interval_seconds=10)

    # 模拟一些操作
    for i in range(5):
        monitor.record_operation(
            operation="data_load",
            duration_ms=100 + i * 50,
            success=True,
            metadata={"symbol": f"STOCK_{i}"}
        )
        time.sleep(1)

    # 获取性能报告
    report = monitor.get_performance_report(hours=1)
    print("Performance Report:")
    print(json.dumps(report, indent=2, default=str))

    # 停止监控
    monitor.stop_monitoring()
