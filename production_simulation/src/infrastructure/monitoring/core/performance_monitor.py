#!/usr/bin/env python3
"""
RQA2025 基础设施层性能监控器

提供组件性能指标收集、监控和分析功能。
支持实时性能跟踪、历史数据分析和性能基准测试。
"""

import threading
import time
import logging
import statistics
from typing import Dict, Any, Optional, List, Callable, TypeVar, Union
from datetime import datetime, timedelta
from functools import wraps
from collections import defaultdict, deque
import psutil
import os

logger = logging.getLogger(__name__)

F = TypeVar('F', bound=Callable[..., Any])


class PerformanceMetrics:
    """性能指标数据类"""

    def __init__(self, name: str):
        """
        初始化性能指标

        Args:
            name: 指标名称
        """
        self.name = name
        self.values: deque = deque(maxlen=1000)  # 保留最近1000个数据点
        self.timestamps: deque = deque(maxlen=1000)
        self.lock = threading.RLock()

    def add_value(self, value: float, timestamp: Optional[datetime] = None):
        """
        添加指标值

        Args:
            value: 指标值
            timestamp: 时间戳
        """
        if timestamp is None:
            timestamp = datetime.now()

        with self.lock:
            self.values.append(value)
            self.timestamps.append(timestamp)

    def get_recent_values(self, minutes: int = 5) -> List[float]:
        """
        获取最近N分钟的数据

        Args:
            minutes: 分钟数

        Returns:
            List[float]: 指标值列表
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)

        with self.lock:
            recent_values = []
            for i, ts in enumerate(self.timestamps):
                if ts >= cutoff_time:
                    recent_values.extend(list(self.values)[i:])
                    break
            return recent_values

    def get_stats(self, minutes: int = 5) -> Dict[str, float]:
        """
        获取统计信息

        Args:
            minutes: 时间窗口（分钟）

        Returns:
            Dict[str, float]: 统计信息
        """
        values = self.get_recent_values(minutes)

        if not values:
            return {
                'count': 0,
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'min': 0.0,
                'max': 0.0,
                'latest': 0.0
            }

        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
            'min': min(values),
            'max': max(values),
            'latest': values[-1]
        }


class PerformanceMonitor:
    """
    性能监控器

    收集和分析系统及组件的性能指标。
    """

    def __init__(self, collection_interval: int = 10):
        """
        初始化性能监控器

        Args:
            collection_interval: 收集间隔（秒）
        """
        self.collection_interval = collection_interval
        self.is_running = False
        self.monitor_thread = None

        # 性能指标存储
        self.metrics: Dict[str, PerformanceMetrics] = {}
        self.metrics_lock = threading.RLock()

        # 系统资源监控
        self.system_metrics = {
            'cpu_usage': PerformanceMetrics('cpu_usage'),
            'memory_usage': PerformanceMetrics('memory_usage'),
            'disk_usage': PerformanceMetrics('disk_usage'),
            'network_io': PerformanceMetrics('network_io')
        }

        # 组件性能指标
        self.component_metrics: Dict[str, Dict[str, PerformanceMetrics]] = defaultdict(dict)

        # 性能阈值
        self.thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'response_time': 1.0
        }

        logger.info("性能监控器初始化完成")

    def start(self):
        """启动性能监控"""
        if self.is_running:
            return

        self.is_running = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.monitor_thread.start()

        logger.info("性能监控器已启动")

    def stop(self):
        """停止性能监控"""
        if not self.is_running:
            return

        self.is_running = False
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=5)

        logger.info("性能监控器已停止")

    def _monitoring_loop(self):
        """监控循环"""
        while self.is_running:
            try:
                self._collect_system_metrics()
                self._check_thresholds()

                time.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"性能监控循环异常: {e}")
                time.sleep(5)  # 错误时等待更长时间

    def _collect_system_metrics(self):
        """收集系统性能指标"""
        try:
            # CPU 使用率
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric('cpu_usage', cpu_percent)

            # 内存使用率
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            self.record_metric('memory_usage', memory_percent)

            # 磁盘使用率
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            self.record_metric('disk_usage', disk_percent)

            # 网络I/O
            network = psutil.net_io_counters()
            if network:
                network_total = network.bytes_sent + network.bytes_recv
                self.record_metric('network_io', network_total)

        except Exception as e:
            logger.warning(f"收集系统指标失败: {e}")

    def _check_thresholds(self):
        """检查性能阈值"""
        for metric_name, threshold in self.thresholds.items():
            current_value = self.get_metric_latest(metric_name)
            if current_value and current_value > threshold:
                logger.warning(f"性能指标 {metric_name} 超过阈值: {current_value:.2f} > {threshold}")

                # 发布性能告警事件
                from .component_bus import global_component_bus, Message, MessageType
                global_component_bus.publish(Message(
                    type=MessageType.PERFORMANCE_ALERT,
                    topic="performance.threshold.exceeded",
                    payload={
                        'metric': metric_name,
                        'value': current_value,
                        'threshold': threshold,
                        'timestamp': datetime.now().isoformat()
                    }
                ))

    def record_metric(self, name: str, value: float, component: Optional[str] = None):
        """
        记录性能指标

        Args:
            name: 指标名称
            value: 指标值
            component: 组件名称
        """
        with self.metrics_lock:
            if component:
                if component not in self.component_metrics:
                    self.component_metrics[component] = {}
                if name not in self.component_metrics[component]:
                    self.component_metrics[component][name] = PerformanceMetrics(f"{component}.{name}")
                self.component_metrics[component][name].add_value(value)
            else:
                if name not in self.metrics:
                    self.metrics[name] = PerformanceMetrics(name)
                self.metrics[name].add_value(value)

    def record_component_metric(self, component: str, metric_name: str, value: float):
        """
        记录组件性能指标

        Args:
            component: 组件名称
            metric_name: 指标名称
            value: 指标值
        """
        self.record_metric(metric_name, value, component)

    def get_metric_stats(self, name: str, minutes: int = 5,
                        component: Optional[str] = None) -> Dict[str, float]:
        """
        获取指标统计信息

        Args:
            name: 指标名称
            minutes: 时间窗口
            component: 组件名称

        Returns:
            Dict[str, float]: 统计信息
        """
        with self.metrics_lock:
            if component:
                if component in self.component_metrics and name in self.component_metrics[component]:
                    return self.component_metrics[component][name].get_stats(minutes)
            else:
                if name in self.metrics:
                    return self.metrics[name].get_stats(minutes)

        return {}

    def get_metric_latest(self, name: str, component: Optional[str] = None) -> Optional[float]:
        """
        获取最新指标值

        Args:
            name: 指标名称
            component: 组件名称

        Returns:
            Optional[float]: 最新值
        """
        stats = self.get_metric_stats(name, minutes=1, component=component)
        return stats.get('latest') if stats else None

    def get_recent_metrics(self, minutes: int = 5) -> Dict[str, float]:
        """
        获取最近的性能指标

        Args:
            minutes: 时间窗口

        Returns:
            Dict[str, float]: 指标字典
        """
        metrics = {}

        with self.metrics_lock:
            # 系统指标
            for name in ['cpu_usage', 'memory_usage', 'disk_usage', 'network_io']:
                latest = self.get_metric_latest(name)
                if latest is not None:
                    metrics[name] = latest

            # 组件指标
            for component, comp_metrics in self.component_metrics.items():
                for metric_name, metric_obj in comp_metrics.items():
                    latest = self.get_metric_latest(metric_name, component)
                    if latest is not None:
                        metrics[f"{component}.{metric_name}"] = latest

        return metrics

    def get_component_performance_report(self, component: str,
                                       minutes: int = 30) -> Dict[str, Any]:
        """
        获取组件性能报告

        Args:
            component: 组件名称
            minutes: 时间窗口

        Returns:
            Dict[str, Any]: 性能报告
        """
        report = {
            'component': component,
            'time_window_minutes': minutes,
            'metrics': {},
            'summary': {}
        }

        with self.metrics_lock:
            if component not in self.component_metrics:
                return report

            total_metrics = 0
            total_values = 0
            max_value = 0
            min_value = float('inf')

            for metric_name, metric_obj in self.component_metrics[component].items():
                stats = metric_obj.get_stats(minutes)
                if stats['count'] > 0:
                    report['metrics'][metric_name] = stats
                    total_metrics += 1
                    total_values += stats['mean']
                    max_value = max(max_value, stats['max'])
                    min_value = min(min_value, stats['min'])

            if total_metrics > 0:
                report['summary'] = {
                    'total_metrics': total_metrics,
                    'average_performance': total_values / total_metrics,
                    'best_performance': min_value,
                    'worst_performance': max_value
                }

        return report

    def set_threshold(self, metric_name: str, threshold: float):
        """
        设置性能阈值

        Args:
            metric_name: 指标名称
            threshold: 阈值
        """
        self.thresholds[metric_name] = threshold
        logger.info(f"设置性能阈值: {metric_name} = {threshold}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能摘要

        Returns:
            Dict[str, Any]: 性能摘要
        """
        summary = {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {},
            'component_count': len(self.component_metrics),
            'alerts': []
        }

        # 系统指标摘要
        for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage']:
            stats = self.get_metric_stats(metric_name, minutes=5)
            if stats['count'] > 0:
                summary['system_metrics'][metric_name] = {
                    'current': stats['latest'],
                    'average': stats['mean'],
                    'status': 'warning' if stats['latest'] > self.thresholds.get(metric_name, 100) else 'normal'
                }

        # 检查是否有活跃告警
        for metric_name, threshold in self.thresholds.items():
            current = self.get_metric_latest(metric_name)
            if current and current > threshold:
                summary['alerts'].append({
                    'metric': metric_name,
                    'current_value': current,
                    'threshold': threshold,
                    'severity': 'high' if current > threshold * 1.5 else 'medium'
                })

        return summary


def monitor_performance(operation_name: Optional[str] = None):
    """
    性能监控装饰器

    Args:
        operation_name: 操作名称，如果不提供则使用函数名

    Returns:
        装饰器函数
    """
    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            operation = operation_name or f"{func.__module__}.{func.__name__}"

            try:
                # 执行函数
                result = func(*args, **kwargs)

                # 记录执行时间
                execution_time = time.time() - start_time
                global_performance_monitor.record_metric(
                    f"{operation}.execution_time",
                    execution_time
                )

                # 记录成功次数
                global_performance_monitor.record_metric(
                    f"{operation}.success_count",
                    1
                )

                return result

            except Exception as e:
                # 记录失败信息
                execution_time = time.time() - start_time
                global_performance_monitor.record_metric(
                    f"{operation}.execution_time",
                    execution_time
                )
                global_performance_monitor.record_metric(
                    f"{operation}.error_count",
                    1
                )

                logger.warning(f"操作 {operation} 执行失败: {e}")
                raise

        return wrapper
    return decorator


# 全局性能监控器实例
global_performance_monitor = PerformanceMonitor()


def start_performance_monitoring():
    """启动全局性能监控"""
    global_performance_monitor.start()


def stop_performance_monitoring():
    """停止全局性能监控"""
    global_performance_monitor.stop()


def get_performance_report() -> Dict[str, Any]:
    """
    获取性能报告

    Returns:
        Dict[str, Any]: 性能报告
    """
    return global_performance_monitor.get_performance_summary()
