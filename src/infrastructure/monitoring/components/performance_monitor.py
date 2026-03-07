#!/usr/bin/env python3
"""
RQA2025 基础设施层性能监控组件

提供对监控系统各组件的性能监控功能，包括响应时间、内存使用、CPU使用等指标。
"""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
import statistics


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    component_name: str
    operation_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    cpu_usage_percent: Optional[float] = None
    success: bool = True
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    start_tick: float = field(default_factory=time.perf_counter)

    def complete(self, success: bool = True, error_message: Optional[str] = None):
        """完成性能测量"""
        self.end_time = time.time()
        if self.duration_ms is None:
            end_tick = time.perf_counter()
            duration_ms = (end_tick - self.start_tick) * 1000
            if duration_ms <= 0:
                duration_ms = 0.01
            self.duration_ms = duration_ms
        self.memory_usage_mb = psutil.Process().memory_info().rss / 1024 / 1024
        self.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
        self.success = success
        if error_message:
            self.error_message = error_message


@dataclass
class ComponentPerformanceStats:
    """组件性能统计"""
    component_name: str
    total_operations: int = 0
    successful_operations: int = 0
    failed_operations: int = 0
    avg_response_time_ms: float = 0.0
    min_response_time_ms: Optional[float] = None
    max_response_time_ms: Optional[float] = None
    p95_response_time_ms: Optional[float] = None
    p99_response_time_ms: Optional[float] = None
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    error_rate: float = 0.0
    last_updated: Optional[datetime] = None

    def update(self, metrics: PerformanceMetrics):
        """更新统计信息"""
        self.total_operations += 1
        if metrics.success:
            self.successful_operations += 1
        else:
            self.failed_operations += 1

        self.error_rate = self.failed_operations / self.total_operations

        # 更新响应时间统计
        if metrics.duration_ms is not None:
            self._update_response_time_stats(metrics.duration_ms)

        # 更新资源使用统计
        if metrics.memory_usage_mb is not None:
            self.memory_usage_mb = metrics.memory_usage_mb
        if metrics.cpu_usage_percent is not None:
            self.cpu_usage_percent = metrics.cpu_usage_percent

        self.last_updated = datetime.now()

    def _update_response_time_stats(self, duration_ms: float):
        """更新响应时间统计"""
        if self.min_response_time_ms is None or duration_ms < self.min_response_time_ms:
            self.min_response_time_ms = duration_ms
        if self.max_response_time_ms is None or duration_ms > self.max_response_time_ms:
            self.max_response_time_ms = duration_ms

        # 简单计算平均值（实际应该使用更精确的方法）
        self.avg_response_time_ms = (
            (self.avg_response_time_ms * (self.total_operations - 1)) + duration_ms
        ) / self.total_operations


class PerformanceMonitor:
    """
    性能监控器

    监控监控系统各组件的性能指标，提供性能分析和优化建议。
    """

    def __init__(self, enable_auto_monitoring: bool = True):
        """
        初始化性能监控器

        Args:
            enable_auto_monitoring: 是否启用自动性能监控
        """
        self.enable_auto_monitoring = enable_auto_monitoring
        self.metrics_history: List[PerformanceMetrics] = []
        self.component_stats: Dict[str, ComponentPerformanceStats] = {}
        self.max_history_size = 10000
        self.monitoring_active = False

        # 自动监控线程
        self.auto_monitor_thread: Optional[threading.Thread] = None
        self.auto_monitor_interval = 60  # 60秒间隔

        # 锁保护并发访问
        self._lock = threading.RLock()

        if enable_auto_monitoring:
            self.start_auto_monitoring()

    def monitor_operation(self, component_name: str, operation_name: str) -> 'PerformanceContext':
        """
        创建性能监控上下文

        Args:
            component_name: 组件名称
            operation_name: 操作名称

        Returns:
            PerformanceContext: 性能监控上下文
        """
        return PerformanceContext(self, component_name, operation_name)

    def record_metrics(self, metrics: PerformanceMetrics):
        """
        记录性能指标

        Args:
            metrics: 性能指标
        """
        with self._lock:
            # 添加到历史记录
            self.metrics_history.append(metrics)
            if len(self.metrics_history) > self.max_history_size:
                self.metrics_history = self.metrics_history[-self.max_history_size:]

            # 更新组件统计
            component_key = metrics.component_name
            if component_key not in self.component_stats:
                self.component_stats[component_key] = ComponentPerformanceStats(
                    component_name=component_key
                )

            self.component_stats[component_key].update(metrics)

    def get_component_stats(self, component_name: str) -> Optional[ComponentPerformanceStats]:
        """
        获取组件性能统计

        Args:
            component_name: 组件名称

        Returns:
            Optional[ComponentPerformanceStats]: 组件性能统计
        """
        return self.component_stats.get(component_name)

    def get_all_component_stats(self) -> Dict[str, ComponentPerformanceStats]:
        """
        获取所有组件性能统计

        Returns:
            Dict[str, ComponentPerformanceStats]: 所有组件的性能统计
        """
        return self.component_stats.copy()

    def get_recent_metrics(self, component_name: Optional[str] = None,
                          limit: int = 100) -> List[PerformanceMetrics]:
        """
        获取最近的性能指标

        Args:
            component_name: 组件名称过滤（可选）
            limit: 返回的最大记录数

        Returns:
            List[PerformanceMetrics]: 性能指标列表
        """
        with self._lock:
            metrics = self.metrics_history
            if component_name:
                metrics = [m for m in metrics if m.component_name == component_name]

            return metrics[-limit:] if limit > 0 else metrics

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        获取性能汇总报告

        Returns:
            Dict[str, Any]: 性能汇总报告
        """
        summary = {
            'total_components': len(self.component_stats),
            'total_operations': sum(stats.total_operations for stats in self.component_stats.values()),
            'total_metrics': len(self.metrics_history),
            'components': {},
            'system_health': self._calculate_system_health(),
            'generated_at': datetime.now().isoformat()
        }

        # 组件详情
        for name, stats in self.component_stats.items():
            summary['components'][name] = {
                'total_operations': stats.total_operations,
                'success_rate': (stats.successful_operations / stats.total_operations) if stats.total_operations > 0 else 0,
                'error_rate': stats.error_rate,
                'avg_response_time_ms': stats.avg_response_time_ms,
                'memory_usage_mb': stats.memory_usage_mb,
                'cpu_usage_percent': stats.cpu_usage_percent
            }

        return summary

    def detect_performance_anomalies(self) -> List[Dict[str, Any]]:
        """
        检测性能异常

        Returns:
            List[Dict[str, Any]]: 检测到的性能异常列表
        """
        anomalies = []

        for component_name, stats in self.component_stats.items():
            # 检查错误率异常
            if stats.error_rate > 0.1:  # 错误率超过10%
                anomalies.append({
                    'type': 'high_error_rate',
                    'component': component_name,
                    'severity': 'high' if stats.error_rate > 0.5 else 'medium',
                    'description': f'组件 {component_name} 错误率过高: {stats.error_rate:.2%}',
                    'current_value': stats.error_rate,
                    'threshold': 0.1
                })

            # 检查响应时间异常
            if stats.avg_response_time_ms > 1000:  # 平均响应时间超过1秒
                anomalies.append({
                    'type': 'slow_response',
                    'component': component_name,
                    'severity': 'high' if stats.avg_response_time_ms > 5000 else 'medium',
                    'description': f'组件 {component_name} 响应时间过慢: {stats.avg_response_time_ms:.1f}ms',
                    'current_value': stats.avg_response_time_ms,
                    'threshold': 1000
                })

            # 检查内存使用异常
            if stats.memory_usage_mb > 500:  # 内存使用超过500MB
                anomalies.append({
                    'type': 'high_memory_usage',
                    'component': component_name,
                    'severity': 'medium',
                    'description': f'组件 {component_name} 内存使用过高: {stats.memory_usage_mb:.1f}MB',
                    'current_value': stats.memory_usage_mb,
                    'threshold': 500
                })

        return anomalies

    def generate_performance_recommendations(self) -> List[Dict[str, Any]]:
        """
        生成性能优化建议

        Returns:
            List[Dict[str, Any]]: 性能优化建议列表
        """
        recommendations = []

        for component_name, stats in self.component_stats.items():
            # 生成各种类型的建议
            component_recommendations = self._generate_component_recommendations(
                component_name, stats
            )
            recommendations.extend(component_recommendations)

        return recommendations

    def _generate_component_recommendations(self, component_name: str,
                                           stats: Any) -> List[Dict[str, Any]]:
        """
        生成单个组件的性能建议

        Args:
            component_name: 组件名称
            stats: 组件统计信息

        Returns:
            List[Dict[str, Any]]: 该组件的建议列表
        """
        recommendations = []

        # 检查各种性能指标并生成相应建议
        recommendations.extend(self._check_error_rate_recommendations(component_name, stats))
        recommendations.extend(self._check_response_time_recommendations(component_name, stats))
        recommendations.extend(self._check_memory_usage_recommendations(component_name, stats))

        return recommendations

    def _check_error_rate_recommendations(self, component_name: str, stats: Any) -> List[Dict[str, Any]]:
        """检查错误率相关的建议"""
        if stats.error_rate > 0.05:
            return [{
                'component': component_name,
                'type': 'error_handling',
                'priority': 'high',
                'title': '改进错误处理机制',
                'description': f'组件 {component_name} 错误率较高 ({stats.error_rate:.1%})，建议改进错误处理和重试机制',
                'actions': [
                    '添加更详细的错误日志',
                    '实现重试机制',
                    '添加断路器模式'
                ]
            }]
        return []

    def _check_response_time_recommendations(self, component_name: str, stats: Any) -> List[Dict[str, Any]]:
        """检查响应时间相关的建议"""
        if stats.avg_response_time_ms > 500:
            return [{
                'component': component_name,
                'type': 'performance_optimization',
                'priority': 'medium',
                'title': '优化响应时间',
                'description': f'组件 {component_name} 平均响应时间较慢 ({stats.avg_response_time_ms:.1f}ms)',
                'actions': [
                    '检查数据库查询性能',
                    '添加缓存机制',
                    '优化算法复杂度'
                ]
            }]
        return []

    def _check_memory_usage_recommendations(self, component_name: str, stats: Any) -> List[Dict[str, Any]]:
        """检查内存使用相关的建议"""
        if stats.memory_usage_mb > 200:
            return [{
                'component': component_name,
                'type': 'memory_optimization',
                'priority': 'medium',
                'title': '优化内存使用',
                'description': f'组件 {component_name} 内存使用较高 ({stats.memory_usage_mb:.1f}MB)',
                'actions': [
                    '检查内存泄漏',
                    '使用内存池',
                    '优化数据结构'
                ]
            }]
        return []

    def start_auto_monitoring(self):
        """启动自动性能监控"""
        if self.monitoring_active:
            return

        self.monitoring_active = True
        self.auto_monitor_thread = threading.Thread(
            target=self._auto_monitor_loop,
            name="PerformanceMonitor",
            daemon=True
        )
        self.auto_monitor_thread.start()

    def stop_auto_monitoring(self):
        """停止自动性能监控"""
        self.monitoring_active = False
        if self.auto_monitor_thread:
            self.auto_monitor_thread.join(timeout=5)

    def _auto_monitor_loop(self):
        """自动监控循环"""
        while self.monitoring_active:
            try:
                # 收集系统级性能指标
                self._collect_system_metrics()
                time.sleep(self.auto_monitor_interval)
            except Exception as e:
                print(f"性能监控循环异常: {e}")
                time.sleep(self.auto_monitor_interval)

    def _collect_system_metrics(self):
        """收集系统级性能指标"""
        try:
            # 系统级性能指标
            metrics = PerformanceMetrics(
                component_name="system",
                operation_name="system_monitoring",
                start_time=time.time()
            )

            # 模拟一些系统监控操作
            time.sleep(0.01)  # 模拟监控操作耗时

            metrics.complete(success=True)
            self.record_metrics(metrics)

        except Exception as e:
            print(f"收集系统性能指标失败: {e}")

    def _calculate_system_health(self) -> Dict[str, Any]:
        """
        计算系统整体健康状态

        Returns:
            Dict[str, Any]: 系统健康状态
        """
        if not self.component_stats:
            return {'status': 'unknown', 'score': 0.0}

        total_operations = sum(stats.total_operations for stats in self.component_stats.values())
        total_errors = sum(stats.failed_operations for stats in self.component_stats.values())

        # 简单健康评分算法
        error_rate = total_errors / total_operations if total_operations > 0 else 0
        avg_response_time = statistics.mean([
            stats.avg_response_time_ms for stats in self.component_stats.values()
            if stats.avg_response_time_ms > 0
        ]) if self.component_stats else 0

        # 健康评分 (0-100)
        health_score = 100 * (1 - error_rate) * (1 - min(avg_response_time / 1000, 1))

        if health_score >= 90:
            status = 'excellent'
        elif health_score >= 70:
            status = 'good'
        elif health_score >= 50:
            status = 'fair'
        else:
            status = 'poor'

        return {
            'status': status,
            'score': round(health_score, 2),
            'error_rate': round(error_rate, 4),
            'avg_response_time_ms': round(avg_response_time, 2)
        }


class PerformanceContext:
    """
    性能监控上下文管理器

    用于自动测量操作的性能指标。
    """

    def __init__(self, monitor: PerformanceMonitor, component_name: str, operation_name: str):
        """
        初始化性能监控上下文

        Args:
            monitor: 性能监控器
            component_name: 组件名称
            operation_name: 操作名称
        """
        self.monitor = monitor
        self.metrics = PerformanceMetrics(
            component_name=component_name,
            operation_name=operation_name,
            start_time=time.time()
        )

    def __enter__(self):
        """进入上下文"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """退出上下文"""
        success = exc_type is None
        error_message = str(exc_val) if exc_val else None

        self.metrics.complete(success=success, error_message=error_message)
        self.monitor.record_metrics(self.metrics)


# 全局性能监控器实例
global_performance_monitor = PerformanceMonitor()


def monitor_performance(component_name: str, operation_name: str):
    """
    性能监控装饰器

    Args:
        component_name: 组件名称
        operation_name: 操作名称

    Returns:
        Callable: 装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with global_performance_monitor.monitor_operation(component_name, operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator
