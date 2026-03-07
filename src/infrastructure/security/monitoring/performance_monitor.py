#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RQA2025 性能监控器

专门负责监控安全模块的性能指标
提供实时性能统计和优化建议
"""

import time
import math
import threading
import psutil
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics


@dataclass
class PerformanceMetrics:
    """性能指标"""
    operation_name: str
    total_calls: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    p95_time: float = 0.0
    p99_time: float = 0.0
    error_count: int = 0
    last_call_time: Optional[datetime] = None
    first_call_time: Optional[datetime] = None
    last_call_micro: Optional[int] = None
    first_call_micro: Optional[int] = None
    last_reported_ts: Optional[float] = None
    call_times: deque = field(default_factory=lambda: deque(maxlen=1000))

    def record_call(self, duration: float, is_error: bool = False, *, timestamp: Optional[float] = None) -> None:
        """记录一次调用"""
        self.total_calls += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        ts = timestamp if timestamp is not None else time.time()
        if self.last_reported_ts is not None and ts <= self.last_reported_ts:
            ts = self.last_reported_ts

        micro = math.floor(ts * 1_000_000)
        logical_ts = micro / 1_000_000
        now_dt = datetime.fromtimestamp(logical_ts)
        if self.first_call_time is None:
            self.first_call_time = now_dt
            self.first_call_micro = micro
        self.last_call_time = now_dt
        self.last_call_micro = micro
        self.last_reported_ts = logical_ts

        if is_error:
            self.error_count += 1

        self.call_times.append(duration)
        self._update_statistics()

    def _update_statistics(self) -> None:
        """更新统计信息"""
        if self.total_calls > 0:
            self.avg_time = self.total_time / self.total_calls

        if len(self.call_times) >= 10:  # 至少需要10个样本
            sorted_times = sorted(self.call_times)
            n = len(sorted_times)

            # 计算百分位数
            p95_index = int(n * 0.95)
            p99_index = int(n * 0.99)

            self.p95_time = sorted_times[min(p95_index, n-1)]
            self.p99_time = sorted_times[min(p99_index, n-1)]

    def get_error_rate(self) -> float:
        """获取错误率"""
        return self.error_count / max(self.total_calls, 1) * 100

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'operation_name': self.operation_name,
            'total_calls': self.total_calls,
            'total_time': self.total_time,
            'min_time': self.min_time,
            'max_time': self.max_time,
            'avg_time': self.avg_time,
            'p95_time': self.p95_time,
            'p99_time': self.p99_time,
            'error_count': self.error_count,
            'error_rate': self.get_error_rate(),
            'first_call_time': self.first_call_time.isoformat() if self.first_call_time else None,
            'last_call_time': self.last_call_time.isoformat() if self.last_call_time else None
        }

    @staticmethod
    def _micro_to_datetime(micro: int) -> datetime:
        seconds, micros = divmod(micro, 1_000_000)
        base = datetime.fromtimestamp(seconds)
        return base.replace(microsecond=micros)


class PerformanceMonitor:
    """性能监控器"""

    def __init__(self, enabled: bool = True, collection_interval: int = 60):
        self.enabled = enabled
        self.collection_interval = collection_interval
        self._metrics: Dict[str, PerformanceMetrics] = {}
        self._lock = threading.RLock()
        self._system_stats = deque(maxlen=100)  # 保留最近100个系统统计

        # 安全监控相关属性
        self.user_activity: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}
        self.resource_access: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}

        if enabled:
            self._start_background_collection()

    def record_operation(self, operation_name: str, duration: float,
                        is_error: bool = False, user_id: Optional[str] = None,
                        resource: Optional[str] = None) -> None:
        """记录操作性能"""
        if not self.enabled:
            return

        with self._lock:
            if operation_name not in self._metrics:
                self._metrics[operation_name] = PerformanceMetrics(operation_name)

            timestamp = time.time()
            self._metrics[operation_name].record_call(duration, is_error, timestamp=timestamp)

            # 记录安全特定的指标
            if user_id:
                self._record_user_activity(user_id, operation_name, duration)
            if resource:
                self._record_resource_access(resource, operation_name, duration)

    def get_metrics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """获取性能指标"""
        with self._lock:
            if operation_name:
                metrics = self._metrics.get(operation_name)
                return metrics.to_dict() if metrics else {}

            # 返回所有指标
            return {
                name: metrics.to_dict()
                for name, metrics in self._metrics.items()
            }

    def get_system_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        # 获取当前系统资源使用情况
        try:
            cpu_percent = psutil.cpu_percent(interval=0.0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
        except Exception as exc:
            logging.error(f"获取系统统计信息失败: {exc}")
            return {"timestamp": datetime.now().isoformat(), "error": str(exc)}

        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_used_gb': memory.used / (1024**3),
            'memory_total_gb': memory.total / (1024**3),
            'disk_percent': disk.percent,
            'disk_used_gb': disk.used / (1024**3),
            'disk_total_gb': disk.total / (1024**3),
            'memory_usage': memory.percent,
            'disk_usage': disk.percent
        }

        # 添加到历史记录
        self._system_stats.append(stats)

        return stats

    def get_performance_report(self) -> Dict[str, Any]:
        """生成性能报告"""
        with self._lock:
            metrics_data = self.get_metrics()
            system_stats = self.get_system_stats()
            security_metrics = {
                'user_activity_summary': self._get_user_activity_summary(),
                'resource_access_summary': self._get_resource_access_summary(),
                'security_operation_trends': self._get_security_operation_trends(),
            }

            # 分析瓶颈
            bottlenecks = self._analyze_bottlenecks(metrics_data)

            # 生成优化建议
            recommendations = self._generate_recommendations(metrics_data, system_stats)

            return {
                'timestamp': datetime.now().isoformat(),
                'operations': metrics_data,
                'metrics': metrics_data,  # 兼容旧版接口
                'system_stats': system_stats,
                'bottlenecks': bottlenecks,
                'recommendations': recommendations,
                'summary': self._generate_summary(metrics_data),
                'security_metrics': security_metrics,
            }

    def reset_metrics(self, operation_name: Optional[str] = None) -> None:
        """重置性能指标"""
        with self._lock:
            if operation_name:
                if operation_name in self._metrics:
                    del self._metrics[operation_name]
            else:
                self._metrics.clear()

    def _analyze_bottlenecks(self, metrics_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """分析性能瓶颈"""
        bottlenecks = []

        for name, metrics in metrics_data.items():
            issues = []

            # 检查响应时间
            if metrics['avg_time'] > 1.0:  # 平均响应时间超过1秒
                issues.append({
                    'type': 'slow_response',
                    'severity': 'high',
                    'description': f'平均响应时间过高: {metrics["avg_time"]:.3f}s'
                })

            # 检查P95响应时间
            if metrics['p95_time'] > 2.0:  # P95响应时间超过2秒
                issues.append({
                    'type': 'p95_high',
                    'severity': 'medium',
                    'description': f'P95响应时间过高: {metrics["p95_time"]:.3f}s'
                })

            # 检查错误率
            if metrics['error_rate'] > 5.0:  # 错误率超过5%
                issues.append({
                    'type': 'high_error_rate',
                    'severity': 'high',
                    'description': f'错误率过高: {metrics["error_rate"]:.1f}%'
                })

            if issues:
                bottlenecks.append({
                    'operation': name,
                    'issues': issues,
                    'metrics': metrics
                })

        return bottlenecks

    def _generate_recommendations(self, metrics_data: Dict[str, Any],
                                system_stats: Dict[str, Any]) -> List[str]:
        """生成优化建议"""
        recommendations = []

        # 基于系统资源使用情况
        if system_stats['cpu_percent'] > 80:
            recommendations.append("CPU使用率过高，考虑增加CPU核心或优化算法")

        if system_stats['memory_percent'] > 85:
            recommendations.append("内存使用率过高，检查是否存在内存泄漏")

        # 基于操作性能
        slow_operations = [
            name for name, metrics in metrics_data.items()
            if metrics['avg_time'] > 0.5
        ]

        if slow_operations:
            recommendations.append(f"以下操作响应较慢，需要优化: {', '.join(slow_operations[:3])}")

        # 基于错误率
        high_error_operations = [
            name for name, metrics in metrics_data.items()
            if metrics['error_rate'] > 1.0
        ]

        if high_error_operations:
            recommendations.append(f"以下操作错误率较高，需要检查: {', '.join(high_error_operations[:3])}")

        # 通用建议
        if not recommendations:
            recommendations.append("系统性能表现良好，继续保持监控")

        return recommendations

    def _generate_summary(self, metrics_data: Dict[str, Any]) -> Dict[str, Any]:
        """生成摘要信息"""
        if not metrics_data:
            return {'status': 'no_data', 'message': '暂无性能数据'}

        total_operations = len(metrics_data)
        total_calls = sum(m['total_calls'] for m in metrics_data.values())
        avg_response_time = statistics.mean(
            m['avg_time'] for m in metrics_data.values() if m['total_calls'] > 0
        ) if metrics_data else 0

        # 性能评分 (0-100, 越高越好)
        score = 100

        # 基于平均响应时间扣分
        if avg_response_time > 1.0:
            score -= 20
        elif avg_response_time > 0.5:
            score -= 10

        # 基于错误率扣分
        total_errors = sum(m['error_count'] for m in metrics_data.values())
        error_rate = total_errors / max(total_calls, 1) * 100
        if error_rate > 5:
            score -= 30
        elif error_rate > 1:
            score -= 10

        score = max(0, min(100, score))

        return {
            'total_operations': total_operations,
            'total_calls': total_calls,
            'avg_response_time': avg_response_time,
            'overall_error_rate': error_rate,
            'error_count': total_errors,
            'performance_score': score,
            'status': 'good' if score >= 80 else 'warning' if score >= 60 else 'critical'
        }

    def _start_background_collection(self) -> None:
        """启动后台统计收集"""
        def collect_stats():
            while self.enabled:
                try:
                    self.get_system_stats()
                    time.sleep(self.collection_interval)
                except Exception as e:
                    logging.error(f"收集系统统计信息失败: {e}")
                    time.sleep(self.collection_interval)

        thread = threading.Thread(target=collect_stats, daemon=True)
        thread.start()

    def shutdown(self):
        """关闭性能监控器"""
        self.enabled = False
        # 这里可以添加其他清理逻辑


# 性能监控装饰器
    def _record_user_activity(self, user_id: str, operation: str, duration: float):
        """记录用户活动"""
        if user_id not in self.user_activity:
            self.user_activity[user_id] = {}

        if operation not in self.user_activity[user_id]:
            self.user_activity[user_id][operation] = []

        self.user_activity[user_id][operation].append({
            'timestamp': datetime.now(),
            'duration': duration
        })

        # 保持最近1000条记录
        if len(self.user_activity[user_id][operation]) > 1000:
            self.user_activity[user_id][operation] = self.user_activity[user_id][operation][-1000:]

    def _record_resource_access(self, resource: str, operation: str, duration: float):
        """记录资源访问"""
        if resource not in self.resource_access:
            self.resource_access[resource] = {}

        if operation not in self.resource_access[resource]:
            self.resource_access[resource][operation] = []

        self.resource_access[resource][operation].append({
            'timestamp': datetime.now(),
            'duration': duration
        })

        # 保持最近1000条记录
        if len(self.resource_access[resource][operation]) > 1000:
            self.resource_access[resource][operation] = self.resource_access[resource][operation][-1000:]

    def _get_user_activity_summary(self) -> Dict[str, Any]:
        """汇总用户活动"""
        summary: Dict[str, Any] = {}
        for user_id, operations in self.user_activity.items():
            summary[user_id] = {}
            for operation, records in operations.items():
                durations = [record["duration"] for record in records]
                summary[user_id][operation] = {
                    "count": len(records),
                    "avg_duration": statistics.mean(durations) if durations else 0.0,
                    "last_activity": records[-1]["timestamp"].isoformat() if records else None,
                }
        return summary

    def _get_resource_access_summary(self) -> Dict[str, Any]:
        """汇总资源访问情况"""
        summary: Dict[str, Any] = {}
        for resource, operations in self.resource_access.items():
            summary[resource] = {}
            for operation, records in operations.items():
                durations = [record["duration"] for record in records]
                summary[resource][operation] = {
                    "count": len(records),
                    "avg_duration": statistics.mean(durations) if durations else 0.0,
                    "last_access": records[-1]["timestamp"].isoformat() if records else None,
                }
        return summary

    def _get_security_operation_trends(self) -> Dict[str, Any]:
        """基于已记录的指标生成安全操作趋势摘要"""
        trends: Dict[str, Any] = {}
        for operation, metrics in self._metrics.items():
            trends[operation] = {
                "total_calls": metrics.total_calls,
                "avg_time": metrics.avg_time,
                "error_rate": metrics.get_error_rate(),
                "last_call_time": metrics.last_call_time.isoformat() if metrics.last_call_time else None,
            }
        return trends


def monitor_performance(operation_name: str, monitor: Optional[PerformanceMonitor] = None):
    """性能监控装饰器"""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            start_time = time.time()
            is_error = False

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                is_error = True
                raise
            finally:
                duration = time.time() - start_time
                if monitor:
                    monitor.record_operation(operation_name, duration, is_error)

        return wrapper
    return decorator

_global_monitor = PerformanceMonitor()

def get_performance_monitor() -> PerformanceMonitor:
    """获取全局性能监控器"""
    return _global_monitor

def record_performance(operation_name: str, duration: float, is_error: bool = False) -> None:
    """记录性能数据"""
    _global_monitor.record_operation(operation_name, duration, is_error)

def record_security_operation(operation: str, duration: float,
                            user_id: Optional[str] = None,
                            resource: Optional[str] = None,
                            is_error: bool = False) -> None:
    """
    记录安全操作性能

    Args:
        operation: 操作名称
        duration: 执行时间(秒)
        user_id: 用户ID
        resource: 资源标识
        is_error: 是否出错
    """
    _global_monitor.record_operation(
        operation,
        duration,
        is_error,
        user_id=user_id,
        resource=resource,
    )

def get_security_performance_report() -> Dict[str, Any]:
    """
    获取安全性能报告

    Returns:
        安全性能报告
    """
    base_report = _global_monitor.get_performance_report()
    security_report = {
        'security_metrics': {
            'user_activity_summary': _global_monitor._get_user_activity_summary(),
            'resource_access_summary': _global_monitor._get_resource_access_summary(),
            'security_operation_trends': _global_monitor._get_security_operation_trends()
        }
    }

    # 合并基础报告和安全报告
    base_report.update(security_report)
    return base_report
