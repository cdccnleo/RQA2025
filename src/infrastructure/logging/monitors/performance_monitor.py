#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层 - 日志系统性能监控器
"""

import time
import threading
from typing import Dict, List, Any, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import psutil
import os

from .base import BaseMonitor


class PerformanceMonitor(BaseMonitor):
    """性能监控器"""

    def __init__(self, name: str = "performance_monitor",
                 sample_interval: float = 1.0,
                 retention_period: int = 3600):
        config = {
            'name': name,
            'sample_interval': sample_interval,
            'retention_period': retention_period
        }
        super().__init__(config)
        self.sample_interval = sample_interval
        self.retention_period = retention_period

        # 性能指标存储
        self._metrics = {
            'log_throughput': [],  # 日志吞吐量 (logs/second)
            'memory_usage': [],    # 内存使用 (MB)
            'cpu_usage': [],       # CPU使用率 (%)
            'disk_io': [],         # 磁盘IO (bytes)
            'response_times': [],  # 响应时间 (ms)
            'error_rates': [],     # 错误率 (%)
            'queue_sizes': [],     # 队列大小
        }

        # 时间戳存储
        self._timestamps = []

        # 统计计数器
        self._stats = {
            'total_logs': 0,
            'error_logs': 0,
            'warning_logs': 0,
            'last_sample_time': time.time(),
            'start_time': time.time(),
        }

        # 锁保护
        self._lock = threading.RLock()

        # 进程信息
        self._process = psutil.Process(os.getpid())

        # 启动监控线程
        self._running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def record_log(self, level: str, message: str, **kwargs):
        """记录日志事件"""
        with self._lock:
            self._stats['total_logs'] += 1

            if level in ['ERROR', 'CRITICAL']:
                self._stats['error_logs'] += 1
            elif level == 'WARNING':
                self._stats['warning_logs'] += 1

            # 记录响应时间
            if 'response_time' in kwargs:
                self._metrics['response_times'].append(kwargs['response_time'] * 1000)  # 转换为ms

            # 记录队列大小
            if 'queue_size' in kwargs:
                self._metrics['queue_sizes'].append(kwargs['queue_size'])

    def get_throughput(self) -> float:
        """获取当前日志吞吐量 (logs/second)"""
        with self._lock:
            elapsed = time.time() - self._stats['start_time']
            if elapsed > 0:
                return self._stats['total_logs'] / elapsed
            return 0.0

    def get_error_rate(self) -> float:
        """获取错误率 (%)"""
        with self._lock:
            if self._stats['total_logs'] > 0:
                return (self._stats['error_logs'] / self._stats['total_logs']) * 100
            return 0.0

    def get_memory_usage(self) -> float:
        """获取内存使用 (MB)"""
        try:
            memory_info = self._process.memory_info()
            return memory_info.rss / (1024 * 1024)  # 转换为MB
        except:
            return 0.0

    def get_cpu_usage(self) -> float:
        """获取CPU使用率 (%)"""
        try:
            return self._process.cpu_percent(interval=0.1)
        except:
            return 0.0

    def get_average_response_time(self) -> float:
        """获取平均响应时间 (ms)"""
        with self._lock:
            if self._metrics['response_times']:
                return sum(self._metrics['response_times']) / len(self._metrics['response_times'])
            return 0.0

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取所有性能指标"""
        with self._lock:
            return {
                'throughput': self.get_throughput(),
                'error_rate': self.get_error_rate(),
                'memory_usage': self.get_memory_usage(),
                'cpu_usage': self.get_cpu_usage(),
                'avg_response_time': self.get_average_response_time(),
                'total_logs': self._stats['total_logs'],
                'error_logs': self._stats['error_logs'],
                'warning_logs': self._stats['warning_logs'],
                'uptime': time.time() - self._stats['start_time'],
            }

    def get_historical_data(self, metric: str, hours: int = 1) -> List[Dict[str, Any]]:
        """获取历史数据"""
        with self._lock:
            cutoff_time = time.time() - (hours * 3600)

            result = []
            for i, timestamp in enumerate(self._timestamps):
                if timestamp >= cutoff_time:
                    if metric in self._metrics and i < len(self._metrics[metric]):
                        result.append({
                            'timestamp': timestamp,
                            'value': self._metrics[metric][i]
                        })

            return result

    def check_thresholds(self) -> List[Dict[str, Any]]:
        """检查阈值告警"""
        alerts = []

        metrics = self.get_performance_metrics()

        # 内存使用阈值
        if metrics['memory_usage'] > 500:  # 500MB
            alerts.append({
                'type': 'memory',
                'severity': 'WARNING',
                'message': f'High memory usage: {metrics["memory_usage"]:.1f}MB',
                'value': metrics['memory_usage']
            })

        # CPU使用阈值
        if metrics['cpu_usage'] > 80:  # 80%
            alerts.append({
                'type': 'cpu',
                'severity': 'WARNING',
                'message': f'High CPU usage: {metrics["cpu_usage"]:.1f}%',
                'value': metrics['cpu_usage']
            })

        # 错误率阈值
        if metrics['error_rate'] > 5:  # 5%
            alerts.append({
                'type': 'error_rate',
                'severity': 'ERROR',
                'message': f'High error rate: {metrics["error_rate"]:.1f}%',
                'value': metrics['error_rate']
            })

        # 响应时间阈值
        if metrics['avg_response_time'] > 1000:  # 1秒
            alerts.append({
                'type': 'response_time',
                'severity': 'WARNING',
                'message': f'Slow response time: {metrics["avg_response_time"]:.1f}ms',
                'value': metrics['avg_response_time']
            })

        return alerts

    def _monitor_loop(self):
        """监控循环"""
        while self._running:
            try:
                current_time = time.time()

                # 采样系统指标
                with self._lock:
                    self._timestamps.append(current_time)
                    self._metrics['memory_usage'].append(self.get_memory_usage())
                    self._metrics['cpu_usage'].append(self.get_cpu_usage())

                    # 计算吞吐量
                    time_diff = current_time - self._stats['last_sample_time']
                    if time_diff >= self.sample_interval:
                        logs_in_interval = self._stats['total_logs'] - getattr(self, '_last_log_count', 0)
                        throughput = logs_in_interval / time_diff if time_diff > 0 else 0
                        self._metrics['log_throughput'].append(throughput)
                        self._last_log_count = self._stats['total_logs']
                        self._stats['last_sample_time'] = current_time

                # 清理过期数据
                self._cleanup_old_data()

                # 等待下一个采样周期
                time.sleep(self.sample_interval)

            except Exception as e:
                print(f"Performance monitor error: {e}")
                time.sleep(1)

    def _cleanup_old_data(self):
        """清理过期数据"""
        cutoff_time = time.time() - self.retention_period

        with self._lock:
            # 找到需要保留的数据索引
            keep_indices = []
            for i, timestamp in enumerate(self._timestamps):
                if timestamp >= cutoff_time:
                    keep_indices.append(i)
                else:
                    break

            if keep_indices:
                # 保留新数据
                self._timestamps = [self._timestamps[i] for i in keep_indices]

                for metric_name in self._metrics:
                    if len(self._metrics[metric_name]) > len(keep_indices):
                        self._metrics[metric_name] = self._metrics[metric_name][-len(keep_indices):]
            else:
                # 所有数据都过期，清空
                self._timestamps.clear()
                for metric_name in self._metrics:
                    self._metrics[metric_name].clear()

    def _check_health(self) -> Dict[str, Any]:
        """检查性能监控器健康状态"""
        try:
            metrics = self.get_performance_metrics()

            # 检查各项指标是否在合理范围内
            issues = []

            if metrics['memory_usage'] > 1000:  # 1GB
                issues.append('High memory usage')

            if metrics['cpu_usage'] > 90:
                issues.append('High CPU usage')

            if metrics['error_rate'] > 10:
                issues.append('High error rate')

            status = 'healthy' if not issues else 'warning'
            if len(issues) > 2:
                status = 'critical'

            return {
                'status': status,
                'issues': issues,
                'metrics': metrics,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }

    def _collect_metrics(self) -> Dict[str, Any]:
        """收集性能指标"""
        return self.get_performance_metrics()

    def _detect_anomalies(self) -> List[Dict[str, Any]]:
        """检测性能异常"""
        return self.check_thresholds()

    def stop(self):
        """停止监控"""
        self._running = False
        if self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=2.0)
