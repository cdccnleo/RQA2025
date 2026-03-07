"""
性能监控扩展模块

提供高级性能监控功能，包括性能分析、瓶颈识别、趋势预测等。
"""

import asyncio
import logging
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Deque
from statistics import mean, median, stdev
import psutil

logger = logging.getLogger(__name__)


def _ensure_event_loop() -> asyncio.AbstractEventLoop:
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


@dataclass
class PerformanceMetric:
    """性能指标"""
    name: str
    value: float
    timestamp: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PerformanceStats:
    """性能统计"""
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    avg_time: float = 0.0
    median_time: float = 0.0
    p95_time: float = 0.0
    p99_time: float = 0.0
    error_count: int = 0
    last_updated: float = 0.0

    def update(self, duration: float, is_error: bool = False):
        """更新统计数据"""
        self.count += 1
        self.total_time += duration
        self.min_time = min(self.min_time, duration)
        self.max_time = max(self.max_time, duration)
        self.avg_time = self.total_time / self.count

        if is_error:
            self.error_count += 1

        self.last_updated = time.time()

    def calculate_percentiles(self, durations: List[float]):
        """计算百分位数"""
        if not durations:
            return

        sorted_durations = sorted(durations)
        n = len(sorted_durations)

        self.median_time = sorted_durations[n // 2]

        p95_index = int(n * 0.95)
        self.p95_time = sorted_durations[min(p95_index, n - 1)]

        p99_index = int(n * 0.99)
        self.p99_time = sorted_durations[min(p99_index, n - 1)]


class AdvancedPerformanceMonitor:
    """
    高级性能监控器

    提供性能分析、瓶颈识别、趋势预测等高级功能
    """

    def __init__(self, max_metrics_history: int = 10000, analysis_interval: int = 60):
        self.max_metrics_history = max_metrics_history
        self.analysis_interval = analysis_interval

        # 性能指标存储
        self._metrics: Deque[PerformanceMetric] = deque(maxlen=max_metrics_history)
        self._stats: Dict[str, PerformanceStats] = defaultdict(PerformanceStats)

        # 最近的持续时间用于百分位计算
        self._recent_durations: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=1000))

        # 锁保护并发访问（需要可重入以避免导出/统计嵌套调用死锁）
        self._lock = threading.RLock()
        _ensure_event_loop()
        self._async_lock = asyncio.Lock()

        # 分析线程
        self._analysis_thread = threading.Thread(target=self._analysis_loop, daemon=True)
        self._analysis_thread.start()

        # 性能阈值
        self._thresholds = {
            'slow_operation_threshold': 1.0,  # 慢操作阈值（秒）
            'error_rate_threshold': 0.05,     # 错误率阈值（5%）
            'memory_usage_threshold': 0.8,    # 内存使用阈值（80%）
            'cpu_usage_threshold': 0.9        # CPU使用阈值（90%）
        }

        logger.info("高级性能监控器已启动")

    def record_metric(self, name: str, duration: float, tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None, is_error: bool = False):
        """
        记录性能指标

        Args:
            name: 指标名称
            duration: 持续时间（秒）
            tags: 标签
            metadata: 元数据
            is_error: 是否为错误操作
        """
        with self._lock:
            metric = PerformanceMetric(
                name=name,
                value=duration,
                timestamp=time.time(),
                tags=tags or {},
                metadata=metadata or {}
            )

            self._metrics.append(metric)

            # 更新统计数据
            self._stats[name].update(duration, is_error)

            # 更新最近持续时间
            self._recent_durations[name].append(duration)

            # 实时百分位计算
            if len(self._recent_durations[name]) >= 10:  # 有足够数据时计算
                self._stats[name].calculate_percentiles(list(self._recent_durations[name]))

        # 检查阈值告警（在锁外执行以避免潜在阻塞）
        self._check_thresholds(name, duration, is_error)

    async def record_metric_async(self, name: str, duration: float,
                                 tags: Optional[Dict[str, str]] = None,
                                 metadata: Optional[Dict[str, Any]] = None,
                                 is_error: bool = False):
        """
        异步记录性能指标

        Args:
            name: 指标名称
            duration: 持续时间（秒）
            tags: 标签
            metadata: 元数据
            is_error: 是否为错误操作
        """
        async with self._async_lock:
            # 在线程池中执行同步记录
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                lambda: self.record_metric(name, duration, tags, metadata, is_error)
            )

    def get_stats(self, name: Optional[str] = None) -> Dict[str, Any]:
        """
        获取性能统计

        Args:
            name: 指标名称，为None时返回所有统计

        Returns:
            Dict[str, Any]: 性能统计数据
        """
        with self._lock:
            if name:
                stats = self._stats.get(name)
                return self._build_stats_snapshot_locked(name, stats)
            return self._build_stats_snapshot_locked()

    def get_system_performance(self) -> Dict[str, Any]:
        """
        获取系统性能指标

        Returns:
            Dict[str, Any]: 系统性能数据
        """
        try:
            cpu_percent = psutil.cpu_percent(interval=0)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            return {
                'cpu_usage': cpu_percent,
                'memory_usage': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'disk_usage': disk.percent,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"获取系统性能失败: {e}")
            return {}

    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        识别性能瓶颈

        Returns:
            List[Dict[str, Any]]: 瓶颈列表
        """
        with self._lock:
            return self._collect_bottlenecks_locked()

    def predict_trends(self, name: str, hours: int = 24) -> Dict[str, Any]:
        """
        预测性能趋势

        Args:
            name: 指标名称
            hours: 预测小时数

        Returns:
            Dict[str, Any]: 趋势预测结果
        """
        with self._lock:
            # 获取最近的数据点
            recent_metrics = [
                m for m in self._metrics
                if m.name == name and (time.time() - m.timestamp) < (hours * 3600)
            ]

            if len(recent_metrics) < 10:
                return {'error': 'Insufficient data for trend prediction'}

            # 简单的线性回归预测（避免依赖scipy）
            values = [m.value for m in recent_metrics]
            times = [m.timestamp for m in recent_metrics]

            if len(set(values)) < 2:
                avg_value = mean(values)
                return {
                    'trend': 'stable',
                    'predicted_value': avg_value,
                    'confidence': 0.4,
                    'current_avg': avg_value,
                }

            min_time = min(times)
            shifted_times = [t - min_time for t in times]
            n = len(shifted_times)
            sum_x = sum(shifted_times)
            sum_y = sum(values)
            sum_xx = sum(x * x for x in shifted_times)
            sum_xy = sum(x * y for x, y in zip(shifted_times, values))

            denominator = n * sum_xx - sum_x * sum_x
            if denominator == 0:
                avg_value = mean(values)
                return {
                    'trend': 'stable',
                    'predicted_value': avg_value,
                    'confidence': 0.3,
                    'current_avg': avg_value,
                }

            slope = (n * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - slope * sum_x) / n
            future_time = shifted_times[-1] + hours * 3600
            predicted_value = slope * future_time + intercept
            trend = 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable'

            residuals = [
                y - (slope * x + intercept)
                for x, y in zip(shifted_times, values)
            ]
            variance = sum(r * r for r in residuals) / n if n else 0.0
            confidence = max(0.0, 1.0 / (1.0 + variance))

            return {
                'trend': trend,
                'slope': slope,
                'predicted_value': max(0.0, predicted_value),
                'confidence': min(1.0, confidence),
                'current_avg': mean(values[-10:]) if len(values) >= 10 else mean(values),
            }

    def _check_thresholds(self, name: str, duration: float, is_error: bool):
        """检查阈值并发出告警"""
        # 慢操作检查
        if duration > self._thresholds['slow_operation_threshold']:
            logger.warning(f"Slow operation detected: {name} took {duration:.3f}s")

        # 系统资源检查
        system_perf = self.get_system_performance()
        if system_perf.get('cpu_usage', 0) > self._thresholds['cpu_usage_threshold']:
            logger.warning(f"High CPU usage: {system_perf['cpu_usage']:.1f}%")

        if system_perf.get('memory_usage', 0) > self._thresholds['memory_usage_threshold']:
            logger.warning(f"High memory usage: {system_perf['memory_usage']:.1f}%")

    def _analysis_loop(self):
        """分析循环 - 定期执行性能分析"""
        while True:
            try:
                time.sleep(self.analysis_interval)

                # 执行瓶颈分析
                bottlenecks = self.identify_bottlenecks()
                if bottlenecks:
                    logger.info(f"Performance bottlenecks detected: {len(bottlenecks)}")

                    for bottleneck in bottlenecks[:5]:  # 只记录前5个
                        logger.warning(f"Bottleneck: {bottleneck['metric_name']} - "
                                     f"{len(bottleneck['issues'])} issues")

                # 清理过期指标
                self._cleanup_expired_metrics()

            except Exception as e:
                logger.error(f"Performance analysis error: {e}")
                time.sleep(10)  # 出错时等待较长时间

    def _cleanup_expired_metrics(self):
        """清理过期指标"""
        cutoff_time = time.time() - (24 * 3600)  # 保留24小时的数据

        with self._lock:
            # 清理指标队列
            while self._metrics and self._metrics[0].timestamp < cutoff_time:
                self._metrics.popleft()

            # 清理统计数据（可选，保留统计但清理详细指标）
            # 这里可以根据需要清理旧的统计数据

    def export_metrics(self, format: str = 'json') -> str:
        """
        导出性能指标

        Args:
            format: 导出格式 ('json', 'csv', 'prometheus')

        Returns:
            str: 导出的指标数据
        """
        if format not in {'json', 'csv', 'prometheus'}:
            raise ValueError(f"Unsupported export format: {format}")

        with self._lock:
            stats_snapshot = self._build_stats_snapshot_locked()
            bottlenecks = self._collect_bottlenecks_locked()

        system_perf = self.get_system_performance()

        if format == 'json':
            return self._export_json(stats_snapshot, system_perf, bottlenecks)
        if format == 'csv':
            return self._export_csv(stats_snapshot)
        return self._export_prometheus(stats_snapshot)

    def _export_json(self, stats: Dict[str, Any], system_perf: Dict[str, Any],
                     bottlenecks: List[Dict[str, Any]]) -> str:
        """导出为JSON格式"""
        import json
        data = {
            'stats': stats,
            'system_performance': system_perf,
            'bottlenecks': bottlenecks,
            'export_time': time.time()
        }
        return json.dumps(data, indent=2, default=str)

    def _export_csv(self, stats: Dict[str, Any]) -> str:
        """导出为CSV格式"""
        lines = ['metric_name,count,total_time,min_time,max_time,avg_time,p95_time,error_count,error_rate']

        for name, stat in stats.items():
            error_rate = stat['error_rate']
            lines.append(','.join(map(str, [
                name, stat['count'], stat['total_time'], stat['min_time'],
                stat['max_time'], stat['avg_time'], stat['p95_time'],
                stat['error_count'], error_rate
            ])))

        return '\n'.join(lines)

    def _export_prometheus(self, stats: Dict[str, Dict[str, Any]]) -> str:
        """导出为Prometheus格式"""
        lines = []

        for name, stat in stats.items():
            # 基础指标
            lines.append(f'# HELP {name}_count Total number of operations')
            lines.append(f'# TYPE {name}_count counter')
            lines.append(f'{name}_count {stat["count"]}')

            lines.append(f'# HELP {name}_duration_seconds Total duration in seconds')
            lines.append(f'# TYPE {name}_duration_seconds counter')
            lines.append(f'{name}_duration_seconds {stat["total_time"]}')

            lines.append(f'# HELP {name}_avg_duration_seconds Average duration in seconds')
            lines.append(f'# TYPE {name}_avg_duration_seconds gauge')
            lines.append(f'{name}_avg_duration_seconds {stat["avg_time"]}')

        return '\n'.join(lines)

    def _build_stats_snapshot_locked(self, name: Optional[str] = None,
                                     stats: Optional[PerformanceStats] = None) -> Dict[str, Any]:
        """构建统计信息快照（需在持有锁的情况下调用）"""
        if name:
            if stats is None:
                return {}
            return {
                name: self._format_stats(stats)
            }

        return {
            metric_name: self._format_stats(metric_stats)
            for metric_name, metric_stats in self._stats.items()
        }

    def _format_stats(self, stats: PerformanceStats) -> Dict[str, Any]:
        """格式化统计信息"""
        error_rate = stats.error_count / stats.count if stats.count > 0 else 0
        return {
            'count': stats.count,
            'total_time': stats.total_time,
            'min_time': stats.min_time,
            'max_time': stats.max_time,
            'avg_time': stats.avg_time,
            'median_time': stats.median_time,
            'p95_time': stats.p95_time,
            'p99_time': stats.p99_time,
            'error_count': stats.error_count,
            'error_rate': error_rate,
            'last_updated': stats.last_updated
        }

    def _collect_bottlenecks_locked(self) -> List[Dict[str, Any]]:
        """收集瓶颈信息（需在持有锁的情况下调用）"""
        bottlenecks = []

        for name, stats in self._stats.items():
            issues = []

            # 检查慢操作
            if stats.p95_time > self._thresholds['slow_operation_threshold']:
                issues.append({
                    'type': 'slow_operation',
                    'p95_time': stats.p95_time,
                    'threshold': self._thresholds['slow_operation_threshold']
                })

            # 检查错误率
            error_rate = stats.error_count / stats.count if stats.count > 0 else 0
            if error_rate > self._thresholds['error_rate_threshold']:
                issues.append({
                    'type': 'high_error_rate',
                    'error_rate': error_rate,
                    'threshold': self._thresholds['error_rate_threshold']
                })

            if issues:
                bottlenecks.append({
                    'metric_name': name,
                    'issues': issues,
                    'stats': {
                        'count': stats.count,
                        'avg_time': stats.avg_time,
                        'error_rate': error_rate
                    }
                })

        return bottlenecks


# 向后兼容的别名，保持旧接口可用
PerformanceMonitor = AdvancedPerformanceMonitor