
import statistics
import time

from .shared_interfaces import ILogger, StandardLogger
from typing import List, Dict, Any, Optional
"""
指标分析器

职责：专门负责分析和统计指标数据
"""


class MetricsAnalyzer:
    """
    指标分析器

    职责：专门负责分析和统计指标数据
    """

    def __init__(self, logger: Optional[ILogger] = None):
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")

    def get_statistics(self, metrics_list: List, time_range: int = 3600) -> Dict[str, Any]:
        """计算统计信息"""
        if not metrics_list:
            return self._get_empty_statistics()

        # 筛选时间范围内的数据
        current_time = time.time()
        cutoff_time = current_time - time_range
        filtered_metrics = [m for m in metrics_list if m.timestamp >= cutoff_time]

        if not filtered_metrics:
            return self._get_empty_statistics()

        try:
            # CPU统计
            cpu_values = [m.cpu_percent for m in filtered_metrics]
            cpu_stats = self._calculate_stats(cpu_values)

            # 内存统计
            memory_values = [m.memory_percent for m in filtered_metrics]
            memory_stats = self._calculate_stats(memory_values)

            # 磁盘统计
            disk_values = [m.disk_usage for m in filtered_metrics]
            disk_stats = self._calculate_stats(disk_values)

            # 进程和线程统计
            process_values = [m.process_count for m in filtered_metrics]
            thread_values = [m.thread_count for m in filtered_metrics]
            process_stats = self._calculate_stats(process_values)
            thread_stats = self._calculate_stats(thread_values)

            # 时间信息
            time_span = filtered_metrics[-1].timestamp - filtered_metrics[0].timestamp

            return {
                'time_range_seconds': time_range,
                'actual_time_span': time_span,
                'sample_count': len(filtered_metrics),
                'cpu': cpu_stats,
                'memory': memory_stats,
                'disk': disk_stats,
                'process': process_stats,
                'thread': thread_stats,
                'timestamp': current_time
            }

        except Exception as e:
            self.logger.log_error(f"计算统计信息失败: {e}")
            return self._get_empty_statistics()

    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """计算基本统计信息"""
        if not values:
            return {'min': 0, 'max': 0, 'avg': 0, 'median': 0, 'std_dev': 0}

        return {
            'min': min(values),
            'max': max(values),
            'avg': statistics.mean(values),
            'median': statistics.median(values),
            'std_dev': statistics.stdev(values) if len(values) > 1 else 0
        }

    def _get_empty_statistics(self) -> Dict[str, Any]:
        """获取空的统计信息"""
        empty_stats = {'min': 0, 'max': 0, 'avg': 0, 'median': 0, 'std_dev': 0}
        return {
            'time_range_seconds': 0,
            'actual_time_span': 0,
            'sample_count': 0,
            'cpu': empty_stats,
            'memory': empty_stats,
            'disk': empty_stats,
            'process': empty_stats,
            'thread': empty_stats,
            'timestamp': time.time()
        }

    def analyze_trends(self, metrics_list: List, window_size: int = 10) -> Dict[str, Any]:
        """分析趋势"""
        if len(metrics_list) < window_size:
            return {'trend': 'insufficient_data', 'confidence': 0.0}

        try:
            # 计算趋势
            recent = metrics_list[-window_size:]
            older = metrics_list[-window_size*2:-
                                 window_size] if len(metrics_list) >= window_size*2 else recent

            recent_avg = statistics.mean([m.cpu_percent for m in recent])
            older_avg = statistics.mean([m.cpu_percent for m in older])

            if abs(recent_avg - older_avg) < 1.0:
                trend = 'stable'
                confidence = 0.8
            elif recent_avg > older_avg:
                trend = 'increasing'
                confidence = min(1.0, (recent_avg - older_avg) / 10.0)
            else:
                trend = 'decreasing'
                confidence = min(1.0, (older_avg - recent_avg) / 10.0)

            return {
                'trend': trend,
                'confidence': confidence,
                'recent_avg': recent_avg,
                'older_avg': older_avg,
                'change_percent': ((recent_avg - older_avg) / older_avg) * 100 if older_avg > 0 else 0
            }

        except Exception as e:
            self.logger.log_error(f"趋势分析失败: {e}")
            return {'trend': 'unknown', 'confidence': 0.0}
