#!/usr/bin/env python3
"""
RQA2025 基础设施层统计收集器

负责收集各种监控统计信息，提供统一的统计数据收集接口。
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
import statistics

from .performance_monitor import monitor_performance

from ..core.parameter_objects import LoggerPoolStatsConfig


class StatsCollector:
    """
    统计收集器

    负责收集和计算各种统计信息，支持历史数据管理和趋势分析。
    """

    def __init__(
        self,
        pool_name: str = "default_pool",
        config: Optional[LoggerPoolStatsConfig] = None,
    ):
        """
        初始化统计收集器

        Args:
            pool_name: 池名称
            config: 统计配置
        """
        self.pool_name = pool_name
        self.config = config or LoggerPoolStatsConfig()

        # 数据存储
        self.current_stats: Optional[Dict[str, Any]] = None
        self.history_stats: List[Dict[str, Any]] = []
        self.max_history_size = 1000

        # 访问时间跟踪
        self.access_times: List[float] = []
        self.max_access_times_size = 1000

    @monitor_performance("StatsCollector", "collect_stats")
    def collect_stats(self) -> Optional[Dict[str, Any]]:
        """
        收集统计信息

        Returns:
            Optional[Dict[str, Any]]: 收集到的统计信息
        """
        try:
            # 这里应该从实际的数据源收集统计信息
            # 现在返回模拟数据
            stats = self._collect_mock_stats()

            # 更新当前统计
            self.current_stats = stats

            # 添加到历史记录
            self._add_to_history(stats)

            return stats

        except Exception as e:
            print(f"收集统计信息失败: {e}")
            return None

    def get_current_stats(self) -> Optional[Dict[str, Any]]:
        """
        获取当前统计信息

        Returns:
            Optional[Dict[str, Any]]: 当前统计信息
        """
        return self.current_stats

    def get_history_stats(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        获取历史统计信息

        Args:
            limit: 返回的最大记录数

        Returns:
            List[Dict[str, Any]]: 历史统计信息列表
        """
        return self.history_stats[-limit:] if limit > 0 else self.history_stats

    def get_access_times(self, limit: int = 100) -> List[float]:
        """
        获取访问时间记录

        Args:
            limit: 返回的最大记录数

        Returns:
            List[float]: 访问时间列表
        """
        return self.access_times[-limit:] if limit > 0 else self.access_times

    def record_access_time(self, access_time: float):
        """
        记录访问时间

        Args:
            access_time: 访问时间（秒）
        """
        self.access_times.append(access_time)

        # 限制列表大小
        if len(self.access_times) > self.max_access_times_size:
            self.access_times = self.access_times[-self.max_access_times_size:]

    def calculate_percentiles(self, data: List[float], percentiles: List[float]) -> Dict[str, float]:
        """
        计算百分位数

        Args:
            data: 数据列表
            percentiles: 要计算的百分位数列表

        Returns:
            Dict[str, float]: 百分位数结果
        """
        if not data:
            return {}

        try:
            sorted_data = sorted(data)
            result = {}

            for p in percentiles:
                if 0 <= p <= 100:
                    index = int(p / 100 * (len(sorted_data) - 1))
                    result[f'p{int(p)}'] = sorted_data[index]
                else:
                    result[f'p{int(p)}'] = 0.0

            return result

        except Exception as e:
            print(f"计算百分位数失败: {e}")
            return {}

    def analyze_trends(self, metric_name: str, window_size: int = 10) -> Dict[str, Any]:
        """
        分析趋势

        Args:
            metric_name: 指标名称
            window_size: 分析窗口大小

        Returns:
            Dict[str, Any]: 趋势分析结果
        """
        try:
            # 获取最近的指标数据
            recent_stats = self.history_stats[-window_size:]
            values = []

            for stat in recent_stats:
                if metric_name in stat:
                    values.append(stat[metric_name])

            if len(values) < 2:
                return {'trend': 'insufficient_data'}

            # 计算趋势
            slope = self._calculate_slope(values)
            avg_change = sum(values[i] - values[i-1] for i in range(1, len(values))) / (len(values) - 1)

            return {
                'trend': 'increasing' if slope > 0.01 else 'decreasing' if slope < -0.01 else 'stable',
                'slope': slope,
                'avg_change': avg_change,
                'current_value': values[-1] if values else 0,
                'change_percent': ((values[-1] - values[0]) / values[0] * 100) if values[0] != 0 else 0
            }

        except Exception as e:
            print(f"趋势分析失败: {e}")
            return {'trend': 'error', 'error': str(e)}

    def _collect_mock_stats(self) -> Dict[str, Any]:
        """
        收集模拟统计信息

        Returns:
            Dict[str, Any]: 模拟统计数据
        """
        import random
        import time

        return {
            'pool_name': self.pool_name,
            'pool_size': random.randint(10, 50),
            'max_size': 100,
            'created_count': random.randint(100, 1000),
            'hit_count': random.randint(1000, 10000),
            'hit_rate': random.uniform(0.7, 0.95),
            'memory_usage_mb': random.uniform(50, 200),
            'avg_access_time': random.uniform(0.001, 0.01),
            'timestamp': time.time(),
            'collection_time': datetime.now().isoformat()
        }

    def _add_to_history(self, stats: Dict[str, Any]):
        """
        添加到历史记录

        Args:
            stats: 统计信息
        """
        self.history_stats.append(stats)

        # 限制历史记录大小
        if len(self.history_stats) > self.max_history_size:
            self.history_stats = self.history_stats[-self.max_history_size:]

    def _calculate_slope(self, values: List[float]) -> float:
        """
        计算斜率（简单线性回归）

        Args:
            values: 值列表

        Returns:
            float: 斜率
        """
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_sum = sum(range(n))
        y_sum = sum(values)
        xy_sum = sum(i * val for i, val in enumerate(values))
        x_squared_sum = sum(i * i for i in range(n))

        denominator = n * x_squared_sum - x_sum * x_sum
        if denominator == 0:
            return 0.0

        return (n * xy_sum - x_sum * y_sum) / denominator
