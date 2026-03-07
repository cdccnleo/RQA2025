"""
集成性能监控器

监控服务集成的性能指标。
"""

import logging
import threading
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PerformanceMonitor:
    """性能监控器 - 职责：监控服务调用性能"""

    def __init__(self):
        self._call_stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_response_time': 0.0,
            'avg_response_time': 0.0,
            'min_response_time': float('inf'),
            'max_response_time': 0.0
        }
        self._lock = threading.Lock()

    def record_call(self, response_time: float, success: bool) -> None:
        """记录调用结果"""
        with self._lock:
            self._call_stats['total_calls'] += 1
            if success:
                self._call_stats['successful_calls'] += 1
            else:
                self._call_stats['failed_calls'] += 1

            self._call_stats['total_response_time'] += response_time
            self._call_stats['avg_response_time'] = (
                self._call_stats['total_response_time'] / self._call_stats['total_calls']
            )

            self._call_stats['min_response_time'] = min(
                self._call_stats['min_response_time'], response_time
            )
            self._call_stats['max_response_time'] = max(
                self._call_stats['max_response_time'], response_time
            )

    def get_stats(self) -> Dict[str, Any]:
        """获取性能统计"""
        with self._lock:
            return self._call_stats.copy()

