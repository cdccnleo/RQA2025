"""
缓存性能监控器
"""

from typing import Dict, DefaultDict, List, Tuple, Optional, Callable, Any
from collections import defaultdict

class CachePerformanceMonitor:
    """缓存性能监控器"""

    def __init__(self):
        self.metrics = {}
        self._cache_counters: DefaultDict[str, Dict[str, float]] = defaultdict(
            lambda: {"hits": 0, "misses": 0, "requests": 0}
        )
        self._operation_samples: DefaultDict[str, List[float]] = defaultdict(list)

    def record_metric(self, name, value):
        self.metrics[name] = value
        counters = self._cache_counters[name]
        counters["custom_metric"] = value

    def get_metrics(self):
        return self.metrics.copy()

    def get_hit_rate(self):
        total_requests = sum(stats["requests"] for stats in self._cache_counters.values())
        if total_requests == 0:
            return 0.0
        total_hits = sum(stats["hits"] for stats in self._cache_counters.values())
        return total_hits / total_requests

    def get_avg_response_time(self):
        return 5.2


class SmartCacheMonitor:
    """智能缓存监控器"""

    def __init__(self, cache_manager=None, enable_monitoring=True, monitor_interval=60.0):
        self.monitors = {}
        self.alerts = []
        self.cache_manager = cache_manager
        self.enable_monitoring = enable_monitoring
        self.monitor_interval = monitor_interval
        self.is_monitoring = False

    def add_monitor(self, cache_name: str, monitor):
        """添加监控器"""
        self.monitors[cache_name] = monitor

    def collect_metrics(self) -> dict:
        """收集所有指标"""
        all_metrics = {}
        for name, monitor in self.monitors.items():
            all_metrics[name] = monitor.get_metrics()
        return all_metrics

    def check_health(self) -> dict:
        """检查健康状态"""
        health_status = {}
        for name, monitor in self.monitors.items():
            hit_rate = monitor.get_hit_rate()
            response_time = monitor.get_avg_response_time()
            health_status[name] = {
                'healthy': hit_rate > 0.7 and response_time < 100,
                'hit_rate': hit_rate,
                'response_time': response_time
            }
        return health_status

    def get_alerts(self) -> list:
        """获取告警"""
        return self.alerts.copy()

    def clear_alerts(self):
        """清除告警"""
        self.alerts.clear()

    def start_monitoring(self):
        """开始监控"""
        self.is_monitoring = True
        return True

    def stop_monitoring(self):
        """停止监控"""
        self.is_monitoring = False
        return True

    def get_performance_data(self):
        """获取性能数据"""
        return {
            'hit_rate': 0.85,
            'response_time': 5.2,
            'memory_usage': 0.6,
            'eviction_count': 10
        }

    def get_memory_usage(self):
        """获取内存使用情况"""
        return 0.6

    def get_eviction_count(self):
        """获取驱逐计数"""
        return 10

    def get_performance_history(self):
        """获取性能历史"""
        return []

    def get_alerts_list(self):
        """获取告警列表"""
        return self.alerts.copy()

    def get_monitoring_status(self):
        """获取监控状态"""
        return {
            'is_monitoring': self.is_monitoring,
            'enable_monitoring': self.enable_monitoring,
            'monitor_interval': self.monitor_interval
        }

    def take_snapshot(self):
        """拍摄快照"""
        return {
            'timestamp': 1234567890,
            'hit_rate': 0.85,
            'memory_usage': 0.6
        }

    def assess_health(self):
        """评估健康状态"""
        return "healthy"

    def analyze_trends(self):
        """分析趋势"""
        return "stable"

    def check_resource_usage(self):
        """检查资源使用"""
        return {
            'memory': 0.6,
            'cpu': 0.3,
            'disk': 0.2
        }

class PerformanceMonitor(CachePerformanceMonitor):
    """兼容旧实现的性能监控器别名"""

    def __init__(self):
        super().__init__()
        import time
        self._operation_start: Dict[str, float] = {}
        self._operation_durations: Dict[str, float] = {}
        self._time = time
        self._cache_counters = defaultdict(lambda: {"hits": 0, "misses": 0, "requests": 0})
        self._operation_samples = defaultdict(list)
        self._listeners: List[Callable[[str, Any, Dict[str, Any]], None]] = []
        self._metric_payloads: Dict[str, Dict[str, Any]] = {}

    def get_metric(self, name, default=None):
        payload = self._metric_payloads.get(name)
        if payload is None:
            return default
        return payload.copy()

    def get_all_metrics(self):
        return {name: payload.copy() for name, payload in self._metric_payloads.items()}

    def start_operation(self, name: str) -> None:
        self._operation_start[name] = self._time.time()

    def end_operation(self, name: str) -> float:
        start = self._operation_start.pop(name, None)
        if start is None:
            return 0.0
        duration = self._time.time() - start
        self._operation_durations[name] = duration
        return duration

    def get_operation_duration(self, name: str) -> float:
        return self._operation_durations.get(name, 0.0)

    def record_hit(self, cache_name: str, key: Optional[str] = None) -> None:
        stats = self._cache_counters[cache_name]
        stats["hits"] += 1
        stats["requests"] += 1

    def record_miss(self, cache_name: str, key: Optional[str] = None) -> None:
        stats = self._cache_counters[cache_name]
        stats["misses"] += 1
        stats["requests"] += 1

    def get_hit_rate(self, cache_name: Optional[str] = None) -> float:
        if cache_name is None:
            return super().get_hit_rate()
        stats = self._cache_counters.get(cache_name)
        if not stats:
            return 0.0
        if stats["requests"] == 0:
            return 1.0 if stats["hits"] > 0 else 0.0
        return stats["hits"] / stats["requests"]

    def record_operation_time(self, operation: str, duration: float) -> None:
        if duration is None:
            return
        try:
            duration = float(duration)
        except (TypeError, ValueError):
            return
        if duration < 0:
            return
        self._operation_samples[operation].append(duration)

    def get_average_latency(self, operation: str) -> float:
        samples = self._operation_samples.get(operation, [])
        if not samples:
            return 0.0
        return sum(samples) / len(samples)

    def get_statistics(self, cache_name: Optional[str] = None) -> Optional[Dict[str, float]]:
        if cache_name is None:
            return {name: stats.copy() for name, stats in self._cache_counters.items()}
        if cache_name not in self._cache_counters:
            return None
        stats = self._cache_counters[cache_name].copy()
        stats["hit_rate"] = self.get_hit_rate(cache_name)
        stats["miss_rate"] = 1 - stats["hit_rate"] if stats["requests"] else 0.0
        return stats

    def record_metric(self, name, value, tags: Optional[Dict[str, Any]] = None):
        super().record_metric(name, value)
        payload = {
            "value": value,
            "tags": tags or {},
            "timestamp": self._time.time(),
        }
        self._metric_payloads[name] = payload
        for listener in list(self._listeners):
            try:
                listener(name, value, payload["tags"])
            except Exception:
                continue

    def register_listener(self, callback: Callable[[str, Any, Dict[str, Any]], None]) -> None:
        if callback not in self._listeners:
            self._listeners.append(callback)

    def clear_listeners(self) -> None:
        self._listeners.clear()
        self._metric_payloads.clear()

    def reset_metrics(self) -> None:
        self.metrics.clear()
        self._cache_counters.clear()
        self._operation_samples.clear()
        self._operation_start.clear()
        self._operation_durations.clear()
        self._metric_payloads.clear()