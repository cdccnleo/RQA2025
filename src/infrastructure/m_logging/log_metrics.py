import time
from typing import Dict, Any, Optional
from collections import defaultdict
from threading import Lock
from prometheus_client import Counter, Gauge, Histogram
from dataclasses import dataclass
import json
import requests

@dataclass
class LogMetricsConfig:
    """日志监控配置"""
    push_interval: int = 60  # 推送间隔(秒)
    push_url: Optional[str] = None  # 监控数据推送URL
    enable_prometheus: bool = True  # 是否启用Prometheus指标

class LogMetrics:
    """日志监控指标收集器"""

    _instance = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """单例模式"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self, config: LogMetricsConfig = None):
        """初始化指标收集器"""
        if self._initialized:
            return

        self.config = config or LogMetricsConfig()
        self._metrics = defaultdict(int)
        self._lock = Lock()
        self._last_push_time = 0

        # Prometheus指标
        if self.config.enable_prometheus:
            self._init_prometheus_metrics()

        self._initialized = True

    def _init_prometheus_metrics(self):
        """初始化Prometheus指标"""
        self.log_total = Counter(
            'log_messages_total',
            'Total log messages',
            ['level', 'logger']
        )

        self.log_sampled = Counter(
            'log_messages_sampled',
            'Sampled log messages',
            ['level', 'logger']
        )

        self.log_latency = Histogram(
            'log_processing_latency_seconds',
            'Log processing latency',
            ['level']
        )

        self.log_buffer_size = Gauge(
            'log_buffer_size',
            'Current log buffer size'
        )

    def record(self, level: str, logger: str, sampled: bool = False, latency: float = None):
        """记录日志指标"""
        with self._lock:
            self._metrics['total'] += 1
            self._metrics[f'level_{level}'] += 1
            self._metrics[f'logger_{logger}'] += 1

            if sampled:
                self._metrics['sampled'] += 1
                self._metrics[f'sampled_level_{level}'] += 1

            if latency is not None:
                self._metrics['total_latency'] += latency
                self._metrics['count_latency'] += 1

        # Prometheus指标
        if self.config.enable_prometheus:
            self.log_total.labels(level=level, logger=logger).inc()
            if sampled:
                self.log_sampled.labels(level=level, logger=logger).inc()
            if latency is not None:
                self.log_latency.labels(level=level).observe(latency)

    def get_metrics(self) -> Dict[str, Any]:
        """获取当前指标快照"""
        with self._lock:
            metrics = dict(self._metrics)

            # 计算平均延迟
            if metrics.get('count_latency', 0) > 0:
                metrics['avg_latency'] = metrics['total_latency'] / metrics['count_latency']

            return metrics

    def push_metrics(self):
        """推送指标到监控系统"""
        if not self.config.push_url:
            return

        current_time = time.time()
        if current_time - self._last_push_time < self.config.push_interval:
            return

        try:
            metrics = self.get_metrics()
            response = requests.post(
                self.config.push_url,
                json=metrics,
                timeout=5
            )
            response.raise_for_status()
            self._last_push_time = current_time
        except Exception as e:
            pass  # 静默失败，避免影响主流程

    def reset(self):
        """重置指标计数器"""
        with self._lock:
            self._metrics.clear()

class LogMetricsSingleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = LogMetrics()
        return cls._instance

# 获取单例实例
def get_metrics_instance():
    """获取LogMetrics单例实例"""
    return LogMetricsSingleton()

def record(level: str, logger: str, sampled: bool = False, latency: float = None):
    """快捷记录日志指标
    
    Args:
        level: 日志级别(DEBUG/INFO/WARNING/ERROR)
        logger: 记录器名称
        sampled: 是否采样日志
        latency: 处理延迟(秒)
    """
    get_metrics_instance().record(level, logger, sampled, latency)

def push_metrics():
    """快捷推送日志指标到监控系统"""
    get_metrics_instance().push_metrics()
