"""
Prometheus指标暴露

提供调度器的Prometheus格式指标，用于监控和告警
"""

import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import threading


class MetricType(Enum):
    """指标类型"""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricValue:
    """指标值"""
    value: float
    timestamp: float = field(default_factory=time.time)
    labels: Dict[str, str] = field(default_factory=dict)


class PrometheusMetrics:
    """
    Prometheus指标收集器

    提供调度器的各项指标：
    - 任务计数器（提交、完成、失败）
    - 任务执行时间直方图
    - 队列大小仪表盘
    - 工作进程状态
    - 调度器运行状态

    指标命名规范：scheduler_<component>_<metric>_<unit>
    """

    def __init__(self):
        """初始化Prometheus指标收集器"""
        # 计数器（只增不减）
        self._counters: Dict[str, Dict[str, MetricValue]] = {}
        self._counter_lock = threading.Lock()

        # 仪表盘（可增可减）
        self._gauges: Dict[str, Dict[str, MetricValue]] = {}
        self._gauge_lock = threading.Lock()

        # 直方图（分布统计）
        self._histograms: Dict[str, Dict[str, List[float]]] = {}
        self._histogram_buckets = [0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        self._histogram_lock = threading.Lock()

        # 元数据
        self._metric_help: Dict[str, str] = {}
        self._metric_type: Dict[str, MetricType] = {}

        # 初始化标准指标
        self._init_standard_metrics()

    def _init_standard_metrics(self):
        """初始化标准指标"""
        # 任务计数器
        self.register_counter(
            "scheduler_tasks_submitted_total",
            "Total number of tasks submitted"
        )
        self.register_counter(
            "scheduler_tasks_completed_total",
            "Total number of tasks completed"
        )
        self.register_counter(
            "scheduler_tasks_failed_total",
            "Total number of tasks failed"
        )
        self.register_counter(
            "scheduler_tasks_cancelled_total",
            "Total number of tasks cancelled"
        )
        self.register_counter(
            "scheduler_tasks_timeout_total",
            "Total number of tasks timed out"
        )
        self.register_counter(
            "scheduler_tasks_retried_total",
            "Total number of task retries"
        )

        # 仪表盘
        self.register_gauge(
            "scheduler_tasks_running",
            "Number of tasks currently running"
        )
        self.register_gauge(
            "scheduler_tasks_pending",
            "Number of tasks pending in queue"
        )
        self.register_gauge(
            "scheduler_workers_active",
            "Number of active workers"
        )
        self.register_gauge(
            "scheduler_workers_idle",
            "Number of idle workers"
        )
        self.register_gauge(
            "scheduler_jobs_enabled",
            "Number of enabled scheduled jobs"
        )
        self.register_gauge(
            "scheduler_up",
            "Whether the scheduler is up (1) or down (0)"
        )

        # 直方图
        self.register_histogram(
            "scheduler_task_execution_duration_seconds",
            "Task execution duration in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60]
        )
        self.register_histogram(
            "scheduler_task_wait_duration_seconds",
            "Task wait time in queue in seconds",
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
        )

    def register_counter(self, name: str, help_text: str):
        """
        注册计数器指标

        Args:
            name: 指标名称
            help_text: 帮助文本
        """
        self._metric_help[name] = help_text
        self._metric_type[name] = MetricType.COUNTER
        if name not in self._counters:
            self._counters[name] = {}

    def register_gauge(self, name: str, help_text: str):
        """
        注册仪表盘指标

        Args:
            name: 指标名称
            help_text: 帮助文本
        """
        self._metric_help[name] = help_text
        self._metric_type[name] = MetricType.GAUGE
        if name not in self._gauges:
            self._gauges[name] = {}

    def register_histogram(
        self,
        name: str,
        help_text: str,
        buckets: Optional[List[float]] = None
    ):
        """
        注册直方图指标

        Args:
            name: 指标名称
            help_text: 帮助文本
            buckets: 分桶边界
        """
        self._metric_help[name] = help_text
        self._metric_type[name] = MetricType.HISTOGRAM
        if name not in self._histograms:
            self._histograms[name] = {}
        if buckets:
            self._histogram_buckets = buckets

    def increment_counter(
        self,
        name: str,
        value: float = 1,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        增加计数器

        Args:
            name: 指标名称
            value: 增加值
            labels: 标签
        """
        if name not in self._counters:
            return

        label_key = self._labels_to_key(labels or {})

        with self._counter_lock:
            if label_key not in self._counters[name]:
                self._counters[name][label_key] = MetricValue(
                    value=0,
                    labels=labels or {}
                )
            self._counters[name][label_key].value += value
            self._counters[name][label_key].timestamp = time.time()

    def set_gauge(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        设置仪表盘值

        Args:
            name: 指标名称
            value: 值
            labels: 标签
        """
        if name not in self._gauges:
            return

        label_key = self._labels_to_key(labels or {})

        with self._gauge_lock:
            self._gauges[name][label_key] = MetricValue(
                value=value,
                labels=labels or {},
                timestamp=time.time()
            )

    def observe_histogram(
        self,
        name: str,
        value: float,
        labels: Optional[Dict[str, str]] = None
    ):
        """
        记录直方图观测值

        Args:
            name: 指标名称
            value: 观测值
            labels: 标签
        """
        if name not in self._histograms:
            return

        label_key = self._labels_to_key(labels or {})

        with self._histogram_lock:
            if label_key not in self._histograms[name]:
                self._histograms[name][label_key] = []
            self._histograms[name][label_key].append(value)

    def _labels_to_key(self, labels: Dict[str, str]) -> str:
        """
        将标签字典转换为键

        Args:
            labels: 标签字典

        Returns:
            str: 键
        """
        if not labels:
            return ""
        return ",".join(f"{k}={v}" for k, v in sorted(labels.items()))

    def _key_to_labels(self, key: str) -> Dict[str, str]:
        """
        将键转换为标签字典

        Args:
            key: 键

        Returns:
            Dict[str, str]: 标签字典
        """
        if not key:
            return {}
        labels = {}
        for pair in key.split(","):
            if "=" in pair:
                k, v = pair.split("=", 1)
                labels[k] = v
        return labels

    def generate_metrics(self) -> str:
        """
        生成Prometheus格式的指标文本

        Returns:
            str: Prometheus格式的指标
        """
        lines = []

        # 计数器
        for name, values in self._counters.items():
            lines.append(f"# HELP {name} {self._metric_help.get(name, '')}")
            lines.append(f"# TYPE {name} counter")
            for label_key, metric in values.items():
                labels_str = self._format_labels(metric.labels)
                lines.append(f"{name}{labels_str} {metric.value}")
            lines.append("")

        # 仪表盘
        for name, values in self._gauges.items():
            lines.append(f"# HELP {name} {self._metric_help.get(name, '')}")
            lines.append(f"# TYPE {name} gauge")
            for label_key, metric in values.items():
                labels_str = self._format_labels(metric.labels)
                lines.append(f"{name}{labels_str} {metric.value}")
            lines.append("")

        # 直方图
        for name, values in self._histograms.items():
            lines.append(f"# HELP {name} {self._metric_help.get(name, '')}")
            lines.append(f"# TYPE {name} histogram")

            for label_key, observations in values.items():
                labels = self._key_to_labels(label_key)
                buckets = self._calculate_histogram_buckets(observations)

                # 输出分桶
                for bucket_upper, count in buckets.items():
                    bucket_labels = {**labels, "le": str(bucket_upper)}
                    labels_str = self._format_labels(bucket_labels)
                    lines.append(f"{name}_bucket{labels_str} {count}")

                # 总和
                sum_value = sum(observations)
                labels_str = self._format_labels(labels)
                lines.append(f"{name}_sum{labels_str} {sum_value}")

                # 计数
                lines.append(f"{name}_count{labels_str} {len(observations)}")

            lines.append("")

        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """
        格式化标签

        Args:
            labels: 标签字典

        Returns:
            str: 格式化后的标签字符串
        """
        if not labels:
            return ""
        label_strs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(label_strs) + "}"

    def _calculate_histogram_buckets(
        self,
        observations: List[float]
    ) -> Dict[str, int]:
        """
        计算直方图分桶

        Args:
            observations: 观测值列表

        Returns:
            Dict[str, int]: 分桶统计
        """
        buckets = {str(b): 0 for b in self._histogram_buckets}
        buckets["+Inf"] = 0

        for obs in observations:
            for bucket in self._histogram_buckets:
                if obs <= bucket:
                    buckets[str(bucket)] += 1
            buckets["+Inf"] += 1

        return buckets

    def get_metric_value(
        self,
        name: str,
        labels: Optional[Dict[str, str]] = None
    ) -> Optional[float]:
        """
        获取指标值

        Args:
            name: 指标名称
            labels: 标签

        Returns:
            Optional[float]: 指标值
        """
        label_key = self._labels_to_key(labels or {})

        if name in self._counters and label_key in self._counters[name]:
            return self._counters[name][label_key].value

        if name in self._gauges and label_key in self._gauges[name]:
            return self._gauges[name][label_key].value

        return None

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        获取所有指标

        Returns:
            Dict[str, Any]: 所有指标数据
        """
        return {
            "counters": {
                name: {
                    key: {"value": m.value, "labels": m.labels}
                    for key, m in values.items()
                }
                for name, values in self._counters.items()
            },
            "gauges": {
                name: {
                    key: {"value": m.value, "labels": m.labels}
                    for key, m in values.items()
                }
                for name, values in self._gauges.items()
            },
            "histograms": {
                name: {
                    key: {"count": len(obs), "sum": sum(obs)}
                    for key, obs in values.items()
                }
                for name, values in self._histograms.items()
            }
        }

    def reset(self):
        """重置所有指标"""
        with self._counter_lock:
            self._counters.clear()
        with self._gauge_lock:
            self._gauges.clear()
        with self._histogram_lock:
            self._histograms.clear()
        self._init_standard_metrics()


# 全局实例
_prometheus_metrics_instance: Optional[PrometheusMetrics] = None
_prometheus_metrics_lock = threading.Lock()


def get_prometheus_metrics() -> PrometheusMetrics:
    """
    获取Prometheus指标收集器实例（单例）

    Returns:
        PrometheusMetrics: Prometheus指标收集器实例
    """
    global _prometheus_metrics_instance

    if _prometheus_metrics_instance is None:
        with _prometheus_metrics_lock:
            if _prometheus_metrics_instance is None:
                _prometheus_metrics_instance = PrometheusMetrics()

    return _prometheus_metrics_instance
