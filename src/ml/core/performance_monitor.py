#!/usr/bin/env python3
"""精简版性能监控模块，为单元测试提供可预测的行为。"""

from __future__ import annotations

import logging
import statistics
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Callable, Deque, Dict, List, Optional

try:  # pragma: no cover
    from src.infrastructure.integration import get_models_adapter as _get_models_adapter
except ImportError:  # pragma: no cover
    class _FallbackModelsAdapter:
        def get_models_logger(self):
            return logging.getLogger(__name__)

    def _get_models_adapter():
        return _FallbackModelsAdapter()


get_models_adapter = _get_models_adapter

try:
    adapter = get_models_adapter()
    logger = adapter.get_models_logger()
except Exception:  # pragma: no cover
    logger = logging.getLogger(__name__)

try:  # pragma: no cover
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None  # type: ignore


@dataclass
class RecordedError:
    timestamp: datetime
    error_type: str
    model_id: Optional[str] = None


class MLPerformanceMetrics:
    """收集并计算核心 ML 指标。"""

    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.inference_latencies: Deque[float] = deque(maxlen=window_size)
        self.inference_throughputs: Deque[float] = deque(maxlen=window_size)
        self.inference_errors: List[RecordedError] = []

        self.model_accuracies: Deque[float] = deque(maxlen=window_size)
        self.model_precisions: Deque[float] = deque(maxlen=window_size)
        self.model_recalls: Deque[float] = deque(maxlen=window_size)
        self.model_f1_scores: Deque[float] = deque(maxlen=window_size)

        self.cpu_usages: Deque[float] = deque(maxlen=window_size)
        self.memory_usages: Deque[float] = deque(maxlen=window_size)

        self.process_latencies: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))
        self.step_latencies: Dict[str, Deque[float]] = defaultdict(lambda: deque(maxlen=window_size))

        self.last_update = datetime.now()

    # --- 记录接口 ---------------------------------------------------------
    def record_inference_latency(self, latency_ms: float, model_id: str = ""):
        self.inference_latencies.append(latency_ms)
        self.last_update = datetime.now()

    def record_inference_throughput(self, throughput: float, model_id: str = ""):
        self.inference_throughputs.append(throughput)
        self.last_update = datetime.now()

    def record_inference_error(self, error_type: str, model_id: str = ""):
        self.inference_errors.append(RecordedError(datetime.now(), error_type, model_id))
        self.last_update = datetime.now()

    def record_model_metrics(self, accuracy: float, precision: float, recall: float, f1_score: float, model_id: str = ""):
        self.model_accuracies.append(accuracy)
        self.model_precisions.append(precision)
        self.model_recalls.append(recall)
        self.model_f1_scores.append(f1_score)
        self.last_update = datetime.now()

    def record_resource_usage(self, cpu_percent: Optional[float] = None, memory_percent: Optional[float] = None):
        if cpu_percent is None or memory_percent is None:
            try:  # pragma: no cover
                import psutil

                cpu_percent = cpu_percent if cpu_percent is not None else psutil.cpu_percent(interval=0)
                mem = psutil.virtual_memory()
                memory_percent = memory_percent if memory_percent is not None else mem.percent
            except Exception:
                cpu_percent = cpu_percent or 0.0
                memory_percent = memory_percent or 0.0

        self.cpu_usages.append(cpu_percent)
        self.memory_usages.append(memory_percent)
        self.last_update = datetime.now()

    def record_process_latency(self, process_id: str, latency_ms: float):
        self.process_latencies[process_id].append(latency_ms)
        self.last_update = datetime.now()

    def record_step_latency(self, step_id: str, latency_ms: float):
        self.step_latencies[step_id].append(latency_ms)
        self.last_update = datetime.now()

    # --- 统计接口 ---------------------------------------------------------
    def get_inference_stats(self) -> Dict[str, Any]:
        if not self.inference_latencies:
            return {}

        error_rate = (
            len(self.inference_errors) / len(self.inference_latencies)
            if self.inference_latencies
            else 0.0
        )
        return {
            "avg_latency_ms": statistics.mean(self.inference_latencies),
            "min_latency_ms": min(self.inference_latencies),
            "max_latency_ms": max(self.inference_latencies),
            "throughput_avg": statistics.mean(self.inference_throughputs) if self.inference_throughputs else 0.0,
            "error_rate": error_rate,
            "total_requests": len(self.inference_latencies),
        }

    def get_model_stats(self) -> Dict[str, Any]:
        if not self.model_accuracies:
            return {}
        return {
            "avg_accuracy": statistics.mean(self.model_accuracies),
            "avg_precision": statistics.mean(self.model_precisions),
            "avg_recall": statistics.mean(self.model_recalls),
            "avg_f1_score": statistics.mean(self.model_f1_scores),
        }

    def get_resource_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        if self.cpu_usages:
            stats["cpu_avg_percent"] = statistics.mean(self.cpu_usages)
            stats["cpu_max_percent"] = max(self.cpu_usages)
        if self.memory_usages:
            stats["memory_avg_percent"] = statistics.mean(self.memory_usages)
            stats["memory_max_percent"] = max(self.memory_usages)
        return stats

    def get_process_stats(self) -> Dict[str, Any]:
        stats: Dict[str, Any] = {}
        for process_id, values in self.process_latencies.items():
            if values:
                stats[process_id] = {
                    "avg_latency_ms": statistics.mean(values),
                    "total_executions": len(values),
                }
        return stats

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        return {
            "timestamp": datetime.now().isoformat(),
            "inference": self.get_inference_stats(),
            "model": self.get_model_stats(),
            "resources": self.get_resource_stats(),
            "processes": self.get_process_stats(),
            "last_update": self.last_update.isoformat(),
        }


class MLPerformanceMonitor:
    """监控器封装，负责调用 metrics 并发出告警。"""

    def __init__(self, collection_interval: int = 60, alert_thresholds: Optional[Dict[str, Any]] = None):
        self.collection_interval = collection_interval
        self.metrics = MLPerformanceMetrics()
        self.alert_thresholds = alert_thresholds or {
            "inference_latency_p95": 1000,
            "inference_error_rate": 0.1,
            "cpu_usage": 90,
            "memory_usage": 90,
        }
        self.alert_callbacks: List[Callable[[Dict[str, Any]], None]] = []
        self.monitoring = False
        self.performance_history: Deque[Dict[str, Any]] = deque(maxlen=100)

    def start_monitoring(self):
        self.monitoring = True

    def stop_monitoring(self):
        self.monitoring = False

    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        self.alert_callbacks.append(callback)

    def _trigger_alert(self, alert: Dict[str, Any]):
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception:  # pragma: no cover
                logger.exception("告警回调执行失败")

    def _check_alerts(self):
        snapshot = self.metrics.get_comprehensive_stats()
        inference = snapshot.get("inference", {})
        if not inference:
            return
        if inference.get("p95_latency_ms", inference.get("avg_latency_ms", 0)) > self.alert_thresholds["inference_latency_p95"]:
            self._trigger_alert({"type": "inference_latency", "level": "warning", "value": inference.get("p95_latency_ms", 0)})
        if inference.get("error_rate", 0) > self.alert_thresholds["inference_error_rate"]:
            self._trigger_alert({"type": "inference_error_rate", "level": "error", "value": inference["error_rate"]})

        resources = snapshot.get("resources", {})
        if resources.get("cpu_max_percent", 0) > self.alert_thresholds["cpu_usage"]:
            self._trigger_alert({"type": "cpu_usage", "level": "warning", "value": resources["cpu_max_percent"]})
        if resources.get("memory_max_percent", 0) > self.alert_thresholds["memory_usage"]:
            self._trigger_alert({"type": "memory_usage", "level": "warning", "value": resources["memory_max_percent"]})

    # 代理方法
    def record_inference_performance(self, latency_ms: float, model_id: str = "", error: Optional[str] = None):
        if error:
            self.metrics.record_inference_error(error, model_id)
        else:
            self.metrics.record_inference_latency(latency_ms, model_id)

    def record_model_performance(self, accuracy: float, precision: float, recall: float, f1_score: float, model_id: str = ""):
        self.metrics.record_model_metrics(accuracy, precision, recall, f1_score, model_id)

    def record_process_performance(self, process_id: str, latency_ms: float):
        self.metrics.record_process_latency(process_id, latency_ms)

    def record_step_performance(self, step_id: str, latency_ms: float):
        self.metrics.record_step_latency(step_id, latency_ms)

    def get_current_stats(self) -> Dict[str, Any]:
        stats = self.metrics.get_comprehensive_stats()
        self._check_alerts()
        self.performance_history.append(stats)
        return stats


_GLOBAL_MONITOR = MLPerformanceMonitor()


def record_inference_performance(latency_ms: float, model_id: str = "", error: Optional[str] = None):
    _GLOBAL_MONITOR.record_inference_performance(latency_ms, model_id, error)


def record_model_performance(accuracy: float, precision: float, recall: float, f1_score: float, model_id: str = ""):
    _GLOBAL_MONITOR.record_model_performance(accuracy, precision, recall, f1_score, model_id)


def get_ml_performance_monitor() -> MLPerformanceMonitor:
    return _GLOBAL_MONITOR


def start_ml_monitoring(config: Optional[Dict[str, Any]] = None) -> bool:
    """
    启动ML性能监控

    Args:
        config: 监控配置

    Returns:
        bool: 启动是否成功
    """
    try:
        return _GLOBAL_MONITOR.start_monitoring(config or {})
    except Exception:
        return False


def stop_ml_monitoring() -> bool:
    """
    停止ML性能监控

    Returns:
        bool: 停止是否成功
    """
    try:
        return _GLOBAL_MONITOR.stop_monitoring()
    except Exception:
        return False


def get_ml_performance_stats() -> Dict[str, Any]:
    """
    获取ML性能统计信息

    Returns:
        Dict[str, Any]: 性能统计数据
    """
    try:
        return _GLOBAL_MONITOR.get_stats()
    except Exception:
        return {}


__all__ = [
    "MLPerformanceMetrics",
    "MLPerformanceMonitor",
    "record_inference_performance",
    "record_model_performance",
    "get_ml_performance_monitor",
    "start_ml_monitoring",
    "stop_ml_monitoring",
    "get_ml_performance_stats",
]

