
"""
测试性能监控组件

验证PerformanceMonitor类的功能，包括指标收集、监控控制等
"""

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import sys

import pytest

import src.infrastructure.resource.models.alert_dataclasses as model_alert_dataclasses
import src.infrastructure.resource.models.alert_enums as model_alert_enums
import src.infrastructure.resource.core.shared_interfaces as model_shared_interfaces

sys.modules.setdefault(
    "src.infrastructure.resource.monitoring.alert_dataclasses",
    model_alert_dataclasses,
)
sys.modules.setdefault(
    "src.infrastructure.resource.monitoring.alert_enums",
    model_alert_enums,
)
sys.modules.setdefault(
    "src.infrastructure.resource.monitoring.shared_interfaces",
    model_shared_interfaces,
)

import src.infrastructure.resource.monitoring.performance.performance_monitor_component as monitor_module
from src.infrastructure.resource.monitoring.performance.performance_monitor_component import (
    MonitoringPerformanceMonitor,
)
from src.infrastructure.resource.models.alert_dataclasses import PerformanceMetrics


class DummyLogger:
    def __init__(self):
        self.infos: List[str] = []
        self.warnings: List[str] = []
        self.errors: List[str] = []

    def log_info(self, message: str, **kwargs):
        self.infos.append(message)

    def log_warning(self, message: str, **kwargs):
        self.warnings.append(message)

    def log_error(self, message: str, error: Optional[Exception] = None, **kwargs):
        if error:
            self.errors.append(f"{message}: {error}")
        else:
            self.errors.append(message)


class DummyErrorHandler:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def handle_error(self, error: Exception, context: Optional[Any] = None):
        self.calls.append({"error": error, "context": context})


class DummyThread:
    def __init__(self, target=None, daemon=True):
        self._target = target
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True
        if callable(self._target):
            self._target()

    def join(self):
        pass


class FakePsutil:
    @staticmethod
    def cpu_percent(interval=1):
        return 33.3

    @staticmethod
    def virtual_memory():
        return type("Mem", (), {"percent": 44.4, "used": 1024, "total": 4096})

    @staticmethod
    def disk_usage(_path):
        return type("Disk", (), {"percent": 55.5, "used": 2048, "total": 8192})

    @staticmethod
    def net_io_counters():
        return type("Net", (), {"bytes_sent": 123, "bytes_recv": 456})


@pytest.fixture
def monitor(monkeypatch):
    logger = DummyLogger()
    error_handler = DummyErrorHandler()

    monkeypatch.setattr(monitor_module, "StandardLogger", lambda name=None: logger)
    monkeypatch.setattr(monitor_module, "BaseErrorHandler", lambda: error_handler)
    monkeypatch.setattr(monitor_module.threading, "Thread", DummyThread)
    monkeypatch.setattr(monitor_module.threading, "active_count", lambda: 7)
    monkeypatch.setattr(monitor_module, "psutil", FakePsutil)
    monkeypatch.setattr(monitor_module.requests, "get", lambda *args, **kwargs: object())

    monitor_instance = MonitoringPerformanceMonitor(update_interval=0)
    monitor_instance.logger = logger
    monitor_instance.error_handler = error_handler
    return monitor_instance, logger, error_handler


def make_metrics(**kwargs):
    base = {
        "cpu_usage": 10.0,
        "memory_usage": 20.0,
        "disk_usage": 30.0,
        "network_latency": 40.0,
        "test_execution_time": 50.0,
        "test_success_rate": 0.8,
        "active_threads": 5,
        "timestamp": datetime.now(),
    }
    base.update(kwargs)
    return PerformanceMetrics(**base)


def test_start_stop_monitoring(monkeypatch, monitor):
    monitor_instance, logger, _ = monitor

    def fake_loop(self):
        self.loop_called = True
        self.monitoring = False

    monkeypatch.setattr(MonitoringPerformanceMonitor, "_monitor_loop", fake_loop, raising=False)

    monitor_instance.start_monitoring()
    assert monitor_instance.monitoring is False
    assert getattr(monitor_instance, "loop_called", False) is True
    assert any("已启动" in msg for msg in logger.infos)

    monitor_instance.stop_monitoring()
    assert any("已停止" in msg for msg in logger.infos)


def test_collect_metrics_updates_state(monitor):
    monitor_instance, _, _ = monitor

    metrics = monitor_instance._collect_metrics()

    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.cpu_usage == pytest.approx(33.3)
    assert monitor_instance.get_current_metrics() is metrics
    assert len(monitor_instance.metrics_history) == 1

    history = monitor_instance.get_metrics_history(minutes=1)
    assert len(history) == 1

    average = monitor_instance.get_average_metrics(minutes=1)
    assert isinstance(average, PerformanceMetrics)
    assert average.cpu_usage == pytest.approx(metrics.cpu_usage)


def test_collect_metrics_fallback_on_import_error(monkeypatch, monitor):
    monitor_instance, _, _ = monitor

    class BrokenPsutil:
        @staticmethod
        def cpu_percent(*_args, **_kwargs):
            raise ImportError("cpu")

    monkeypatch.setattr(monitor_module, "psutil", BrokenPsutil)

    metrics = monitor_instance._collect_metrics()
    assert isinstance(metrics, PerformanceMetrics)
    assert len(monitor_instance.metrics_history) == 1  # fallback appended


def test_history_filtering_and_average(monitor):
    monitor_instance, _, _ = monitor

    recent = make_metrics(cpu_usage=60.0)
    old = make_metrics(cpu_usage=20.0, timestamp=datetime.now() - timedelta(minutes=30))

    monitor_instance.add_performance_data(old)
    monitor_instance.add_performance_data(recent)

    filtered = monitor_instance.get_metrics_history(minutes=10)
    assert all(item.timestamp > datetime.now() - timedelta(minutes=10) for item in filtered)

    avg = monitor_instance.get_average_metrics(minutes=60)
    assert avg.cpu_usage > 0


def test_alert_callbacks_and_error_handling(monitor):
    monitor_instance, _, error_handler = monitor
    alerts: List[Dict[str, Any]] = []

    monitor_instance.add_alert_callback(alerts.append)
    monitor_instance.add_alert_callback(alerts.append)  # duplicate ignored

    def failing_callback(_alert):
        raise ValueError("boom")

    monitor_instance.add_alert_callback(failing_callback)

    high_metrics = make_metrics(cpu_usage=95.0, memory_usage=92.0, disk_usage=96.0, network_latency=250.0, active_threads=150)
    monitor_instance._check_smart_alerts(high_metrics)

    assert len(alerts) >= 1
    assert any(call["context"].startswith("告警回调执行失败") for call in error_handler.calls)


def test_predict_performance_and_report(monitor):
    monitor_instance, _, _ = monitor

    empty_prediction = monitor_instance.predict_performance()
    assert empty_prediction["confidence"] == 0.0

    metrics = make_metrics(test_success_rate=0.4, test_execution_time=80.0)
    monitor_instance.add_performance_data(metrics)
    monitor_instance.add_performance_data(make_metrics(test_success_rate=0.6, test_execution_time=60.0))

    prediction = monitor_instance.predict_performance()
    assert 0.4 <= prediction["predicted_hit_rate"] <= 0.6
    assert prediction["confidence"] >= 0.3

    monitor_instance.current_metrics = make_metrics(cpu_usage=85.0, memory_usage=81.0, disk_usage=91.0, network_latency=120.0)
    report = monitor_instance.get_performance_report()
    assert report["summary"]["current_cpu_usage"] == pytest.approx(85.0)
    assert any("优化CPU" in rec or "CPU" in rec for rec in report["recommendations"])


def test_detect_anomaly_levels(monitor):
    monitor_instance, _, _ = monitor

    normal = monitor_instance.detect_anomaly(make_metrics())
    assert normal["severity"] == "normal"

    warning = monitor_instance.detect_anomaly(make_metrics(cpu_usage=85.0))
    assert warning["severity"] == "warning"

    critical = monitor_instance.detect_anomaly(make_metrics(memory_usage=95.0, test_success_rate=0.2))
    assert critical["severity"] == "critical"


def test_monitoring_stats_and_helper_methods(monitor):
    monitor_instance, _, _ = monitor
    monitor_instance._collect_metrics()
    monitor_instance.current_metrics.test_success_rate = 0.75
    monitor_instance.current_metrics.test_execution_time = 120.0

    stats = monitor_instance.get_monitoring_stats()
    assert stats["total_metrics"] >= 1
    assert stats["update_interval"] == 0

    assert monitor_instance._get_hit_rate() == pytest.approx(0.75)
    assert monitor_instance._get_eviction_rate() == 0.0
    assert monitor_instance._get_miss_penalty() == pytest.approx(30.0)
