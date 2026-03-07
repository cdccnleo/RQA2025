import sys
import time
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
import threading

import pytest

import src.infrastructure.monitoring.core.performance_monitor as perf_module


class DummyBus:
    def __init__(self):
        self.published = []

    def publish(self, message):
        self.published.append(message)


class DummyMessage:
    def __init__(self, *args, **kwargs):
        self.payload = kwargs.get("payload", {})


class DummyMessageType:
    PERFORMANCE_ALERT = "performance_alert"


@pytest.fixture(autouse=True)
def patch_message_bus(monkeypatch):
    bus = DummyBus()
    fake_module = ModuleType("component_bus")
    fake_module.global_component_bus = bus
    fake_module.Message = DummyMessage
    fake_module.MessageType = DummyMessageType
    monkeypatch.setitem(sys.modules, "src.infrastructure.monitoring.core.component_bus", fake_module)
    return bus


@pytest.fixture
def monitor(monkeypatch):
    # Avoid psutil real calls
    monkeypatch.setattr(perf_module.psutil, "cpu_percent", lambda interval=0: 55.0)

    monkeypatch.setattr(perf_module.psutil, "virtual_memory", lambda: SimpleNamespace(percent=65.0))
    monkeypatch.setattr(perf_module.psutil, "disk_usage", lambda path: SimpleNamespace(percent=45.0))
    monkeypatch.setattr(perf_module.psutil, "net_io_counters", lambda: SimpleNamespace(bytes_sent=100, bytes_recv=200))

    m = perf_module.PerformanceMonitor(collection_interval=0.01)
    yield m
    m.stop()


def test_performance_metrics_stats():
    metrics = perf_module.PerformanceMetrics("latency")
    metrics.add_value(10.0, timestamp=datetime.now() - timedelta(minutes=1))
    metrics.add_value(20.0)
    stats = metrics.get_stats(minutes=5)
    assert stats["count"] == 2
    assert stats["mean"] == pytest.approx(15.0)
    assert stats["max"] == 20.0


def test_record_and_fetch_metrics(monitor):
    monitor.record_metric("cpu_usage", 70.0)
    monitor.record_component_metric("api", "response_time", 0.2)
    stats = monitor.get_metric_stats("cpu_usage")
    assert stats["latest"] == 70.0
    comp_stats = monitor.get_metric_stats("response_time", component="api")
    assert comp_stats["latest"] == 0.2


def test_collect_system_metrics(monkeypatch, monitor):
    monitor._collect_system_metrics()
    metrics = monitor.get_recent_metrics()
    assert metrics["cpu_usage"] == 55.0
    assert metrics["memory_usage"] == 65.0


def test_collect_system_metrics_psutil_failure(monkeypatch, monitor):
    monkeypatch.setattr(perf_module.psutil, "virtual_memory", MagicMock(side_effect=RuntimeError("mem fail")))
    monitor._collect_system_metrics()
    metrics = monitor.get_recent_metrics()
    assert "memory_usage" not in metrics


def test_get_recent_metrics_combines_system_and_components(monkeypatch, monitor):
    monitor.record_metric("cpu_usage", 70.0)
    monitor.record_component_metric("service", "latency", 0.5)
    metrics = monitor.get_recent_metrics()
    assert metrics["cpu_usage"] == 70.0
    assert metrics["service.latency"] == 0.5


def test_check_thresholds_triggers_alert(monkeypatch, monitor, patch_message_bus):
    monitor.record_metric("cpu_usage", 90.0)
    monitor.set_threshold("cpu_usage", 80.0)
    monitor._check_thresholds()
    assert patch_message_bus.published
    payload = patch_message_bus.published[0].payload
    assert payload["metric"] == "cpu_usage"
    assert payload["value"] > 80.0


def test_get_component_performance_report(monitor):
    monitor.record_component_metric("service", "latency", 0.4)
    monitor.record_component_metric("service", "latency", 0.6)
    report = monitor.get_component_performance_report("service")
    assert report["summary"]["total_metrics"] == 1
    assert report["summary"]["worst_performance"] == pytest.approx(0.6)


def test_get_performance_summary_with_alert(monitor):
    monitor.record_metric("cpu_usage", 90.0)
    monitor.record_metric("memory_usage", 40.0)
    monitor.record_metric("disk_usage", 30.0)
    monitor.set_threshold("cpu_usage", 80.0)
    summary = monitor.get_performance_summary()
    assert summary["alerts"]
    assert summary["system_metrics"]["cpu_usage"]["status"] == "warning"


def test_get_performance_summary_without_alert(monitor):
    monitor.record_metric("cpu_usage", 30.0)
    monitor.record_metric("memory_usage", 20.0)
    monitor.record_metric("disk_usage", 10.0)
    summary = monitor.get_performance_summary()
    assert summary["alerts"] == []
    assert summary["system_metrics"]["cpu_usage"]["status"] == "normal"


def test_monitor_performance_decorator(monkeypatch):
    calls = []

    class StubMonitor:
        def record_metric(self, name, value, component=None):
            calls.append((name, value))

    original = perf_module.global_performance_monitor
    perf_module.global_performance_monitor = StubMonitor()

    @perf_module.monitor_performance("demo.operation")
    def success(x):
        time.sleep(0.01)
        return x * 2

    @perf_module.monitor_performance("demo.failure")
    def failure():
        time.sleep(0.01)
        raise RuntimeError("boom")

    assert success(3) == 6
    with pytest.raises(RuntimeError):
        failure()

    perf_module.global_performance_monitor = original
    assert any("demo.operation.execution_time" in name for name, _ in calls)
    assert any(name.endswith("demo.failure.error_count") for name, _ in calls)


def test_start_and_stop_monitor(monkeypatch):
    monitor = perf_module.PerformanceMonitor(collection_interval=0.01)

    def fake_loop(self):
        self.is_running = False

    monkeypatch.setattr(perf_module.PerformanceMonitor, "_monitoring_loop", fake_loop, raising=False)
    monitor.start()
    time.sleep(0.05)
    monitor.start()  # second start should no-op
    monitor.stop()
    monitor.stop()  # stopping twice should be safe
    assert not monitor.is_running


def test_collect_system_metrics_exception(monkeypatch, monitor):
    monkeypatch.setattr(perf_module.psutil, "cpu_percent", MagicMock(side_effect=RuntimeError("fail")))
    monitor._collect_system_metrics()  # should handle internally without raising


def test_monitoring_loop_handles_failure(monkeypatch):
    monitor = perf_module.PerformanceMonitor(collection_interval=0.01)
    monkeypatch.setattr(monitor, "_collect_system_metrics", MagicMock(side_effect=RuntimeError("boom")))
    monkeypatch.setattr(monitor, "_check_thresholds", MagicMock())
    monitor.start()
    time.sleep(0.03)
    monitor.stop()
    assert not monitor.is_running


def test_monitoring_loop_triggers_threshold(monkeypatch, patch_message_bus):
    monitor = perf_module.PerformanceMonitor(collection_interval=0.01)
    monitor.set_threshold("cpu_usage", 80.0)

    def fake_collect():
        monitor.record_metric("cpu_usage", 90.0)

    original_check = monitor._check_thresholds

    def wrapped_check():
        original_check()
        monitor.is_running = False

    monkeypatch.setattr(monitor, "_collect_system_metrics", fake_collect)
    monkeypatch.setattr(monitor, "_check_thresholds", wrapped_check)
    monkeypatch.setattr(perf_module.time, "sleep", lambda interval: None)

    monitor.start()
    monitor.monitor_thread.join(timeout=0.2)
    monitor.stop()

    assert patch_message_bus.published
    payload = patch_message_bus.published[0].payload
    assert payload["metric"] == "cpu_usage"
    assert payload["value"] == 90.0


def test_check_thresholds_no_alert(monkeypatch, monitor, patch_message_bus):
    monitor.record_metric("cpu_usage", 70.0)
    monitor.set_threshold("cpu_usage", 80.0)
    monitor._check_thresholds()
    assert not patch_message_bus.published


def test_global_start_stop_functions(monkeypatch):
    started = []
    stopped = []

    class StubMonitor:
        def start(self):
            started.append(True)

        def stop(self):
            stopped.append(True)

        def get_performance_summary(self):
            return {"summary": True}

    original = perf_module.global_performance_monitor
    perf_module.global_performance_monitor = StubMonitor()

    try:
        perf_module.start_performance_monitoring()
        perf_module.stop_performance_monitoring()
        assert started and stopped
        assert perf_module.get_performance_report() == {"summary": True}
    finally:
        perf_module.global_performance_monitor = original

