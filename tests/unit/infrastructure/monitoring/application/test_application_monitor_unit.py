import time
from collections import deque
from unittest.mock import MagicMock

import pytest

import src.infrastructure.monitoring.application.application_monitor as app_monitor_module
from src.infrastructure.monitoring.application.application_monitor import ApplicationMonitor


@pytest.fixture(autouse=True)
def stub_psutil(monkeypatch):
    """Provide deterministic psutil responses for the tests."""

    class FakeVirtualMemory:
        total = 16 * 1024 ** 3
        percent = 42.0
        used = 6 * 1024 ** 3
        available = 10 * 1024 ** 3

    class FakeDisk:
        percent = 18.0
        used = 100
        free = 50
        total = 150

    class FakeNet:
        bytes_sent = 111
        bytes_recv = 222

    monkeypatch.setattr(app_monitor_module.psutil, "cpu_count", lambda logical=False: 8 if logical else 4)
    monkeypatch.setattr(app_monitor_module.psutil, "virtual_memory", lambda: FakeVirtualMemory)
    monkeypatch.setattr(app_monitor_module.psutil, "disk_usage", lambda path: FakeDisk)
    monkeypatch.setattr(app_monitor_module.psutil, "net_io_counters", lambda: FakeNet)

    cpu_values = iter([55.0, 20.0])

    def fake_cpu_percent(interval=1):
        return next(cpu_values, 20.0)

    monkeypatch.setattr(app_monitor_module.psutil, "cpu_percent", fake_cpu_percent)
    yield


def test_initialization_collects_system_metrics():
    monitor = ApplicationMonitor(app_name="demo")
    assert monitor.get_metric("system.cpu_count")["value"] == 4
    assert monitor.get_metric("system.memory.total")["value"] == 16 * 1024 ** 3


def test_collect_performance_metrics_triggers_alerts(monkeypatch, caplog):
    monitor = ApplicationMonitor(app_name="alert-demo")
    monitor.performance_history = deque(maxlen=10)
    monitor.set_alert_threshold("cpu_percent", 10.0)  # trigger alert with fake data
    monitor.set_alert_threshold("memory_percent", 10.0)
    monitor.set_alert_threshold("disk_usage_percent", 10.0)

    with caplog.at_level("WARNING"):
        monitor.collect_performance_metrics()

    assert any("性能告警" in record.msg for record in caplog.records)
    assert monitor.performance_history
    assert monitor.get_metric("performance.cpu_percent")["value"] == 55.0


def test_collect_performance_metrics_handles_exception(monkeypatch, caplog):
    monitor = ApplicationMonitor()
    monkeypatch.setattr(app_monitor_module.psutil, "cpu_percent", MagicMock(side_effect=RuntimeError("boom")))

    with caplog.at_level("ERROR"):
        monitor.collect_performance_metrics()

    assert any("Failed to collect performance metrics" in record.msg for record in caplog.records)


def test_get_recent_metrics_sorted_limit():
    monitor = ApplicationMonitor(app_name="recent")
    monitor.metrics.clear()
    now = time.time()
    monitor.metrics = {
        "m1": {"value": 1, "timestamp": now - 1, "tags": {}},
        "m2": {"value": 2, "timestamp": now, "tags": {}},
    }

    recent = monitor.get_recent_metrics(limit=1)
    assert recent[0]["name"] == "m2"


def test_monitoring_loop_executes_once(monkeypatch):
    monitor = ApplicationMonitor()
    calls = {"collect": 0}

    def fake_collect():
        calls["collect"] += 1
        monitor.monitoring_active = False  # stop after first run

    monitor.collect_performance_metrics = fake_collect  # type: ignore
    monitor.monitoring_active = True

    monitor._monitoring_loop(interval=0)
    assert calls["collect"] == 1


def test_get_performance_summary(mock_time=None):
    monitor = ApplicationMonitor()
    now = time.time()
    monitor.performance_history = deque(
        [
            {"timestamp": now - 10, "cpu_percent": 10, "memory_percent": 20, "disk_percent": 30},
            {"timestamp": now - 5, "cpu_percent": 30, "memory_percent": 40, "disk_percent": 50},
        ]
    )

    summary = monitor.get_performance_summary(hours=1)
    assert summary["data_points"] == 2
    assert summary["cpu_percent"]["avg"] == 20


def test_get_performance_summary_no_data():
    monitor = ApplicationMonitor()
    monitor.performance_history.clear()

    summary = monitor.get_performance_summary(hours=1)
    assert summary["error"] == "No performance data available"


def test_set_and_get_alert_threshold():
    monitor = ApplicationMonitor()
    monitor.set_alert_threshold("cpu_percent", 70.0)
    thresholds = monitor.get_alert_thresholds()
    assert thresholds["cpu_percent"] == 70.0


def test_health_check_reports_metrics_count():
    monitor = ApplicationMonitor()
    monitor.record_metric("custom.metric", 123)

    health = monitor.health_check()
    assert health["metrics_count"] >= 1

