import time
from types import SimpleNamespace

import pytest

from src.infrastructure.cache.monitoring.performance_monitor import PerformanceMonitor


def test_performance_monitor_records_and_resets():
    monitor = PerformanceMonitor()
    monitor.record_metric("latency_ms", 12.5, tags={"cache": "fast"})

    metric = monitor.get_metric("latency_ms")
    assert metric["value"] == 12.5
    assert metric["tags"] == {"cache": "fast"}
    assert isinstance(metric["timestamp"], float)

    all_metrics = monitor.get_all_metrics()
    assert all_metrics is not monitor.metrics
    assert "latency_ms" in all_metrics

    monitor.reset_metrics()
    assert monitor.get_all_metrics() == {}


def test_performance_monitor_operation_duration(monkeypatch):
    fake_time = SimpleNamespace(value=0.0)

    def fake_now():
        return fake_time.value

    monkeypatch.setattr(time, "time", fake_now)

    monitor = PerformanceMonitor()
    monitor.start_operation("refresh")
    fake_time.value = 2.5
    duration = monitor.end_operation("refresh")

    assert duration == pytest.approx(2.5, rel=1e-6)
    assert monitor.end_operation("unknown") == 0.0


def test_performance_monitor_triggers_callback(monkeypatch):
    records = []

    def listener(name, value, tags):
        records.append((name, value, tags))

    monitor = PerformanceMonitor()
    monitor.register_listener(listener)

    monitor.record_metric("throughput", 999, tags={"cache": "hot"})
    assert records == [("throughput", 999, {"cache": "hot"})]

    monitor.start_operation("bulk_load")
    monitor.clear_listeners()
    assert monitor._listeners == []

