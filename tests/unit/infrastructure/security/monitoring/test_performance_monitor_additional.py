from __future__ import annotations

import pytest

import src.infrastructure.security.monitoring.performance_monitor as perf_module
from src.infrastructure.security.monitoring.performance_monitor import PerformanceMonitor, monitor_performance


@pytest.fixture
def isolated_monitor(monkeypatch):
    monitor = PerformanceMonitor(enabled=True, collection_interval=0)
    monkeypatch.setattr(perf_module, "_global_monitor", monitor, raising=False)
    return monitor


def test_monitor_performance_decorator_tracks_errors():
    monitor = PerformanceMonitor(enabled=True, collection_interval=0)

    @monitor_performance("decorated_call", monitor)
    def _failing():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        _failing()

    metrics = monitor.get_metrics("decorated_call")
    assert metrics["total_calls"] == 1
    assert metrics["error_count"] == 1


def test_record_security_operation_updates_activity(isolated_monitor):
    perf_module.record_security_operation(
        operation="user_login",
        duration=0.2,
        user_id="alice",
        resource="/auth/login",
        is_error=True,
    )

    metrics = isolated_monitor.get_metrics("user_login")
    assert metrics["total_calls"] == 1
    assert metrics["error_count"] == 1

    assert "alice" in isolated_monitor.user_activity
    assert "/auth/login" in isolated_monitor.resource_access


def test_get_security_performance_report_combines_sections(monkeypatch, isolated_monitor):
    base_report = {"base": "report"}
    monkeypatch.setattr(isolated_monitor, "get_performance_report", lambda: base_report.copy())

    setattr(isolated_monitor, "_get_user_activity_summary", lambda: {"users": 1})
    setattr(isolated_monitor, "_get_resource_access_summary", lambda: {"resources": 1})
    setattr(isolated_monitor, "_get_security_operation_trends", lambda: {"trends": []})

    report = perf_module.get_security_performance_report()

    assert report["base"] == "report"
    security_metrics = report["security_metrics"]
    assert security_metrics["user_activity_summary"] == {"users": 1}
    assert security_metrics["resource_access_summary"] == {"resources": 1}
    assert security_metrics["security_operation_trends"] == {"trends": []}


def test_record_performance_uses_global_monitor(isolated_monitor):
    perf_module.record_performance("global_op", 0.05, is_error=False)
    metrics = isolated_monitor.get_metrics("global_op")
    assert metrics["total_calls"] == 1
    assert metrics["error_count"] == 0

    assert perf_module.get_performance_monitor() is isolated_monitor
