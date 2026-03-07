import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import importlib
import json
import sys
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

import src.data.monitoring.performance_monitor as perf_module
from src.data.monitoring.performance_monitor import PerformanceMonitor, PerformanceAlert


@pytest.fixture
def monitor():
    return PerformanceMonitor(max_history=50)


def test_record_metrics_and_statistics(monitor):
    monitor.record_metric("cache_hit_rate", 0.9, "%")
    monitor.record_metric("cache_hit_rate", 0.85, "%")
    history = monitor.get_metric_history("cache_hit_rate", hours=1)
    assert len(history) == 2

    current = monitor.get_current_metric("cache_hit_rate")
    assert current.value == 0.85

    stats = monitor.get_metric_statistics("cache_hit_rate", hours=1)
    assert stats["count"] == 2
    summary = monitor.get_all_metrics_summary()
    assert "cache_hit_rate" in summary


def test_alert_thresholds_and_customization(monitor):
    monitor.set_alert_threshold("cache_hit_rate", "warning", 0.95)
    monitor.record_cache_hit_rate(0.8)
    assert any(alert.metric_name == "cache_hit_rate" for alert in monitor.alerts)

    monitor.set_alert_threshold("data_load_time", "warning", 1)
    monitor.record_data_load_time(2)
    assert sum(alert.metric_name == "data_load_time" for alert in monitor.alerts) >= 1


def test_monitor_system_resources(monkeypatch, monitor):
    class DummyPsutil:
        @staticmethod
        def virtual_memory():
            return SimpleNamespace(percent=70)

        @staticmethod
        def cpu_percent(interval=1):
            return 55

        @staticmethod
        def disk_usage(path):
            return SimpleNamespace(percent=40)

    monkeypatch.setattr("src.data.monitoring.performance_monitor.psutil", DummyPsutil)
    monitor._monitor_system_resources()
    assert monitor.get_current_metric("memory_usage").value == 70
    assert monitor.get_current_metric("cpu_usage").value == 55
    assert monitor.get_current_metric("disk_usage").value == 40


def test_cleanup_old_alerts(monitor):
    old_alert = PerformanceAlert(
        level="warning",
        message="old",
        metric_name="cache_hit_rate",
        threshold=0.9,
        current_value=0.5,
        timestamp=datetime.now() - timedelta(hours=30),
    )
    monitor.alerts.append(old_alert)
    monitor.alerts.append(
        PerformanceAlert(
            level="warning",
            message="recent",
            metric_name="cache_hit_rate",
            threshold=0.9,
            current_value=0.5,
        )
    )
    monitor._cleanup_old_alerts()
    assert len(monitor.alerts) == 1


def test_export_metrics_and_report(monkeypatch, monitor):
    monitor.record_memory_usage(70)

    json_data = monitor.export_metrics("json")
    parsed = json.loads(json_data)
    assert "metrics" in parsed

    csv_data = monitor.export_metrics("csv")
    assert "memory_usage" in csv_data

    with pytest.raises(ValueError):
        monitor.export_metrics("xml")

    report = monitor.get_performance_report()
    assert report["monitoring_status"] == "inactive"
    assert "metrics_summary" in report


def test_record_metrics_and_statistics():
    monitor = PerformanceMonitor(max_history=5)
    monitor.record_cache_hit_rate(0.95)
    monitor.record_cache_hit_rate(0.90)
    monitor.record_data_load_time(3.2)
    monitor.record_memory_usage(0.75)
    monitor.record_throughput(1200)

    history = monitor.get_metric_history("cache_hit_rate", hours=1)
    assert len(history) == 2
    current = monitor.get_current_metric("cache_hit_rate")
    assert current.value == pytest.approx(0.90)

    stats = monitor.get_metric_statistics("cache_hit_rate", hours=1)
    assert stats["count"] == 2
    assert stats["min"] == pytest.approx(0.90)
    assert stats["max"] == pytest.approx(0.95)
    assert stats["avg"] == pytest.approx((0.95 + 0.90) / 2)

    assert monitor.get_metric_statistics("unknown") == {}
    summary = monitor.get_all_metrics_summary()
    assert "cache_hit_rate" in summary


def test_alert_generation_and_threshold_configuration():
    monitor = PerformanceMonitor()
    monitor.alerts.clear()

    monitor.record_cache_hit_rate(0.5)
    assert any(alert.metric_name == "cache_hit_rate" for alert in monitor.alerts)

    monitor.set_alert_threshold("custom_metric", "warning", 10)
    monitor.record_metric("custom_metric", 12)
    assert any(alert.metric_name == "custom_metric" for alert in monitor.alerts)

    recent_alerts = monitor.get_recent_alerts(hours=1)
    assert recent_alerts


def test_export_metrics_and_report(monkeypatch):
    monitor = PerformanceMonitor()
    monitor.record_metric("cpu_usage", 55.5, unit="%", metadata={"core": "all"})

    json_data = monitor.export_metrics("json")
    parsed = json.loads(json_data)
    assert "metrics" in parsed and "cpu_usage" in parsed["metrics"]

    csv_data = monitor.export_metrics("csv")
    assert "metric_name" in csv_data and "cpu_usage" in csv_data

    with pytest.raises(ValueError):
        monitor.export_metrics("xml")

    monitor.alerts.append(
        PerformanceAlert(
            level="warning",
            message="test",
            metric_name="cpu_usage",
            threshold=50,
            current_value=55.5,
        )
    )
    report = monitor.get_performance_report()
    assert report["metrics_summary"]
    assert report["alert_count"] == 1
    assert report["monitoring_status"] == "inactive"


def test_system_resource_monitoring_and_cleanup(monkeypatch):
    class DummyMem:
        percent = 42.0

    class DummyDisk:
        percent = 70.0

    class DummyPsutil:
        @staticmethod
        def virtual_memory():
            return DummyMem()

        @staticmethod
        def cpu_percent(interval=None):
            return 55.0

        @staticmethod
        def disk_usage(_):
            return DummyDisk()

    monkeypatch.setattr(perf_module, "psutil", DummyPsutil, raising=False)

    monitor = PerformanceMonitor()
    old_alert = PerformanceAlert(
        level="warning",
        message="old",
        metric_name="memory_usage",
        threshold=80.0,
        current_value=85.0,
        timestamp=datetime.now() - timedelta(hours=30),
    )
    monitor.alerts.append(old_alert)

    monitor._monitor_system_resources()
    assert monitor.get_current_metric("memory_usage").value == pytest.approx(42.0)
    assert monitor.get_current_metric("cpu_usage").value == pytest.approx(55.0)
    assert monitor.get_current_metric("disk_usage").value == pytest.approx(70.0)

    monitor._cleanup_old_alerts()
    assert not monitor.alerts


def test_monitor_control_flow(monkeypatch):
    loop_calls = []

    def fast_loop(self):
        loop_calls.append("ran")
        self.is_monitoring = False

    monkeypatch.setattr(PerformanceMonitor, "_monitor_loop", fast_loop)
    monitor = PerformanceMonitor()

    monitor.start_monitoring()
    monitor.stop_monitoring()

    assert loop_calls


def test_record_error_rate_helper():
    monitor = PerformanceMonitor()
    monitor.record_error_rate(0.2)
    current = monitor.get_current_metric("error_rate")
    assert current.value == pytest.approx(0.2)


def test_monitor_system_resources_psutil_missing(monkeypatch):
    monitor = PerformanceMonitor()
    monkeypatch.setattr(perf_module, "psutil", None, raising=False)
    monitor._monitor_system_resources()
    assert monitor.get_current_metric("memory_usage") is None


def test_monitor_system_resources_handles_errors(monkeypatch):
    class BrokenPsutil:
        @staticmethod
        def virtual_memory():
            class Dummy:
                percent = 0

            return Dummy()

        @staticmethod
        def cpu_percent(interval=1):
            raise RuntimeError("cpu fail")

        @staticmethod
        def disk_usage(path):
            return SimpleNamespace(percent=50)

    monitor = PerformanceMonitor()
    monkeypatch.setattr(perf_module, "psutil", BrokenPsutil, raising=False)
    monitor._monitor_system_resources()
    assert monitor.get_current_metric("disk_usage") is None


def test_monitor_loop_handles_exception(monkeypatch):
    monitor = PerformanceMonitor()
    monitor.is_monitoring = True

    def failing_resources():
        monitor.is_monitoring = False
        raise RuntimeError("boom")

    monitor._monitor_system_resources = failing_resources  # type: ignore
    monitor._cleanup_old_alerts = lambda: None  # type: ignore
    monkeypatch.setattr(perf_module.time, "sleep", lambda *_: None, raising=False)
    monitor._monitor_loop()
    assert monitor.is_monitoring is False


def test_monitor_loop_normal_flow(monkeypatch):
    monitor = PerformanceMonitor()
    monitor.is_monitoring = True
    calls = {"clean": 0, "sleep": 0}

    def safe_resources():
        monitor.is_monitoring = False

    monitor._monitor_system_resources = safe_resources  # type: ignore

    def clean():
        calls["clean"] += 1

    monitor._cleanup_old_alerts = clean  # type: ignore

    def fast_sleep(*_):
        calls["sleep"] += 1

    monkeypatch.setattr(perf_module.time, "sleep", fast_sleep, raising=False)
    monitor._monitor_loop()
    assert calls["clean"] == 1
    assert calls["sleep"] == 1


def test_performance_monitor_fallback_logger(monkeypatch):
    original_import = __import__

    def fake_import(name, *args, **kwargs):
        if name == "src.infrastructure.logging":
            raise ImportError("forced")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    fallback_logger = perf_module._resolve_logger()
    logger = fallback_logger("fallback-logger")
    assert logger.name == "fallback-logger"



