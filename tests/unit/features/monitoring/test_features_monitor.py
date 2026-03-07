import builtins
import json
import time
import types
import sys
from unittest.mock import MagicMock, call

import pandas as pd
import pytest

from src.features.monitoring.features_monitor import (
    FeaturesMonitor,
    MetricType,
    MetricValue,
    get_monitor,
    monitor_operation,
)


@pytest.fixture
def monitor(monkeypatch):
    monitor = FeaturesMonitor(config={"monitor_interval": 0.01})
    monitor.metrics_collector = MagicMock()
    monitor.alert_manager = MagicMock()
    monitor.performance_analyzer = MagicMock()
    monitor.performance_analyzer.analyze.return_value = {"summary": "ok"}
    monitor.performance_analyzer.analyze_performance.return_value = {"trend": "stable"}
    return monitor


def test_register_and_unregister_component(monitor):
    monitor.register_component("processor", "technical")
    assert "processor" in monitor.components
    monitor.unregister_component("processor")
    assert "processor" not in monitor.components


def test_register_component_duplicate_warns(monkeypatch, caplog):
    caplog.set_level("WARNING")
    monitor = FeaturesMonitor()
    monitor.register_component("processor", "technical")
    monitor.register_component("processor", "technical")
    assert "已存在" in caplog.text


def test_update_component_status_and_metrics(monitor):
    monitor.register_component("processor", "technical")
    monitor.update_component_status("processor", "running", {"response_time": 0.5})
    status = monitor.get_component_status("processor")
    assert status["status"] == "running"
    metrics = monitor.get_component_metrics("processor")
    assert "response_time" in metrics


def test_update_component_status_requires_registration(caplog):
    caplog.set_level("WARNING")
    monitor = FeaturesMonitor()
    monitor.update_component_status("processor", "running")
    assert "未注册" in caplog.text


def test_collect_metrics_triggers_alert(monitor):
    monitor.thresholds["processor.cpu_usage"] = 70
    monitor.register_component("processor", "technical")
    monitor.collect_metrics(
        "processor", "cpu_usage", 80.0, MetricType.GAUGE, labels={"role": "worker"}
    )
    monitor.alert_manager.send_alert.assert_called_once()


def test_collect_metrics_requires_registration(caplog):
    caplog.set_level("WARNING")
    monitor = FeaturesMonitor()
    monitor.collect_metrics("processor", "latency", 0.5)
    assert "未注册" in caplog.text


def test_get_all_metrics_and_status(monitor):
    monitor.register_component("processor", "technical")
    monitor.collect_metrics("processor", "cpu", 10.0)
    all_metrics = monitor.get_all_metrics()
    all_status = monitor.get_all_status()
    assert "processor" in all_metrics
    assert "processor" in all_status


def test_start_and_stop_monitoring(monkeypatch, monitor):
    monitor.register_component("processor", "technical")
    monitor.collect_metrics = MagicMock()
    monitor._analyze_performance = MagicMock()
    monitor._check_component_health = MagicMock()

    monitor.start_monitoring()
    time.sleep(0.05)
    monitor.stop_monitoring()

    assert monitor.collect_metrics.called


def test_collect_system_metrics_handles_missing_psutil(monkeypatch, monitor):
    original_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "psutil":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    monitor.collect_metrics = MagicMock()
    monitor._collect_system_metrics()
    monitor.collect_metrics.assert_not_called()


def test_get_component_metrics_returns_empty_when_unregistered(monitor):
    assert monitor.get_component_metrics("unknown") == {}


def test_metrics_history_keeps_latest(monitor):
    monitor.register_component("processor", "technical")
    for i in range(5):
        monitor.collect_metrics("processor", "latency", i * 0.1)
    history = monitor.metrics_history["processor.latency"]
    assert len(history) == 5
    assert isinstance(history[-1], MetricValue)


def test_threshold_check_skips_when_not_configured(monitor):
    monitor.register_component("processor", "technical")
    monitor.alert_manager.reset_mock()
    monitor.collect_metrics("processor", "latency", 0.6)
    monitor.alert_manager.send_alert.assert_not_called()


def test_performance_analysis_failure_suppressed(monkeypatch):
    monitor = FeaturesMonitor(config={"monitor_interval": 0.01})
    monitor.performance_analyzer = MagicMock()
    monitor.performance_analyzer.analyze.side_effect = RuntimeError("boom")
    monitor.metrics_collector = MagicMock()
    monitor._analyze_performance()
    monitor.metrics_collector.record_performance_report.assert_not_called()


def test_monitor_loop_recovers_from_collect_error(monkeypatch):
    monitor = FeaturesMonitor(config={"monitor_interval": 0})
    monitor.is_monitoring = True

    def fail_collect():
        raise RuntimeError("boom")

    monkeypatch.setattr(monitor, "_collect_system_metrics", fail_collect)
    monitor._analyze_performance = MagicMock()
    monitor._check_component_health = MagicMock()

    def stop_loop(_interval):
        monitor.is_monitoring = False

    monkeypatch.setattr("src.features.monitoring.features_monitor.time.sleep", stop_loop)
    monitor._monitor_loop()
    monitor._analyze_performance.assert_not_called()
    monitor._check_component_health.assert_not_called()
    assert monitor.is_monitoring is False


def test_collect_system_metrics_runtime_error_fallback(monkeypatch):
    monitor = FeaturesMonitor()
    monitor.collect_metrics = MagicMock()

    def raise_error(*args, **kwargs):
        raise RuntimeError("boom")

    dummy_psutil = types.ModuleType("psutil")
    dummy_psutil.cpu_percent = raise_error
    dummy_psutil.virtual_memory = raise_error
    dummy_psutil.disk_usage = raise_error
    monkeypatch.setitem(sys.modules, "psutil", dummy_psutil)

    monitor._collect_system_metrics()

    monitor.collect_metrics.assert_has_calls(
        [
            call("system", "cpu_usage", 0.0),
            call("system", "memory_usage", 0.0),
            call("system", "disk_usage", 0.0),
        ],
        any_order=False,
    )


def test_check_component_health_timeout_alert(monkeypatch):
    monitor = FeaturesMonitor(config={"component_timeout": 0.01})
    monitor.alert_manager = MagicMock()
    monitor.register_component("processor", "technical")
    monitor.components["processor"].last_update = time.time() - 1
    monitor._check_component_health()
    monitor.alert_manager.send_alert.assert_called_once()


def test_check_component_health_skips_when_no_last_update(monitor):
    monitor.register_component("processor", "technical")
    monitor.alert_manager.reset_mock()
    monitor._check_component_health()
    monitor.alert_manager.send_alert.assert_not_called()


def test_stop_monitoring_when_not_started(monkeypatch):
    monitor = FeaturesMonitor()
    monitor.stop_monitoring()  # should not raise even if not started
    assert monitor.is_monitoring is False


def test_start_monitoring_when_already_running(monkeypatch):
    monitor = FeaturesMonitor()
    monitor.is_monitoring = True
    monitor.start_monitoring()
    assert monitor.monitor_thread is None


def test_stop_monitoring_waits_for_thread(monkeypatch):
    monitor = FeaturesMonitor()
    fake_thread = MagicMock()
    monitor.is_monitoring = True
    monitor.monitor_thread = fake_thread
    monitor.stop_monitoring()
    fake_thread.join.assert_called_once()


def test_export_metrics_creates_file(tmp_path, monitor):
    monitor.register_component("processor", "technical")
    monitor.collect_metrics("processor", "latency", 0.5)
    monitor.alert_manager.get_recent_alerts.return_value = [{"level": "warning"}]

    file_path = tmp_path / "metrics.json"
    monitor.export_metrics(str(file_path))
    data = json.loads(file_path.read_text(encoding="utf-8"))

    assert data["components"]["processor"]["status"] == "registered"
    assert data["alerts"][0]["level"] == "warning"


def test_export_metrics_failure_logs_error(monkeypatch, caplog):
    caplog.set_level("ERROR")
    monitor = FeaturesMonitor()
    monitor.alert_manager = MagicMock()

    def fake_open(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", fake_open)
    monitor.export_metrics("invalid/path.json")
    assert "导出指标数据失败" in caplog.text


def test_get_performance_report_success(monitor):
    monitor.register_component("processor", "technical")
    monitor.alert_manager.get_recent_alerts.return_value = []
    report = monitor.get_performance_report()
    assert report["summary"]["total_components"] == 1
    assert "performance_analysis" in report


def test_get_performance_report_failure_returns_empty(monkeypatch):
    monitor = FeaturesMonitor()
    monitor.alert_manager = MagicMock()
    monitor.performance_analyzer = MagicMock()
    monitor.performance_analyzer.analyze_performance.side_effect = RuntimeError("boom")
    report = monitor.get_performance_report()
    assert report == {}


def test_get_component_status_returns_none_for_unknown(monitor):
    assert monitor.get_component_status("unknown") is None


def test_analyze_performance_triggers_alert(monitor):
    monitor.register_component("processor", "technical")
    history = monitor.metrics_history["processor.response_time"]
    for _ in range(12):
        history.append(
            MetricValue(
                name="response_time",
                value=2.0,
                timestamp=time.time(),
                labels={},
                metric_type=MetricType.GAUGE,
            )
        )
    monitor.alert_manager.reset_mock()
    monitor._analyze_performance()
    monitor.alert_manager.send_alert.assert_called_once()


def test_monitor_context_manager_controls_monitoring(monkeypatch):
    start_calls = []
    stop_calls = []

    def fake_start(self):
        start_calls.append(self)

    def fake_stop(self):
        stop_calls.append(self)

    monkeypatch.setattr(FeaturesMonitor, "start_monitoring", fake_start)
    monkeypatch.setattr(FeaturesMonitor, "stop_monitoring", fake_stop)

    with FeaturesMonitor() as ctx_monitor:
        assert isinstance(ctx_monitor, FeaturesMonitor)

    assert start_calls == [ctx_monitor]
    assert stop_calls == [ctx_monitor]


def test_get_monitor_returns_singleton(monkeypatch):
    from src.features.monitoring import features_monitor as fm_module

    monkeypatch.setattr(fm_module, "_global_monitor", None)
    monitor_one = get_monitor({"monitor_interval": 0.1})
    monitor_two = get_monitor()
    assert monitor_one is monitor_two


def test_monitor_operation_success(monkeypatch):
    fake_monitor = MagicMock()
    fake_monitor.collect_metrics = MagicMock()
    fake_monitor.update_component_status = MagicMock()
    monkeypatch.setattr(
        "src.features.monitoring.features_monitor.get_monitor",
        lambda config=None: fake_monitor,
    )

    @monitor_operation("processor", "sync_features")
    def succeed(value):
        return value * 2

    assert succeed(2) == 4
    fake_monitor.update_component_status.assert_any_call("processor", "running")
    fake_monitor.update_component_status.assert_any_call("processor", "active")
    fake_monitor.collect_metrics.assert_any_call(
        "processor", "sync_features_success", 1, MetricType.COUNTER
    )


def test_monitor_operation_failure(monkeypatch):
    fake_monitor = MagicMock()
    fake_monitor.collect_metrics = MagicMock()
    fake_monitor.update_component_status = MagicMock()
    monkeypatch.setattr(
        "src.features.monitoring.features_monitor.get_monitor",
        lambda config=None: fake_monitor,
    )

    @monitor_operation("processor", "sync_features")
    def fail():
        raise ValueError("boom")

    with pytest.raises(ValueError):
        fail()

    fake_monitor.collect_metrics.assert_any_call(
        "processor", "sync_features_error", 1, MetricType.COUNTER
    )
    fake_monitor.update_component_status.assert_any_call("processor", "error")


def test_unregister_component_missing_warns(caplog):
    caplog.set_level("WARNING")
    monitor = FeaturesMonitor()
    monitor.unregister_component("ghost")
    assert "不存在" in caplog.text

