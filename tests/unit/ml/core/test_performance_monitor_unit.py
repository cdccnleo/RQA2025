import pytest
from unittest.mock import Mock, patch
from datetime import datetime

import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

import src.ml.core.performance_monitor as performance_monitor_module
from src.ml.core.performance_monitor import (
    MLPerformanceMetrics,
    MLPerformanceMonitor,
    record_inference_performance,
    record_model_performance,
)


@pytest.fixture(autouse=True)
def patch_models_adapter():
    with patch("src.ml.core.performance_monitor.get_models_adapter") as mock_adapter:
        mock_adapter.return_value = Mock(get_models_logger=lambda: Mock())
        yield mock_adapter


@pytest.fixture
def metrics(monkeypatch):
    # 避免真实 GPU 检测和 psutil 延迟
    monkeypatch.setattr("src.ml.core.performance_monitor.psutil.cpu_percent", lambda interval=0: 50.0)

    class MemInfo:
        percent = 60.0

    monkeypatch.setattr("src.ml.core.performance_monitor.psutil.virtual_memory", lambda: MemInfo())

    m = MLPerformanceMetrics(window_size=5)
    return m


def test_metrics_record_and_stats(metrics):
    metrics.record_inference_latency(120.0)
    metrics.record_inference_latency(80.0)
    metrics.record_inference_throughput(15.0)
    metrics.record_inference_error("timeout", model_id="model-1")
    metrics.record_model_metrics(0.9, 0.88, 0.91, 0.89, model_id="model-1")
    metrics.record_process_latency("training", 210.0)
    metrics.record_step_latency("load_data", 45.0)
    metrics.record_resource_usage()

    stats = metrics.get_comprehensive_stats()

    assert stats["inference"]["avg_latency_ms"] == pytest.approx(100.0)
    assert stats["inference"]["error_rate"] == pytest.approx(0.5)
    assert stats["model"]["avg_accuracy"] == pytest.approx(0.9)
    assert stats["resources"]["cpu_avg_percent"] == pytest.approx(50.0)
    assert stats["processes"]["training"]["avg_latency_ms"] == pytest.approx(210.0)


def test_monitor_alert_trigger(monkeypatch):
    monitor = MLPerformanceMonitor(collection_interval=1)
    alerts = []
    monitor.add_alert_callback(lambda alert: alerts.append(alert))

    mock_stats = {
        "timestamp": datetime.now().isoformat(),
        "inference": {
            "p95_latency_ms": 1500,
            "error_rate": 0.2,
        },
        "model": {},
        "resources": {
            "cpu_max_percent": 95,
            "memory_max_percent": 92,
        },
        "processes": {},
        "last_update": datetime.now().isoformat(),
        "window_size": 100,
    }

    monkeypatch.setattr(
        monitor.metrics,
        "get_comprehensive_stats",
        lambda: mock_stats,
    )

    monitor._check_alerts()

    alert_types = {alert["type"] for alert in alerts}
    assert {"inference_latency", "inference_error_rate", "cpu_usage", "memory_usage"} <= alert_types


def test_monitor_record_helpers(metrics, monkeypatch):
    monitor = MLPerformanceMonitor(collection_interval=1)
    monitor.metrics = metrics

    monitor.record_inference_performance(120.0, model_id="model-A")
    monitor.record_model_performance(0.92, 0.9, 0.91, 0.905, model_id="model-A")
    monitor.record_process_performance("proc-1", 330.0)
    monitor.record_step_performance("step-1", 55.0)

    stats = monitor.get_current_stats()

    assert stats["inference"]["avg_latency_ms"] == pytest.approx(120.0)
    assert stats["model"]["avg_accuracy"] == pytest.approx(0.92)
    assert stats["processes"]["proc-1"]["total_executions"] == 1


def test_metrics_empty_stats_return_defaults():
    metrics = MLPerformanceMetrics(window_size=2)
    assert metrics.get_inference_stats() == {}
    assert metrics.get_model_stats() == {}


def test_record_resource_usage_handles_failures(monkeypatch):
    def broken_cpu_percent(interval=0):
        raise RuntimeError("cpu failure")

    def broken_virtual_memory():
        raise RuntimeError("mem failure")

    monkeypatch.setattr("psutil.cpu_percent", broken_cpu_percent)
    monkeypatch.setattr("psutil.virtual_memory", broken_virtual_memory)
    metrics = MLPerformanceMetrics(window_size=2)
    metrics.record_resource_usage()

    assert metrics.cpu_usages[-1] == 0.0
    assert metrics.memory_usages[-1] == 0.0


def test_monitor_start_and_stop_flags():
    monitor = MLPerformanceMonitor(collection_interval=1)
    monitor.start_monitoring()
    assert monitor.monitoring is True
    monitor.stop_monitoring()
    assert monitor.monitoring is False


def test_trigger_alert_logs_exceptions(monkeypatch):
    monitor = MLPerformanceMonitor(collection_interval=1)

    def bad_callback(alert):
        raise RuntimeError("callback failure")

    monitor.add_alert_callback(bad_callback)
    mock_logger = Mock()
    monkeypatch.setattr(performance_monitor_module, "logger", mock_logger)

    monitor._trigger_alert({"type": "test"})
    mock_logger.exception.assert_called_once()


def test_check_alerts_returns_when_no_inference():
    monitor = MLPerformanceMonitor(collection_interval=1)
    monitor.metrics.get_comprehensive_stats = lambda: {
        "timestamp": datetime.now().isoformat(),
        "inference": {},
        "resources": {},
        "processes": {},
        "last_update": datetime.now().isoformat(),
    }
    monitor._check_alerts()  # should not raise or trigger alerts


def test_record_inference_performance_with_error():
    monitor = MLPerformanceMonitor(collection_interval=1)
    monitor.record_inference_performance(0.0, model_id="m1", error="timeout")
    assert monitor.metrics.inference_errors


def test_global_record_helpers_use_singleton(monkeypatch):
    global_monitor = performance_monitor_module._GLOBAL_MONITOR
    global_monitor.metrics.inference_latencies.clear()
    global_monitor.metrics.model_accuracies.clear()

    record_inference_performance(150.0, model_id="global")
    record_model_performance(0.8, 0.7, 0.75, 0.72, model_id="global")

    stats = global_monitor.get_current_stats()
    assert stats["inference"]["total_requests"] >= 1
    assert stats["model"]["avg_accuracy"] == pytest.approx(0.8)


def test_alert_penalties_trigger_for_thresholds(monkeypatch):
    monitor = MLPerformanceMonitor(collection_interval=1)
    monitor.metrics.record_inference_latency(2500.0)
    monitor.metrics.record_inference_error("timeout")
    monitor.metrics.record_resource_usage(cpu_percent=95.0, memory_percent=96.0)

    monitor.add_alert_callback(lambda alert: None)
    monitor_alerts = []
    monitor.add_alert_callback(lambda alert: monitor_alerts.append(alert))

    stats = monitor.get_current_stats()
    assert any(alert["type"] in {"inference_latency", "inference_error_rate", "cpu_usage", "memory_usage"} for alert in monitor_alerts)

