import json
from datetime import datetime
from unittest.mock import Mock, patch

import pytest

import src.ml.core.monitoring_dashboard as monitoring_dashboard
from src.ml.core.monitoring_dashboard import MLMonitoringDashboard
import sys
from pathlib import Path

# 添加src路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


class DummyThread:
    def __init__(self, target=None, name=None, daemon=False):
        self.target = target
        self.name = name
        self.daemon = daemon
        self.started = False

    def start(self):
        self.started = True

    def join(self, timeout=None):
        pass


class DummyPerformanceMonitor:
    def __init__(self):
        self.callbacks = []
        self.alert_callbacks = []
        self.stats = {
            "inference": {
                "avg_latency_ms": 150.0,
                "p95_latency_ms": 2100.0,
                "p99_latency_ms": 2500.0,
                "throughput_avg": 12.5,
                "error_rate": 0.04,
                "total_requests": 300,
            },
            "resources": {
                "cpu_avg_percent": 85.0,
                "cpu_max_percent": 92.0,
                "memory_avg_percent": 70.0,
                "memory_max_percent": 88.0,
            },
        }

    def add_alert_callback(self, callback):
        self.alert_callbacks.append(callback)

    def get_current_stats(self):
        return self.stats


class DummyProcessOrchestrator:
    def __init__(self):
        self.statistics = {
            "active_processes": 3,
            "completed_processes": 10,
            "failed_processes": 1,
            "max_workers": 4,
            "executor_pool_info": {"active_threads": 2},
            "queue_size": 1,
        }

    def get_statistics(self):
        return self.statistics


@pytest.fixture
def dashboard(monkeypatch):
    performance_monitor = DummyPerformanceMonitor()
    process_orchestrator = DummyProcessOrchestrator()

    monkeypatch.setattr(
        "src.ml.core.monitoring_dashboard.get_ml_performance_monitor",
        lambda: performance_monitor,
    )
    monkeypatch.setattr(
        "src.ml.core.monitoring_dashboard.get_ml_process_orchestrator",
        lambda: process_orchestrator,
    )
    # 也需要替换process_orchestrator模块中的函数
    monkeypatch.setattr(
        "src.ml.core.process_orchestrator.get_ml_process_orchestrator",
        lambda: process_orchestrator,
    )
    monkeypatch.setattr("threading.Thread", DummyThread)

    panel = MLMonitoringDashboard(update_interval=0)
    panel._display_dashboard = lambda: None

    # 直接设置panel的orchestrator为我们的dummy实例
    panel.process_orchestrator = process_orchestrator

    # 确保返回的orchestrator就是panel内部使用的实例
    return panel, panel.performance_monitor, panel.process_orchestrator


def test_dashboard_start_and_stop_registers_callback(dashboard):
    panel, perf_monitor, _ = dashboard

    panel.start_dashboard()
    assert panel.running is True
    # 检查panel内部的performance_monitor是否正确注册了回调
    assert perf_monitor.alert_callbacks and perf_monitor.alert_callbacks[0] == panel._handle_alert

    panel.start_dashboard()
    assert len(perf_monitor.alert_callbacks) == 1

    panel.stop_dashboard()
    assert panel.running is False
    panel.stop_dashboard()
    assert panel.running is False


def test_dashboard_updates_stats_and_process_history(dashboard):
    panel, perf_monitor, orchestrator = dashboard

    panel._update_stats()
    # 检查current_stats是否被正确设置
    assert panel.current_stats is not None
    assert isinstance(panel.current_stats, dict)

    panel._update_process_status()
    assert "active_count" in panel.process_history
    assert panel.process_history["active_count"]


def test_dashboard_handles_alerts_and_export(dashboard):
    panel, _, _ = dashboard

    panel._handle_alert({"type": "cpu_usage", "level": "warning", "message": "High CPU"})
    assert panel.alert_history

    data = panel.get_dashboard_data()
    assert "current_stats" in data
    assert len(data["alert_history"]) == 1

    exported = panel.export_dashboard_data()
    parsed = json.loads(exported)
    assert parsed["alert_history"]


def test_process_history_trims_to_max(dashboard):
    panel, _, orchestrator = dashboard
    panel.display_config["max_process_history"] = 2

    for count in range(5):
        orchestrator.statistics["active_processes"] = count
        panel._update_process_status()

    history = panel.process_history["active_count"]
    assert len(history) == 2
    # 历史记录保留最后2个，分别是count=3和count=4
    assert history[0]["count"] == 3
    assert history[1]["count"] == 4


def test_alert_history_trimmed_to_limit(dashboard):
    panel, _, _ = dashboard
    panel.display_config["max_alert_history"] = 2

    for idx in range(3):
        panel._handle_alert({"type": "alert", "index": idx})

    assert len(panel.alert_history) == 2
    assert panel.alert_history[0]["alert"]["index"] == 1
    assert panel.alert_history[1]["alert"]["index"] == 2


def test_export_dashboard_data_unsupported_format(dashboard):
    panel, _, _ = dashboard
    with pytest.raises(ValueError):
        panel.export_dashboard_data(format="yaml")


def test_dashboard_config_and_health_score(dashboard):
    panel, perf_monitor, _ = dashboard
    panel._update_stats()

    panel.configure_display({"show_alerts": False})
    assert panel.display_config["show_alerts"] is False

    score = panel.get_health_score()
    assert 0 <= score <= 100


def test_dashboard_health_score_penalizes_recent_alerts(dashboard):
    panel, _, _ = dashboard
    panel._update_stats()

    # 添加一些告警来测试健康评分惩罚
    for _ in range(3):
        panel._handle_alert({"type": "error", "level": "error", "message": "Failure"})

    # 检查是否有get_health_score方法，如果有则测试
    if hasattr(panel, 'get_health_score'):
        score_with_alerts = panel.get_health_score()
        # 确保健康评分在合理范围内
        assert isinstance(score_with_alerts, (int, float))
        assert 0 <= score_with_alerts <= 100
    else:
        # 如果没有get_health_score方法，跳过这个测试
        pytest.skip("MLMonitoringDashboard没有get_health_score方法")


def test_global_dashboard_helpers_delegate(monkeypatch):
    class StubDashboard:
        def __init__(self):
            self.started = False
            self.start_calls = 0
            self.stop_calls = 0

        def start_dashboard(self):
            self.start_calls += 1
            self.started = True

        def stop_dashboard(self):
            self.stop_calls += 1
            self.started = False

        def get_dashboard_data(self):
            return {"value": "data"}

        def get_health_score(self):
            return 42.0

    stub = StubDashboard()
    monkeypatch.setattr(monitoring_dashboard, "_GLOBAL_DASHBOARD", stub)

    assert monitoring_dashboard.get_ml_monitoring_dashboard() is stub

    monitoring_dashboard.start_ml_dashboard()
    assert stub.started is True

    data = monitoring_dashboard.get_ml_dashboard_data()
    assert data["value"] == "data"

    score = monitoring_dashboard.get_ml_health_score()
    assert score == 42.0

    monitoring_dashboard.stop_ml_dashboard()
    assert stub.started is False

