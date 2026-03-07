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


import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

if "src.data.enhanced_integration_manager" not in sys.modules:
    enhanced_module = ModuleType("src.data.enhanced_integration_manager")

    class _EnhancedDataIntegrationManager:
        ...

    enhanced_module.EnhancedDataIntegrationManager = _EnhancedDataIntegrationManager
    sys.modules["src.data.enhanced_integration_manager"] = enhanced_module

from src.data.monitoring.dashboard import (
    AlertRule,
    DashboardConfig,
    DataDashboard,
    MetricWidget,
)


class DummyEnhancedManager:
    def __init__(self):
        self.performance_metrics = {
            "performance": {
                "distributed_load_time": {"avg": 2.5},
            },
            "cache": {"hits": 80, "misses": 20},
            "nodes": {
                "node_a": {"status": "active"},
                "node_b": {"status": "inactive"},
            },
            "streams": {
                "stream_a": {"is_running": True},
                "stream_b": {"is_running": False},
            },
        }
        self.quality_report = {"score": 0.93}
        self.alert_history = [{"level": "info", "message": "ok"}]

    def get_performance_metrics(self):
        return self.performance_metrics

    def get_quality_report(self, days=1):
        return self.quality_report

    def get_alert_history(self, hours=24):
        return self.alert_history


@pytest.fixture
def dashboard():
    manager = DummyEnhancedManager()
    config = DashboardConfig(title="Test Dashboard", refresh_interval=10)
    return DataDashboard(enhanced_manager=manager, config=config)


def test_collect_metrics_computes_availability_and_cache_rate(dashboard):
    metrics = dashboard._collect_metrics()

    performance = metrics["performance"]
    assert performance["cache_hit_rate"] == pytest.approx(0.8)
    assert performance["node_availability"] == pytest.approx(0.5)
    assert performance["stream_availability"] == pytest.approx(0.5)
    assert performance["overall_score"] == pytest.approx(0.5)
    assert metrics["quality"]["score"] == 0.93
    assert metrics["alerts"][0]["level"] == "info"


def test_collect_metrics_handles_exception(monkeypatch, dashboard):
    monkeypatch.setattr(
        dashboard.enhanced_manager,
        "get_performance_metrics",
        lambda: (_ for _ in ()).throw(RuntimeError("fail")),
    )

    metrics = dashboard._collect_metrics()
    assert "error" in metrics
    assert metrics["error"] == "fail"


def test_get_dashboard_data_contains_widgets_and_alerts(dashboard):
    data = dashboard.get_dashboard_data()

    assert data["config"]["title"] == "Test Dashboard"
    assert data["status"]["widget_count"] >= 6  # default widgets
    assert data["status"]["alert_rule_count"] >= 3
    assert "performance_overview" in data["widgets"]
    assert "performance" in data["current_metrics"]


def test_export_dashboard_report_writes_json(tmp_path, dashboard):
    output_file = tmp_path / "dashboard_report.json"
    path = dashboard.export_dashboard_report(file_path=str(output_file))

    assert Path(path).exists()
    content = json.loads(output_file.read_text(encoding="utf-8"))
    assert content["config"]["title"] == "Test Dashboard"


def test_callbacks_are_triggered_even_when_one_fails(dashboard):
    received = []

    def failing_callback(data):
        raise ValueError("boom")

    def success_callback(data):
        received.append(data)

    dashboard.add_callback("metrics_update", failing_callback)
    dashboard.add_callback("metrics_update", success_callback)

    dashboard._trigger_callback("metrics_update", {"value": 1})
    assert received == [{"value": 1}]

