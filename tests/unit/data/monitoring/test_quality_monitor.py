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
from datetime import datetime, timedelta

import pandas as pd

from src.data.monitoring.quality_monitor import DataModel, DataQualityMonitor


def test_evaluate_quality_handles_missing_and_timeliness(tmp_path, monkeypatch):
    monitor = DataQualityMonitor(report_dir=str(tmp_path))
    df = pd.DataFrame(
        {
            "value": [1.0, None, 3.0],
            "volume": [10, 12, None],
        },
        index=pd.date_range("2024-01-01", periods=3, freq="D"),
    )
    model = DataModel(df)
    created_at = (datetime.now() - timedelta(days=2)).isoformat()
    model.metadata = {"source": "test_source", "created_at": created_at}

    metrics = monitor.evaluate_quality(model)

    assert 0.0 < metrics.completeness < 1.0
    assert metrics.timeliness < 1.0
    history_file = tmp_path / "quality_history.json"
    assert history_file.exists()
    history = json.loads(history_file.read_text(encoding="utf-8"))
    assert "test_source" in history
    assert len(history["test_source"]) == 1


def test_generate_quality_report_includes_sources(tmp_path, monkeypatch):
    monitor = DataQualityMonitor(report_dir=str(tmp_path))
    model = DataModel(pd.DataFrame({"value": [1]}))
    model.metadata = {"source": "alpha"}
    monitor.evaluate_quality(model)

    class _FakeNow:
        def __init__(self):
            self._value = datetime(2024, 1, 1, 12, 0, 0)

        def isoformat(self):
            return self._value.isoformat()

        def strftime(self, _format):
            return "20240101_120000"

    class _FakeDateTime:
        @staticmethod
        def now():
            return _FakeNow()

    monkeypatch.setattr("src.data.monitoring.quality_monitor.datetime", _FakeDateTime)

    report = monitor.generate_quality_report()
    assert "sources" in report and "alpha" in report["sources"]
    report_files = list(tmp_path.glob("quality_report_*.json"))
    assert report_files, "report file should be created"


def test_get_quality_summary_defaults_sources(tmp_path):
    monitor = DataQualityMonitor(report_dir=str(tmp_path))
    summary = monitor.get_quality_summary()
    assert summary["overall"]["total_sources"] == len(summary["sources"])


def test_alert_and_trend_helpers(tmp_path):
    monitor = DataQualityMonitor(report_dir=str(tmp_path))
    monitor.set_thresholds({"completeness": 0.8})
    monitor.set_alert_config({"enabled": False})

    alerts = monitor.get_alerts(days=3)
    assert alerts and alerts[0]["completeness"] == 0.8

    trend = monitor.get_quality_trend("src", "completeness")
    assert trend["data"]["values"]
    assert trend["statistics"]["trend"] == "up"

