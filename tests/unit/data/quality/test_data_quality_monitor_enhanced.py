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


import pandas as pd

from src.data.cache.smart_data_cache import DataCacheConfig
from src.data.quality.data_quality_monitor import (
    DataQualityMonitor,
    DataQualityRule,
    QualityMetric,
)


class DummyRule(DataQualityRule):
    def __init__(self, name: str = "dummy", value: float = 0.2, status: str = "poor"):
        super().__init__(name=name, weight=1.0)
        self._value = value
        self._status = status

    def check(self, data):
        return QualityMetric(
            name=self.name,
            value=self._value,
            threshold=0.8,
            weight=self.weight,
            status=self._status,
        )


def test_check_quality_auto_repair_thresholds_and_alerts():
    monitor = DataQualityMonitor(
        data_source="unit",
        config={
            "metrics_enabled": ["dummy"],
            "auto_repair": True,
            "thresholds": {"dummy": 0.6},
            "alert_threshold": 0.9,
        },
    )

    monitor.rules = [DummyRule()]
    monitor.metrics_enabled = ["dummy"]

    alerts = []

    def handler(payload):
        alerts.append(payload)

    monitor.register_alert_handler(handler)

    raw_data = pd.DataFrame({"a": [1.0, None, 3.0]})
    report = monitor.check_quality(raw_data)

    assert report["overall_score"] == 0.2
    assert report["alert_triggered"] is True
    assert len(alerts) == 1
    assert alerts[0]["quality_score"] == 0.2

    assert "repair_actions" in report
    assert isinstance(report["repair_actions"], list)
    assert "dummy" in report
    assert report["threshold_violations"]

    history = monitor.get_metrics_history()
    assert len(history) == 1
    assert history[0]["metrics"]["dummy"] == 0.2


def test_record_metrics_and_history_truncation():
    monitor = DataQualityMonitor(data_source="history", config={"metrics_enabled": ["dummy"]})
    monitor.rules = [DummyRule()]
    monitor.metrics_enabled = ["dummy"]

    df = pd.DataFrame({"value": [1, 2, 3]})

    entry = monitor.record_metrics(df)
    assert "metrics" in entry
    assert entry["metrics"]["dummy"] == 0.2

    history = monitor.get_metrics_history(limit=1)
    assert len(history) == 1
    assert history[0]["metrics"]["dummy"] == 0.2

