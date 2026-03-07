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
import pytest

from src.data.quality.data_quality_monitor import (
    AlertLevel,
    DataQualityMonitor,
    DataQualityRule,
    QualityLevel,
    QualityMetric,
    QualityReport,
)


class _StaticRule(DataQualityRule):
    def __init__(self, name: str, value: float, status: str = "excellent", weight: float = 1.0):
        super().__init__(name=name, weight=weight)
        self._value = value
        self._status = status

    def set(self, value: float, status: str = "excellent"):
        self._value = value
        self._status = status

    def check(self, _data: pd.DataFrame) -> QualityMetric:
        return QualityMetric(
            name=self.name,
            value=self._value,
            threshold=0.8,
            weight=self.weight,
            status=self._status,
        )


class _BrokenRule(DataQualityRule):
    def __init__(self, name: str = "broken"):
        super().__init__(name=name)

    def check(self, _data: pd.DataFrame) -> QualityMetric:
        raise RuntimeError("boom")


class _OverflowRule(DataQualityRule):
    def __init__(self):
        super().__init__(name="overflow")

    def check(self, _data: pd.DataFrame) -> QualityMetric:
        return QualityMetric(name=self.name, value=1.5, threshold=0.0, status="excellent")


def test_check_quality_handles_rule_exception():
    monitor = DataQualityMonitor(config={"metrics_enabled": ["broken"], "alert_enabled": False})
    monitor.rules = [_BrokenRule()]
    monitor.metrics_enabled = ["broken"]

    result = monitor.check_quality(pd.DataFrame({"a": [1, 2, 3]}))

    assert result["overall_score"] == 0.0
    assert result["metrics"]["broken"] == 0.0
    assert result["alert_level"] == AlertLevel.CRITICAL.value
    assert result["anomalies"] == []
    assert result["quality_report"].metrics["broken"].details["error"] == "boom"


def test_check_quality_resolves_quality_and_alert_levels():
    rule = _StaticRule(name="dummy", value=0.75, status="poor")
    monitor = DataQualityMonitor(
        data_source="trade",
        config={"metrics_enabled": ["dummy"], "alert_threshold": 0.9, "thresholds": {}, "auto_repair": False},
    )
    monitor.rules = [rule]
    monitor.metrics_enabled = ["dummy"]

    report = monitor.check_quality(pd.DataFrame({"x": [1, 2, 3]}))

    assert report["overall_score"] == pytest.approx(0.75)
    assert report["quality_level"] == QualityLevel.FAIR.value
    assert report["alert_level"] == AlertLevel.WARNING.value
    assert report["recommendations"] == []


def test_calculate_metrics_clamps_values():
    monitor = DataQualityMonitor(config={"metrics_enabled": ["overflow"], "alert_enabled": False})
    monitor.rules = [_OverflowRule()]
    monitor.metrics_enabled = ["overflow"]

    metrics = monitor.calculate_metrics(pd.DataFrame({"x": [1, 2, 3]}))
    assert metrics["overflow"] == 1.0


def test_check_thresholds_reports_violations():
    rule = _StaticRule("dummy", value=0.7, status="poor")
    monitor = DataQualityMonitor(config={"metrics_enabled": ["dummy"], "thresholds": {"dummy": 0.8}})
    monitor.rules = [rule]
    monitor.metrics_enabled = ["dummy"]

    violations = monitor.check_thresholds(pd.DataFrame({"x": [1]}))
    assert violations == [{"metric": "dummy", "value": 0.7, "threshold": 0.8}]


def test_generate_summary_report_with_and_without_history():
    monitor = DataQualityMonitor(config={"metrics_enabled": ["dummy"], "alert_enabled": False})
    summary = monitor.generate_summary_report()
    assert summary == {"message": "没有历史数据"}

    rule = _StaticRule("dummy", value=0.95, status="excellent")
    monitor.rules = [rule]
    monitor.metrics_enabled = ["dummy"]

    monitor.check_quality(pd.DataFrame({"x": [1, 2]}))
    rule.set(0.55, status="critical")
    monitor.check_quality(pd.DataFrame({"x": [3, 4]}))

    summary = monitor.generate_summary_report(days=1)
    assert summary["total_reports"] == 2
    assert summary["quality_level_distribution"]["excellent"] == 1
    assert summary["quality_level_distribution"]["critical"] == 1
    assert summary["alert_level_distribution"]["critical"] == 1
    assert summary["average_score"] == pytest.approx((0.95 + 0.55) / 2, rel=1e-6)
    assert summary["top_issues"][0]["issue"] == "dummy"


def test_get_quality_and_anomaly_history_filters_by_days():
    monitor = DataQualityMonitor(config={"metrics_enabled": ["dummy"], "alert_enabled": False})
    rule = _StaticRule("dummy", value=0.4, status="critical")
    monitor.rules = [rule]
    monitor.metrics_enabled = ["dummy"]

    monitor.check_quality(pd.DataFrame({"x": [1, 2, 3]}))

    old_report = QualityReport(
        timestamp=datetime.now() - timedelta(days=10),
        data_source="legacy",
        data_shape=(0, 0),
        overall_score=0.5,
        quality_level=QualityLevel.POOR,
        metrics={},
        anomalies=[],
        recommendations=[],
    )
    monitor.report_history.append(old_report)

    history = monitor.get_quality_history(days=7)
    assert all(report.timestamp >= datetime.now() - timedelta(days=7) for report in history)

    anomaly_record = monitor.anomaly_history[0]
    anomaly_record.timestamp = datetime.now() - timedelta(days=10)
    anomalies = monitor.get_anomaly_history(days=7)
    assert anomalies == []


def test_export_report_formats_and_errors():
    report = QualityReport(
        timestamp=datetime(2025, 1, 1, 12, 0, 0),
        data_source="test",
        data_shape=(10, 3),
        overall_score=0.8,
        quality_level=QualityLevel.GOOD,
        metrics={},
        anomalies=[],
        recommendations=[],
        alert_level=AlertLevel.WARNING,
    )
    monitor = DataQualityMonitor()

    json_output = monitor.export_report(report, format="json")
    payload = json.loads(json_output)
    assert payload["data_source"] == "test"
    assert payload["overall_score"] == 0.8

    csv_output = monitor.export_report(report, format="csv")
    assert "data_source" in csv_output
    assert "test" in csv_output

    with pytest.raises(ValueError):
        monitor.export_report(report, format="xml")


def test_auto_repair_numeric_and_categorical():
    monitor = DataQualityMonitor()
    df = pd.DataFrame({"num": [1.0, None, 3.0], "cat": ["a", None, "a"]})

    repaired, actions = monitor._auto_repair_data(df)

    assert repaired["num"].isna().sum() == 0
    assert repaired["cat"].isna().sum() == 0
    assert any(action["column"] == "num" and action["method"] == "mean_fill" for action in actions)
    assert any(action["column"] == "cat" and action["method"] == "mode_fill" for action in actions)

