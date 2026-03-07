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


from datetime import datetime, timedelta

import pytest

from src.data.quality.monitor import DataQualityMonitor
from src.data.quality.validator import ValidationResult


def _build_result(
    metrics=None,
    is_valid=True,
    errors=None,
    ts=None,
) -> ValidationResult:
    return ValidationResult(
        is_valid=is_valid,
        metrics=metrics or {"completeness": 0.95, "accuracy": 0.96},
        errors=errors or [],
        timestamp=(ts or datetime.now()).isoformat(),
        data_type="unit",
    )


def test_monitor_records_history_and_triggers_alert(monkeypatch):
    monitor = DataQualityMonitor()
    captured = {"email": [], "sms": []}

    monkeypatch.setattr(monitor, "_send_email_alert", captured["email"].append)
    monkeypatch.setattr(monitor, "_send_sms_alert", captured["sms"].append)

    result = _build_result(metrics={"completeness": 0.6, "accuracy": 0.9})
    monitor.monitor(result)

    assert monitor.history
    assert monitor.history[-1]["alert_level"] == "critical"
    assert captured["email"]
    assert captured["sms"]


def test_monitor_with_high_scores_does_not_alert(monkeypatch):
    monitor = DataQualityMonitor()
    called = {"triggered": False}
    monkeypatch.setattr(
        monitor,
        "_trigger_alert",
        lambda level, result: called.__setitem__("triggered", True),
    )

    result = _build_result(metrics={"completeness": 0.99, "accuracy": 0.98})
    monitor.monitor(result)

    assert called["triggered"] is False
    assert monitor.history[-1]["alert_level"] is None


def test_generate_report_filters_by_days():
    monitor = DataQualityMonitor()
    now = datetime.now()
    recent = {
        "timestamp": (now - timedelta(days=1)).isoformat(),
        "metrics": {"completeness": 0.8, "accuracy": 0.9},
        "errors": [],
        "alert_level": "warning",
    }
    old = {
        "timestamp": (now - timedelta(days=10)).isoformat(),
        "metrics": {"completeness": 0.7, "accuracy": 0.85},
        "errors": [],
        "alert_level": None,
    }
    monitor.history.extend([recent, old])

    report = monitor.generate_report(days=7)
    assert report["metrics"]["completeness"]["avg"] == pytest.approx(0.8)
    assert report["total_alerts"] == 1


def test_create_alert_message_contains_details():
    monitor = DataQualityMonitor()
    result = _build_result(metrics={"completeness": 0.6}, errors=["missing data"])
    msg = monitor._create_alert_message("critical", result)

    assert "CRITICAL" in msg
    assert "missing data" in msg
    assert "completeness" in msg

