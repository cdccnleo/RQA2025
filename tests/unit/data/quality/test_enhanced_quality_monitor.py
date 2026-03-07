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
import pytest
from datetime import datetime, timedelta

from src.data.quality.enhanced_quality_monitor import (
    EnhancedQualityMonitor,
    QualityMetrics,
    TrendAnalyzer,
    StatisticalAnomalyDetector,
    QualityAlertManager,
    QualityTrend,
)


@pytest.fixture(autouse=True)
def disable_background_thread(monkeypatch):
    monkeypatch.setattr(
        "src.data.quality.enhanced_quality_monitor.EnhancedQualityMonitor._start_quality_monitoring",
        lambda self: None,
    )


def _build_metrics(timestamp: datetime, data_type: str, overall: float) -> QualityMetrics:
    return QualityMetrics(
        timestamp=timestamp.isoformat(),
        data_type=data_type,
        completeness=overall,
        accuracy=overall,
        consistency=overall,
        timeliness=overall,
        validity=overall,
        uniqueness=overall,
        overall_score=overall,
        details={"id": data_type},
    )


def test_check_data_quality_generates_metrics_and_alerts(monkeypatch):
    monitor = EnhancedQualityMonitor(enable_alerting=True, enable_trend_analysis=False)
    captured = []

    def fake_send(anomalies):
        captured.extend(anomalies)

    monitor._alert_manager.send_alerts = fake_send

    metrics = monitor.check_data_quality(pd.DataFrame(), "empty", "dataset")

    assert metrics.overall_score == 0.0
    assert monitor._quality_history
    assert captured
    assert all(anomaly.severity in {"high", "medium"} for anomaly in captured)


def test_check_data_quality_with_real_data_updates_trends():
    monitor = EnhancedQualityMonitor(enable_alerting=False, enable_trend_analysis=True)
    dates = pd.date_range("2024-01-01", periods=3, freq="D")
    df = pd.DataFrame(
        {
            "close": [10.0, 10.5, 10.7],
            "volume": [1000, 1100, 900],
            "date": dates.strftime("%Y-%m-%d"),
            "symbol": ["000001.SH"] * 3,
        }
    )

    monitor.check_data_quality(df, "stock", "dataset-1")
    monitor.check_data_quality(df, "stock", "dataset-2")

    assert monitor.get_overall_quality_score() > 0
    trends = monitor.get_quality_trends()
    assert "completeness" in trends
    assert monitor._quality_history


def test_get_quality_report_filters_by_type_and_time():
    monitor = EnhancedQualityMonitor(enable_alerting=False, enable_trend_analysis=False)

    assert monitor.get_quality_report() == {"error": "No quality data available"}

    now = datetime.now()
    monitor._quality_history.append(_build_metrics(now, "stock", 0.9))
    monitor._quality_history.append(
        _build_metrics(now - timedelta(days=5), "bond", 0.5)
    )

    report = monitor.get_quality_report(data_type="stock", time_range="1d")
    assert report["summary"]["total_checks"] == 1
    assert report["summary"]["avg_overall_score"] == pytest.approx(0.9)
    assert "overall_score" in report["metrics"]
    assert report["trends"]

    missing = monitor.get_quality_report(data_type="bond", time_range="1h")
    assert missing["error"] == "No data matching criteria"


def test_get_quality_report_with_default_time_range():
    monitor = EnhancedQualityMonitor(enable_alerting=False, enable_trend_analysis=False)
    now = datetime.now()
    monitor._quality_history.append(_build_metrics(now - timedelta(days=5), "stock", 0.7))

    report = monitor.get_quality_report(time_range="xyz")
    assert report["summary"]["total_checks"] == 1


def test_trend_analyzer_detects_direction():
    analyzer = TrendAnalyzer(window_size=5)
    analyzer.add_value(0.5, datetime.now().isoformat())
    analyzer.add_value(0.6, datetime.now().isoformat())
    analyzer.add_value(0.8, datetime.now().isoformat())
    trend = analyzer.get_trend()

    assert trend.trend_direction == "improving"
    assert 0.0 <= trend.prediction <= 1.0

    analyzer = TrendAnalyzer()
    analyzer.add_value(0.5, datetime.now().isoformat())
    minimal = analyzer.get_trend()
    assert minimal.trend_direction == "stable"


def test_cleanup_old_data_prunes_history():
    monitor = EnhancedQualityMonitor(enable_alerting=False, enable_trend_analysis=False)
    monitor._quality_history.append(
        _build_metrics(datetime.now() - timedelta(days=40), "old", 0.4)
    )
    monitor._quality_history.append(_build_metrics(datetime.now(), "new", 0.9))

    monitor._cleanup_old_data()

    remaining = list(monitor._quality_history)
    assert len(remaining) == 1
    assert remaining[0].data_type == "new"


def test_periodic_quality_check_warns_on_declining_trend(monkeypatch, caplog):
    monitor = EnhancedQualityMonitor(enable_alerting=False, enable_trend_analysis=False)
    warning_trend = QualityTrend(
        metric_name="overall",
        trend_direction="declining",
        trend_strength=0.9,
        change_rate=-0.2,
        prediction=0.3,
        confidence=0.8,
    )
    monkeypatch.setattr(
        monitor, "get_quality_trends", lambda: {"overall": warning_trend, "other": warning_trend}
    )

    with caplog.at_level("WARNING"):
        monitor._perform_periodic_quality_check()

    assert any("下降趋势" in record.message for record in caplog.records)


def test_statistical_anomaly_detector_reports():
    detector = StatisticalAnomalyDetector()
    metrics = _build_metrics(datetime.now(), "equity", 0.5)
    metrics.completeness = 0.6

    anomalies = detector.detect(metrics)

    assert len(anomalies) == 2
    assert {a.anomaly_type for a in anomalies} == {
        "low_quality_score",
        "low_completeness",
    }


def test_quality_alert_manager_records_history(caplog):
    manager = QualityAlertManager()
    metrics = _build_metrics(datetime.now(), "credit", 0.4)
    metrics.completeness = 0.6
    anomalies = StatisticalAnomalyDetector().detect(metrics)

    with caplog.at_level("WARNING"):
        manager.send_alerts(anomalies)

    assert manager.alert_history == anomalies
    assert any("质量异常" in record.message for record in caplog.records)

