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
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

if "advanced_quality_monitor" not in sys.modules:
    sys.modules["advanced_quality_monitor"] = importlib.import_module(
        "src.data.quality.advanced_quality_monitor"
    )

from src.data.quality.advanced_quality_monitor import (
    DataQualityReport,
    QualityAlert,
    QualityDimension,
    QualityLevel,
    QualityMetric,
)
from src.data.quality.enhanced_quality_monitor_v2 import EnhancedQualityMonitorV2


def _build_sample_dataframe(rows: int = 20) -> pd.DataFrame:
    data = {
        "id": np.arange(rows),
        "value": np.linspace(0, 1, rows),
        "category": ["A"] * (rows // 2) + ["B"] * (rows - rows // 2),
        "timestamp": pd.date_range("2024-01-01", periods=rows, freq="H"),
    }
    df = pd.DataFrame(data)
    df.loc[0, "value"] = np.nan
    return df


def _metric(dimension: QualityDimension, score: float) -> QualityMetric:
    return QualityMetric(
        dimension=dimension,
        score=score,
        level=QualityLevel.GOOD,
        details={},
        timestamp=datetime.now(),
        source="unit",
    )


@pytest.mark.asyncio
async def test_monitor_quality_real_time_generates_full_report():
    monitor = EnhancedQualityMonitorV2(config={"monitoring_enabled": False})
    data = _build_sample_dataframe()

    report = await monitor.monitor_quality_real_time(data, data_source="test_source")

    assert isinstance(report, DataQualityReport)
    assert report.data_source == "test_source"
    assert len(report.metrics) == len(QualityDimension)
    assert any(alert.severity in {"warning", "critical"} for alert in report.alerts)
    assert report.overall_level in list(QualityLevel)


@pytest.mark.asyncio
async def test_detect_anomalies_flags_low_scores_and_outliers(monkeypatch):
    monitor = EnhancedQualityMonitorV2(config={"monitoring_enabled": False})
    base = np.ones(100)
    outliers = np.linspace(50, 60, 30)
    data = pd.DataFrame({"value": np.concatenate([base, outliers])})

    monkeypatch.setattr(
        monitor.scaler,
        "fit_transform",
        lambda values: values,
    )
    monkeypatch.setattr(
        monitor.anomaly_detector,
        "fit_predict",
        lambda values: np.array(
            [1] * 90 + [-1] * (len(values) - 90), dtype=int
        ),
    )
    metrics = {
        QualityDimension.ACCURACY: QualityMetric(
            dimension=QualityDimension.ACCURACY,
            score=0.5,
            level=QualityLevel.POOR,
            details={},
            timestamp=datetime.now(),
            source="unit-test",
        )
    }

    anomalies = await monitor._detect_anomalies(data, metrics)

    assert any(a["dimension"] == QualityDimension.ACCURACY.value for a in anomalies)
    assert any(a["dimension"] == "data_anomaly" for a in anomalies)


def test_export_quality_report_serializes_enums():
    monitor = EnhancedQualityMonitorV2(config={"monitoring_enabled": False})
    metrics = {
        QualityDimension.COMPLETENESS: QualityMetric(
            dimension=QualityDimension.COMPLETENESS,
            score=0.95,
            level=QualityLevel.EXCELLENT,
            details={},
            timestamp=datetime.now(),
            source="test",
        )
    }
    alert = QualityAlert(
        alert_id="alert-1",
        dimension=QualityDimension.ACCURACY,
        severity="warning",
        message="accuracy low",
        details={},
        timestamp=datetime.now(),
        status="active",
    )
    report = DataQualityReport(
        report_id="r1",
        overall_score=0.9,
        overall_level=QualityLevel.GOOD,
        metrics=metrics,
        alerts=[alert],
        recommendations=["fix accuracy"],
        timestamp=datetime.now(),
        data_source="demo",
    )

    payload = json.loads(monitor.export_quality_report(report, format="json"))

    assert payload["overall_level"] == QualityLevel.GOOD.value
    assert payload["metrics"]["completeness"]["level"] in {
        QualityLevel.EXCELLENT.value,
        str(QualityLevel.EXCELLENT),
    }
    assert payload["alerts"][0]["dimension"] == QualityDimension.ACCURACY.value


@pytest.mark.asyncio
async def test_analyze_quality_trends_requires_history():
    monitor = EnhancedQualityMonitorV2(config={"monitoring_enabled": False})
    for score in [0.9, 0.88, 0.85, 0.82, 0.8, 0.78]:
        monitor.quality_history.append(_metric(QualityDimension.COMPLETENESS, score))

    metrics = {
        QualityDimension.COMPLETENESS: _metric(QualityDimension.COMPLETENESS, 0.75)
    }

    trends = await monitor._analyze_quality_trends(metrics)
    assert any(t.dimension == QualityDimension.COMPLETENESS for t in trends)


@pytest.mark.asyncio
async def test_generate_repair_suggestions_returns_messages():
    monitor = EnhancedQualityMonitorV2(config={"monitoring_enabled": False})
    data = _build_sample_dataframe()
    metrics = {
        QualityDimension.COMPLETENESS: _metric(QualityDimension.COMPLETENESS, 0.6),
        QualityDimension.TIMELINESS: _metric(QualityDimension.TIMELINESS, 0.5),
    }

    suggestions = await monitor._generate_repair_suggestions(data, metrics)

    assert any("填充" in suggestion for suggestion in suggestions)
    assert any("时效性" in suggestion or "时效" in suggestion for suggestion in suggestions)


@pytest.mark.asyncio
async def test_generate_quality_alerts_merges_metric_and_anomaly():
    monitor = EnhancedQualityMonitorV2(config={"monitoring_enabled": False})
    metrics = {
        QualityDimension.ACCURACY: _metric(QualityDimension.ACCURACY, 0.5),
    }
    anomalies = [{"dimension": "data_anomaly", "severity": "critical"}]

    alerts = await monitor._generate_quality_alerts(metrics, anomalies)

    assert any(alert.details.get("dimension") == "data_anomaly" for alert in alerts)
    assert any(alert.dimension == QualityDimension.ACCURACY for alert in alerts)


def test_shutdown_joins_monitor_thread():
    monitor = EnhancedQualityMonitorV2(config={"monitoring_enabled": False})
    joined = {"called": False}

    class DummyThread:
        def join(self, timeout=None):
            joined["called"] = True

    monitor._monitor_thread = DummyThread()
    monitor.shutdown()

    assert joined["called"] is True


def test_check_quality_trends_logs_warning(caplog):
    monitor = EnhancedQualityMonitorV2(config={"monitoring_enabled": False})
    for score in [0.9, 0.7, 0.5, 0.3, 0.2, 0.1]:
        monitor.quality_history.append(_metric(QualityDimension.ACCURACY, score))

    with caplog.at_level("WARNING"):
        monitor._check_quality_trends()

    assert any("Quality trend decreasing" in record.message for record in caplog.records)

