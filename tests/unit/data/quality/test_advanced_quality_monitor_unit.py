#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
AdvancedQualityMonitor 单元测试
覆盖质量等级映射、告警/建议生成、主监控流程、趋势统计与报告导出等核心路径。
"""

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

from src.data.quality.advanced_quality_monitor import (
    AdvancedQualityMonitor,
    DataQualityReport,
    QualityAlert,
    QualityDimension,
    QualityLevel,
    QualityMetric,
)


def _build_metric(dimension: QualityDimension, score: float,
                  level: QualityLevel = None) -> QualityMetric:
    level = level or (
        QualityLevel.GOOD if score >= 80 else QualityLevel.POOR
    )
    return QualityMetric(
        dimension=dimension,
        score=score,
        level=level,
        details={"score": score},
        timestamp=datetime.now(),
        source="unit-test",
    )


def test_quality_level_mapping():
    """验证质量等级映射逻辑"""
    monitor = AdvancedQualityMonitor()

    assert monitor._get_quality_level(95) is QualityLevel.EXCELLENT
    assert monitor._get_quality_level(85) is QualityLevel.GOOD
    assert monitor._get_quality_level(75) is QualityLevel.FAIR
    assert monitor._get_quality_level(65) is QualityLevel.POOR
    assert monitor._get_quality_level(40) is QualityLevel.UNACCEPTABLE


@pytest.mark.asyncio
async def test_generate_alerts_and_recommendations():
    """低质量维度应产生告警与建议"""
    monitor = AdvancedQualityMonitor()
    metrics = {
        QualityDimension.COMPLETENESS: _build_metric(
            QualityDimension.COMPLETENESS, 55, QualityLevel.UNACCEPTABLE
        ),
        QualityDimension.ACCURACY: _build_metric(
            QualityDimension.ACCURACY, 68, QualityLevel.POOR
        ),
        QualityDimension.TIMELINESS: _build_metric(
            QualityDimension.TIMELINESS, 92, QualityLevel.EXCELLENT
        ),
    }

    alerts = await monitor.generate_alerts(metrics)
    recommendations = await monitor.generate_recommendations(metrics)

    assert len(alerts) == 2
    assert all(isinstance(alert, QualityAlert) for alert in alerts)
    assert len(monitor.active_alerts) == 2
    assert any("完整性" in rec for rec in recommendations)
    assert any("准确性" in rec for rec in recommendations)


@pytest.mark.asyncio
async def test_monitor_quality_with_stubbed_checks(monkeypatch):
    """monitor_quality 应聚合各项指标并记录报告"""
    monitor = AdvancedQualityMonitor()
    fake_metrics = {
        QualityDimension.COMPLETENESS: _build_metric(
            QualityDimension.COMPLETENESS, 90, QualityLevel.EXCELLENT
        ),
        QualityDimension.ACCURACY: _build_metric(
            QualityDimension.ACCURACY, 85, QualityLevel.GOOD
        ),
        QualityDimension.CONSISTENCY: _build_metric(
            QualityDimension.CONSISTENCY, 82, QualityLevel.GOOD
        ),
        QualityDimension.TIMELINESS: _build_metric(
            QualityDimension.TIMELINESS, 50, QualityLevel.UNACCEPTABLE
        ),
        QualityDimension.VALIDITY: _build_metric(
            QualityDimension.VALIDITY, 88, QualityLevel.GOOD
        ),
        QualityDimension.RELIABILITY: _build_metric(
            QualityDimension.RELIABILITY, 92, QualityLevel.EXCELLENT
        ),
    }

    def _stub(metric_key):
        async def _inner(*_, **__):
            return fake_metrics[metric_key]

        return _inner

    monkeypatch.setattr(monitor, "check_completeness", _stub(QualityDimension.COMPLETENESS))
    monkeypatch.setattr(monitor, "check_accuracy", _stub(QualityDimension.ACCURACY))
    monkeypatch.setattr(monitor, "check_consistency", _stub(QualityDimension.CONSISTENCY))
    monkeypatch.setattr(monitor, "check_timeliness", _stub(QualityDimension.TIMELINESS))
    monkeypatch.setattr(monitor, "check_validity", _stub(QualityDimension.VALIDITY))
    monkeypatch.setattr(monitor, "check_reliability", _stub(QualityDimension.RELIABILITY))

    df = pd.DataFrame({"value": [1, 2, 3]})
    report = await monitor.monitor_quality(df, data_source="unit")

    expected_score = sum(metric.score for metric in fake_metrics.values()) / len(fake_metrics)
    assert report.overall_score == pytest.approx(expected_score)
    assert report.metrics[QualityDimension.TIMELINESS].level == QualityLevel.UNACCEPTABLE
    assert len(report.alerts) == 1  # 仅时效性较差
    assert len(monitor.quality_history) == 1


@pytest.mark.asyncio
async def test_get_quality_trends(monkeypatch):
    """质量历史应可聚合趋势"""
    monitor = AdvancedQualityMonitor()
    result_empty = await monitor.get_quality_trends()
    assert "message" in result_empty

    recent_metric_low = _build_metric(QualityDimension.COMPLETENESS, 60, QualityLevel.POOR)
    recent_metric_high = _build_metric(QualityDimension.COMPLETENESS, 80, QualityLevel.GOOD)

    report_old = DataQualityReport(
        report_id="r1",
        overall_score=65,
        overall_level=QualityLevel.POOR,
        metrics={QualityDimension.COMPLETENESS: recent_metric_low},
        alerts=[],
        recommendations=[],
        timestamp=datetime.now() - timedelta(days=1),
        data_source="ds1",
    )
    report_new = DataQualityReport(
        report_id="r2",
        overall_score=85,
        overall_level=QualityLevel.GOOD,
        metrics={QualityDimension.COMPLETENESS: recent_metric_high},
        alerts=[],
        recommendations=[],
        timestamp=datetime.now(),
        data_source="ds1",
    )
    monitor.quality_history.extend([report_old, report_new])

    trends = await monitor.get_quality_trends(days=7)
    assert trends["total_reports"] == 2
    assert trends["trends"]["completeness"]["trend"] == "improving"


@pytest.mark.asyncio
async def test_export_report_json(tmp_path, monkeypatch):
    """导出报告应生成 JSON 文件"""
    monitor = AdvancedQualityMonitor()
    monkeypatch.chdir(tmp_path)

    report = DataQualityReport(
        report_id="unit",
        overall_score=90,
        overall_level=QualityLevel.EXCELLENT,
        metrics={QualityDimension.ACCURACY: _build_metric(QualityDimension.ACCURACY, 90)},
        alerts=[],
        recommendations=["keep going"],
        timestamp=datetime.now(),
        data_source="unit",
    )

    path_str = await monitor.export_report(report)
    report_path = tmp_path / path_str
    assert report_path.exists()

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["report_id"] == "unit"
    assert data["metrics"]["accuracy"]["score"] == 90


@pytest.mark.asyncio
async def test_check_data_quality_returns_scores():
    """check_data_quality 应返回总体分数"""
    monitor = AdvancedQualityMonitor()
    df = pd.DataFrame(
        {
            "price": [100, 200, 300],
            "symbol": ["BTC", "ETH", "BNB"],
        }
    )

    result = await monitor.check_data_quality(df)
    assert result["overall_score"] > 0
    assert "completeness" in result["details"]


def test_track_metrics_dataframe_and_invalid():
    """track_metrics 应处理 DataFrame 与非法输入"""
    monitor = AdvancedQualityMonitor()
    df = pd.DataFrame({"value": [1, None, 3]})

    metrics_result = monitor.track_metrics(df, data_type="market")
    assert metrics_result["overall_score"] > 0
    assert metrics_result["metrics"]["completeness"]["details"]["null_count"] == 1

    invalid_result = monitor.track_metrics("invalid", data_type="text")
    assert invalid_result["overall_score"] == 0
    assert invalid_result["error"] == "Invalid data format"

