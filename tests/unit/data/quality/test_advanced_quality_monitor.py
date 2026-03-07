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
from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from src.data.quality.advanced_quality_monitor import (
    AdvancedQualityMonitor,
    QualityDimension,
    QualityLevel,
)


@pytest.mark.asyncio
async def test_check_completeness_handles_missing_values():
    data = pd.DataFrame(
        {
            "value": [1, np.nan, 3],
            "score": [4, 5, np.nan],
        }
    )
    monitor = AdvancedQualityMonitor()

    metric = await monitor.check_completeness(data)

    assert metric.dimension == QualityDimension.COMPLETENESS
    assert metric.level == QualityLevel.POOR
    assert pytest.approx(metric.score, rel=1e-4) == 66.6666666667
    assert metric.details["missing_cells"] == 2
    assert metric.details["column_completeness"]["value"]["missing_count"] == 1


@pytest.mark.asyncio
async def test_check_accuracy_with_reference_data():
    data = pd.DataFrame({"price": [10, 11, 12, 13]})
    reference = pd.DataFrame({"price": [9.5, 10.5, 11.5, 12.5]})

    monitor = AdvancedQualityMonitor()
    metric = await monitor.check_accuracy(data, reference_data=reference)

    assert metric.dimension == QualityDimension.ACCURACY
    assert metric.level == QualityLevel.EXCELLENT
    assert metric.details["total_checks"] == 2  # range + correlation
    assert pytest.approx(metric.details["accuracy_checks"]["price_correlation"], rel=1e-6) == 1.0


@pytest.mark.asyncio
async def test_check_timeliness_without_timestamp_column():
    data = pd.DataFrame({"value": [1, 2, 3]})
    monitor = AdvancedQualityMonitor()

    metric = await monitor.check_timeliness(data)

    assert metric.dimension == QualityDimension.TIMELINESS
    assert metric.level == QualityLevel.EXCELLENT
    assert "no_timestamp" in metric.details
    assert metric.details["no_timestamp"]["delay_score"] == 100


@pytest.mark.asyncio
async def test_monitor_quality_generates_alerts_and_history():
    old_dates = pd.date_range("2010-01-01", periods=5, freq="D")
    data = pd.DataFrame(
        {
            "price": [1, np.nan, np.nan, 1000, np.nan],
            "volume": [np.nan, 2, 3, 4, 5],
            "timestamp": old_dates,
        }
    )

    monitor = AdvancedQualityMonitor()
    report = await monitor.monitor_quality(data, data_source="unknown")

    assert report.overall_level in {QualityLevel.POOR, QualityLevel.UNACCEPTABLE, QualityLevel.FAIR}
    assert monitor.quality_history and monitor.quality_history[-1] is report
    assert monitor.active_alerts or report.alerts

    # 至少应包含针对完整性或时效性的建议
    assert any("完整" in rec or "时效" in rec for rec in report.recommendations)


@pytest.mark.asyncio
async def test_export_report_creates_json_file(tmp_path, monkeypatch):
    data = pd.DataFrame(
        {
            "price": [100, 101, 102],
            "timestamp": pd.date_range(datetime.now(), periods=3, freq="h"),
        }
    )

    monitor = AdvancedQualityMonitor()
    report = await monitor.monitor_quality(data, data_source="binance")

    monkeypatch.chdir(tmp_path)
    path_str = await monitor.export_report(report)
    path = tmp_path / path_str

    assert path.exists()
    content = json.loads(path.read_text(encoding="utf-8"))
    assert content["report_id"] == report.report_id
    assert content["metrics"]["completeness"]["level"] == QualityLevel.EXCELLENT.value

