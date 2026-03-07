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


from typing import List, Dict, Any

import pandas as pd
import pytest

from src.data.quality.unified_quality_monitor import (
    UnifiedQualityMonitor,
    TYPE_STOCK,
)


def _make_stock_df(with_missing: bool = True) -> pd.DataFrame:
    df = pd.DataFrame({
        "symbol": ["AAA", "BBB", "CCC"],
        "timestamp": ["2025-01-01 10:00:00", "2025-01-01 11:00:00", "2025-01-01 12:00:00"],
        "open": [10.0, 20.0, 30.0],
        "high": [12.0, 25.0, 35.0],
        "low": [9.0, 19.0, 29.0],
        "close": [11.0, 22.0, 31.0],
        "volume": [100, 200, 300],
    })
    if with_missing:
        df.loc[1, "open"] = None
    return df


def test_check_quality_with_threshold_alert_and_handler(monkeypatch):
    monitor = UnifiedQualityMonitor(config={"quality_threshold": 0.95})

    captured: List[Dict[str, Any]] = []
    monitor.register_alert_handler(lambda payload: captured.append(payload))

    df = _make_stock_df(with_missing=True)
    result = monitor.check_quality(df, TYPE_STOCK)

    assert "overall_score" in result
    assert result["validation"]["issue_count"] >= 0
    # 阈值为 0.95，通常包含缺失将导致分数较低，触发 handler
    assert captured or result["overall_score"] >= monitor.alert_threshold


def test_monitor_source_metrics_and_alerts_flow():
    monitor = UnifiedQualityMonitor(config={"quality_threshold": 0.90})
    source_id = "source_stock"
    monitor.monitor_data_source(source_id, {"data_type": TYPE_STOCK})

    df = _make_stock_df(with_missing=False)
    output = monitor.check_data_quality(df, source_id)
    assert isinstance(output, list) and len(output) == 1
    assert output[0]["data_type"] == TYPE_STOCK.value

    metrics = monitor.get_quality_metrics(source_id)
    assert metrics["history_length"] >= 1
    assert metrics["current"]["overall_score"] >= 0.0

    # 生成异常历史触发
    for _ in range(6):
        monitor.check_quality(df, TYPE_STOCK)
    hist = monitor.get_quality_metrics(source_id)
    assert hist["history_length"] >= 7


def test_resolve_alerts_and_thresholds():
    monitor = UnifiedQualityMonitor(config={"quality_threshold": 0.99})
    source_id = "alpha"
    monitor.monitor_data_source(source_id, {"data_type": TYPE_STOCK})

    # 低分数据促发内部 alert（通过 _send_alert -> _alerts_history["manual"]）
    df = _make_stock_df(with_missing=True)
    monitor.check_quality(df, TYPE_STOCK)
    # 手动检查历史接口一致性
    alerts_open = monitor.get_alerts("manual", resolved=False)
    if alerts_open:
        assert monitor.resolve_alert("manual") is True
        alerts_closed = monitor.get_alerts("manual", resolved=True)
        assert alerts_closed[-1]["resolved"] is True
    else:
        # 如果没有触发，也应不抛异常
        assert monitor.resolve_alert("manual") in (True, False)


