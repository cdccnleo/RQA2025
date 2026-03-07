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

from src.data.quality.unified_quality_monitor import (
    UnifiedQualityMonitor,
    QualityConfig,
)


def _mk_df(ts_offset_days=0, cols=("open", "high", "low", "close")):
    data = {
        "open": [1.0, 2.0, 3.0],
        "high": [1.5, 2.5, 3.5],
        "low": [0.5, 1.5, 2.5],
        "close": [1.1, 2.1, 3.1],
        "timestamp": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
    }
    df = pd.DataFrame(data)[list(cols) + (["timestamp"] if "timestamp" in data else [])]
    return df


def test_normalize_data_type_variants_and_history_aggregation():
    mon = UnifiedQualityMonitor(config=QualityConfig(enable_auto_repair=False))
    df = _mk_df()

    # 字符串类型 -> 兜底
    r1 = mon.check_quality(df, data_type="api")
    assert isinstance(r1, dict) and "overall_score" in r1 and isinstance(r1["data_type"], str)

    # 兼容对象（带 .value）
    class _Compat:
        def __init__(self, v): self.value = v
    r2 = mon.check_quality(df, data_type=_Compat("stock"))
    assert isinstance(r2["data_type"], str) and r2["data_type"].upper() in {"STOCK", "DATABASE", "TABLE"}

    # None -> 默认 STOCK
    r3 = mon.check_quality(df, data_type=None)
    assert isinstance(r3["data_type"], str)

    # 历史聚合：未指定类型时应汇总所有类型
    all_hist = mon.get_quality_history()
    stock_hist = mon.get_quality_history("stock")  # type: ignore[arg-type]
    assert len(all_hist) >= 3
    assert len(stock_hist) >= 1  # 至少一次 STOCK（None 回退或兼容对象）


def test_alert_threshold_trigger_and_handler_called(monkeypatch):
    # 提高阈值，确保触发告警
    cfg = QualityConfig(quality_threshold=0.99, alert_cooldown_minutes=0)
    mon = UnifiedQualityMonitor(config=cfg)
    df = _mk_df()

    called = {"n": 0, "payloads": []}
    def _handler(payload):
        called["n"] += 1
        called["payloads"].append(payload)
    mon.register_alert_handler(_handler)

    rep = mon.check_quality(df, data_type="stock")  # type: ignore[arg-type]
    assert rep["overall_score"] < 0.99
    # 告警处理器至少被调用一次
    assert called["n"] >= 1
    assert all("quality_score" in p for p in called["payloads"])


def test_register_alert_handler_type_error():
    mon = UnifiedQualityMonitor()
    with pytest.raises(TypeError):
        mon.register_alert_handler(123)  # type: ignore[arg-type]


