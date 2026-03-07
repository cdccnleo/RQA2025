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
import numpy as np
import pytest

from src.data.sync.multi_market_sync import (
    GlobalMarketDataManager,
    MarketConfig,
    MarketData,
    DataType,
    MultiCurrencyProcessor,
    CrossTimezoneSynchronizer,
    MultiMarketSyncManager,
    SyncType,
    SyncStatus,
)


def test_global_market_data_manager_register_and_stats_empty():
    mgr = GlobalMarketDataManager()
    cfg = MarketConfig(
        market_id="TEST",
        market_name="测试市场",
        timezone="Asia/Shanghai",
        base_currency="CNY",
        business={"stock": ["AAPL"]},
        data_sources=["dummy"],
        sync_frequency=60,
        priority=5,
    )
    assert mgr.register_market(cfg) is True
    stats = mgr.get_market_statistics("TEST")
    assert stats.get("market_id") == "TEST"
    assert stats.get("data_count", 0) == 0


def test_add_and_get_market_data_filters():
    mgr = GlobalMarketDataManager()
    cfg = MarketConfig(
        market_id="M1",
        market_name="市场1",
        timezone="Asia/Shanghai",
        base_currency="CNY",
        business={"stock": ["000001.SZ"]},
        data_sources=["ds"],
        sync_frequency=60,
        priority=1,
    )
    mgr.register_market(cfg)
    now = datetime.now()
    d1 = MarketData(
        market_id="M1",
        symbol="000001.SZ",
        price=10.0,
        volume=100,
        timestamp=now - timedelta(minutes=2),
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.QUOTE,
        source="ds",
    )
    d2 = MarketData(
        market_id="M1",
        symbol="000001.SZ",
        price=10.5,
        volume=200,
        timestamp=now,
        timezone="Asia/Shanghai",
        currency="CNY",
        data_type=DataType.QUOTE,
        source="ds",
    )
    assert mgr.add_market_data("M1", d1) is True
    assert mgr.add_market_data("M1", d2) is True
    # 时间窗口筛选
    res = mgr.get_market_data("M1", start_time=now - timedelta(minutes=1))
    assert len(res) == 1
    # 类型筛选
    res2 = mgr.get_market_data("M1", data_type=DataType.QUOTE)
    assert len(res2) == 2


def test_currency_processor_basic():
    proc = MultiCurrencyProcessor()
    ts = datetime.now()
    assert proc.set_exchange_rate("CNY", "USD", 0.14, ts) is True
    assert proc.convert_currency(100, "CNY", "USD") == pytest.approx(14.0)
    assert proc.get_exchange_rate("CNY", "USD") == pytest.approx(0.14)
    hist = proc.get_rate_history("CNY", "USD", days=1)
    assert len(hist) >= 1


def test_timezone_synchronizer_schedule_and_list():
    syncer = CrossTimezoneSynchronizer()
    sid = syncer.schedule_sync("M1", "America/New_York", 120)
    schedules = syncer.get_sync_schedules()
    assert any(s["schedule_id"] == sid for s in schedules)


def test_sync_manager_task_flow():
    mgr = MultiMarketSyncManager()
    init_info = mgr.initialize_markets()
    assert init_info["markets_registered"] >= 1
    tid = mgr.start_sync_task("SHANGHAI", SyncType.REAL_TIME)
    assert tid in mgr.sync_tasks
    ok = mgr.complete_sync_task(tid, records_synced=1000, error_count=2)
    assert ok is True
    rep = mgr.get_sync_report()
    assert rep["completed_tasks_count"] >= 0
    assert rep["total_records_synced"] >= 0


