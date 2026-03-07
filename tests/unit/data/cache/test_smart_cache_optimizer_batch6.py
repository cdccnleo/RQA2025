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

from src.data.cache.smart_cache_optimizer import (
    CacheEntry,
    SmartCacheOptimizer,
    _InMemoryDataCache,
)
from src.data.interfaces.standard_interfaces import DataSourceType


@pytest.fixture
def optimizer(monkeypatch):
    monkeypatch.setattr(SmartCacheOptimizer, "_start_preload_scheduler", lambda self: None)
    opt = SmartCacheOptimizer(_InMemoryDataCache())
    yield opt
    opt.shutdown()


def test_smart_set_creates_entry_with_priority_and_ttl(optimizer):
    try:
        success = optimizer.smart_set("k1", {"value": 1}, DataSourceType.STOCK)
        print(f"Success: {success}")
        if not success:
            print(f"Cache entries: {list(optimizer._cache_entries.keys())}")
            print(f"Cache get: {optimizer.cache.get('k1')}")
        assert success is True
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        raise

    entry = optimizer._cache_entries["k1"]
    assert entry.priority == 3  # STOCK 的优先级
    assert entry.ttl_seconds == 3600
    assert optimizer.cache.get("k1") == {"value": 1}


def test_smart_get_invalidates_expired_entries(optimizer):
    expired_entry = CacheEntry(
        key="k2",
        data_type=DataSourceType.STOCK,
        value="doomed",
        timestamp=datetime.now() - timedelta(hours=2),
        expiry_time=datetime.now() - timedelta(seconds=1),
    )
    optimizer._cache_entries["k2"] = expired_entry
    optimizer.cache.set("k2", "doomed")

    result = optimizer.smart_get("k2", DataSourceType.STOCK)

    assert result is None
    assert "k2" not in optimizer._cache_entries
    assert optimizer.cache.get("k2") is None


def test_should_invalidate_by_access_pattern_detects_idle_entry():
    entry = CacheEntry(
        key="k3",
        data_type=DataSourceType.NEWS,
        value="idle",
        timestamp=datetime.now(),
        last_access=datetime.now() - timedelta(hours=25),
        access_count=1,
    )
    optimizer = object.__new__(SmartCacheOptimizer)
    assert SmartCacheOptimizer._should_invalidate_by_access_pattern(optimizer, entry) is True


def test_should_invalidate_by_freshness_respects_data_type_requirements():
    entry = CacheEntry(
        key="k4",
        data_type=DataSourceType.CRYPTO,
        value="stale",
        timestamp=datetime.now() - timedelta(seconds=400),
    )
    optimizer = object.__new__(SmartCacheOptimizer)
    assert SmartCacheOptimizer._should_invalidate_by_freshness(optimizer, entry) is True

