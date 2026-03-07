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


import pytest

from src.data.cache.cache_manager import CacheConfig, CacheManager


@pytest.fixture
def cache_manager(tmp_path):
    config = CacheConfig(
        max_size=2,
        ttl=5,
        enable_disk_cache=False,
        enable_stats=False,
    )
    manager = CacheManager(config)
    yield manager
    manager.stop()


def test_set_get_exists_updates_stats(cache_manager):
    assert cache_manager.set("k1", {"value": 1}, ttl=10) is True
    assert cache_manager.get("k1") == {"value": 1}
    assert cache_manager.exists("k1") is True

    stats = cache_manager.get_stats()
    assert stats["cache_size"] == 1
    assert stats["total_requests"] == 1  # one hit


def test_ttl_expiration_removes_entry(cache_manager):
    cache_manager.set("k_expire", {"value": 2}, ttl=1)
    entry = cache_manager._cache["k_expire"]
    entry.created_at -= 10  # simulate old entry

    assert cache_manager.get("k_expire") is None
    assert cache_manager.exists("k_expire") is False


def test_eviction_on_max_size(cache_manager):
    cache_manager.set("a", 1)
    cache_manager.set("b", 2)
    cache_manager.set("c", 3)  # should evict least recently used "a"

    keys = cache_manager.list_keys()
    assert len(keys) == 2
    assert "a" not in keys


def test_clear_removes_all_entries(cache_manager):
    cache_manager.set("x", 1)
    cache_manager.set("y", 2)
    assert cache_manager.clear() is True
    assert cache_manager.list_keys() == []


def test_health_check_success(cache_manager):
    result = cache_manager.health_check()
    assert result["status"] == "healthy"


def test_strategy_hooks_invoked(monkeypatch):
    calls = []

    class DummyStrategy:
        def on_set(self, cache, key, entry, config):
            calls.append(("set", key))

        def on_get(self, cache, key, entry, config):
            calls.append(("get", key))

        def on_evict(self, cache, config):
            return None

    config = CacheConfig(max_size=1, enable_disk_cache=False, enable_stats=False)
    manager = CacheManager(config, strategy=DummyStrategy())
    try:
        manager.set("key1", "v1")
        manager.get("key1")
    finally:
        manager.stop()

    assert ("set", "key1") in calls
    assert ("get", "key1") in calls

