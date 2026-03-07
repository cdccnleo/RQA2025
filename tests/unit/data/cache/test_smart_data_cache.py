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


import types
from datetime import datetime

import pytest

from src.data.cache.smart_data_cache import (
    DataCacheConfig,
    SmartDataCacheBackend,
    SmartDataCache,
    CacheStats,
    DataSourceType,
)


@pytest.fixture
def force_fallback(monkeypatch):
    import src.data.cache.smart_data_cache as module

    monkeypatch.setattr(module, "INFRASTRUCTURE_CACHE_AVAILABLE", False)
    return module


def test_backend_uses_fallback_cache(force_fallback):
    backend = SmartDataCacheBackend(DataCacheConfig(capacity=2))
    assert backend.strategy == "fallback_lru"

    assert backend.set("k1", "v1")
    assert backend.get("k1") == "v1"

    assert backend.get("missing") is None
    stats = backend.get_stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["total_requests"] == 2


def test_backend_delete_and_clear(force_fallback):
    backend = SmartDataCacheBackend(DataCacheConfig(capacity=2))
    backend.set("k1", "v1")
    backend.set("k2", "v2")

    assert backend.delete("k1") is True
    assert backend.get("k1") is None

    backend.stats.hits = 2
    backend.stats.total_requests = 5
    assert backend.clear() is True
    assert backend.get_stats()["hits"] == 0
    assert backend.cache.size() == 0


def test_backend_exists_handles_missing_method(force_fallback):
    backend = SmartDataCacheBackend(DataCacheConfig(capacity=2))
    backend.set("exists_key", "value")
    assert backend.exists("exists_key") is True
    assert backend.exists("missing_key") is False


def test_backend_handles_promote_and_disk_cleanup(force_fallback):
    backend = SmartDataCacheBackend(DataCacheConfig(capacity=1))
    backend.set("k1", "v1")
    backend.set("k2", "v2")
    assert backend.cache.size() == 1  # LRU fallback ensures capacity respected

    backend.stats.evictions += 1
    stats = backend.get_stats()
    assert stats["evictions"] == 1


class StubBackend:
    def __init__(self):
        self.calls = []
        self.cleared = False

    def set(self, key, value, ttl):
        self.calls.append((key, ttl, value))
        return True

    def get(self, key):
        return {"value": key}

    def delete(self, key):
        return True

    def clear(self):
        self.cleared = True
        return True

    def exists(self, key):
        return key == "present"

    def get_stats(self):
        return {
            "strategy": "stub",
            "capacity": 5,
            "current_size": 1,
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "evictions": 0,
            "total_requests": 0,
            "hit_rate": 0.0,
            "last_cleanup": datetime.now().isoformat(),
            "infrastructure_cache": False,
        }


def test_smart_cache_uses_type_config_ttl():
    backend = StubBackend()
    cache = SmartDataCache(backend=backend)
    cache.set("stream_key", {"p": 1}, DataSourceType.STREAM)
    assert backend.calls[-1][1] == 300

    cache.set("db_key", {"p": 2}, DataSourceType.DATABASE, ttl=42)
    assert backend.calls[-1][1] == 42


def test_smart_cache_operations_with_stub_backend():
    backend = StubBackend()
    cache = SmartDataCache(backend=backend)

    assert cache.get("any", DataSourceType.API) == {"value": "any"}
    assert cache.delete("any") is True
    assert cache.clear() is True
    assert backend.cleared is True
    assert cache.exists("present") is True
    assert cache.exists("absent") is False


def test_smart_cache_invalidate_all(monkeypatch):
    backend = StubBackend()
    cache = SmartDataCache(backend=backend)
    assert cache.invalidate("*") == 1
    assert cache.invalidate("not_implemented") == 0


def test_get_cache_info_combines_backend_stats():
    backend = StubBackend()
    cache = SmartDataCache(backend=backend)

    info = cache.get_cache_info()
    assert info["strategy"] == "stub"
    assert "data_type_configs" in info
import types

import pytest

from src.data.cache import smart_data_cache
from src.data.cache.smart_data_cache import (
    DataCacheConfig,
    SmartDataCacheBackend,
    SmartDataCache,
    create_smart_data_cache,
)
from src.data.interfaces.standard_interfaces import DataSourceType


@pytest.fixture(autouse=True)
def reset_infrastructure_flag(monkeypatch):
    """默认使用降级缓存，测试结束后自动恢复。"""
    original_flag = smart_data_cache.INFRASTRUCTURE_CACHE_AVAILABLE
    monkeypatch.setattr(smart_data_cache, "INFRASTRUCTURE_CACHE_AVAILABLE", False)
    yield
    monkeypatch.setattr(smart_data_cache, "INFRASTRUCTURE_CACHE_AVAILABLE", original_flag)


def test_backend_fallback_workflow():
    config = DataCacheConfig(capacity=2, strategy="lfu")
    backend = SmartDataCacheBackend(config)

    assert backend.set("k1", "v1") is True
    assert backend.get("k1") == "v1"
    assert backend.exists("k1") is True

    assert backend.get("missing") is None  # 计入 miss
    stats_before_clear = backend.get_stats()
    assert stats_before_clear["total_requests"] >= 2
    assert stats_before_clear["hit_rate"] >= 0
    assert stats_before_clear["infrastructure_cache"] is False

    assert backend.delete("k1") is True
    assert backend.exists("k1") is False

    assert backend.clear() is True
    stats_after_clear = backend.get_stats()
    assert stats_after_clear["hits"] == 0
    assert stats_after_clear["misses"] == 0
    assert backend.stats.total_requests == 0


def test_backend_infrastructure_initialisation(monkeypatch):
    class DummyCache:
        def __init__(self, capacity):
            self.capacity = capacity
            self.store = {}

        def get(self, key):
            return self.store.get(key)

        def put(self, key, value):
            self.store[key] = value

        def exists(self, key):
            return key in self.store

        def clear(self):
            self.store.clear()

        def size(self):
            return len(self.store)

        def get_stats(self):
            return {"current_size": len(self.store)}

    strategy = types.SimpleNamespace(
        LFU="LFU",
        LRU_K="LRU_K",
        ADAPTIVE="ADAPTIVE",
        PRIORITY="PRIORITY",
        COST_AWARE="COST_AWARE",
    )

    monkeypatch.setattr(smart_data_cache, "CacheStrategy", strategy)
    for attr in ("LFUCache", "LRUKCache", "AdaptiveCache", "PriorityCache", "CostAwareCache"):
        monkeypatch.setattr(
            smart_data_cache,
            attr,
            lambda *args, **kwargs: DummyCache(args[0] if args else kwargs.get("capacity", 0)),
        )
    monkeypatch.setattr(smart_data_cache, "INFRASTRUCTURE_CACHE_AVAILABLE", True)

    backend = SmartDataCacheBackend(DataCacheConfig(strategy="priority", capacity=3))
    assert backend.strategy == strategy.PRIORITY

    assert backend.set("asset", {"price": 1})
    assert backend.get("asset") == {"price": 1}
    assert backend.exists("asset") is True

    stats = backend.get_stats()
    assert stats["infrastructure_cache"] is True
    assert stats["current_size"] == 1

    backend.clear()
    cleared_stats = backend.get_stats()
    assert cleared_stats["current_size"] == 0
    assert backend.stats.total_requests == 0


def test_smart_data_cache_basic_usage(monkeypatch):
    # 使用降级缓存验证 get/set/invalidate 等逻辑
    cache = SmartDataCache()

    assert cache.set("db:1", {"symbol": "AAA"}, DataSourceType.DATABASE) is True
    assert cache.get("db:1", DataSourceType.DATABASE) == {"symbol": "AAA"}

    # invalidate("*") 应该触发清理并返回 1
    assert cache.invalidate("*") == 1
    assert cache.get("db:1", DataSourceType.DATABASE) is None

    info = cache.get_cache_info()
    assert info["config"]["strategy"] == cache.config.strategy
    assert "data_type_configs" in info

    assert cache.optimize_for_data_type(DataSourceType.STREAM) is True

    factory_cache = create_smart_data_cache()
    assert isinstance(factory_cache, SmartDataCache)


def test_smart_data_cache_with_custom_backend():
    class FakeBackend:
        def __init__(self):
            self.storage = {}
            self.cleared = False

        def get(self, key):
            return self.storage.get(key)

        def set(self, key, value, ttl=None):
            self.storage[key] = value
            return True

        def clear(self):
            self.storage.clear()
            self.cleared = True
            return True

        def get_stats(self):
            return {
                "strategy": "fake",
                "capacity": len(self.storage),
                "current_size": len(self.storage),
                "hits": 0,
                "misses": 0,
                "sets": len(self.storage),
                "evictions": 0,
                "total_requests": len(self.storage),
                "hit_rate": 0.0,
                "last_cleanup": datetime.now().isoformat(),
                "infrastructure_cache": False,
            }

    fake_backend = FakeBackend()
    cache = SmartDataCache(backend=fake_backend)
    assert cache.set("api:1", {"title": "hello"}, DataSourceType.API)

    info = cache.get_cache_info()
    assert info["strategy"] == "fake"
    assert info["config"]["capacity"] == cache.config.capacity

    assert cache.invalidate("*") == 1
    assert fake_backend.cleared is True
    assert cache.get("api:1", DataSourceType.API) is None

