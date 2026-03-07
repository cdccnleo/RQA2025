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


import os
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from src.data.cache.multi_level_cache import CacheConfig, MultiLevelCache


class DummyRedisAdapter:
    def __init__(self):
        self.storage = {}

    def get(self, key: str) -> Any:
        return self.storage.get(key)

    def set(self, key: str, value: Any, ttl: int):
        self.storage[key] = value

    def delete(self, key: str):
        self.storage.pop(key, None)

    def clear(self):
        self.storage.clear()

    def get_stats(self):
        return {"size": len(self.storage)}


@pytest.fixture
def temp_cache_dir(tmp_path: Path):
    cache_dir = tmp_path / "multi_cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def cache_without_redis(temp_cache_dir):
    config = CacheConfig(
        memory_max_size=2,
        memory_ttl=1,
        disk_enabled=True,
        disk_cache_dir=str(temp_cache_dir),
        disk_ttl=60,
        redis_enabled=False,
    )
    return MultiLevelCache(config)


def test_multi_level_cache_basic_flow(cache_without_redis, tmp_path):
    cache = cache_without_redis

    assert cache.get("missing") is None

    assert cache.set("key", {"value": 1})
    assert cache.get("key") == {"value": 1}

    assert cache.delete("key") is True
    assert cache.get("key") is None

    assert cache.set("key2", {"value": 2})
    stats = cache.get_stats()
    assert stats["memory_cache"]["size"] <= 2
    assert stats["disk_cache"]["enabled"] is True
    assert stats["redis_cache"]["enabled"] is False

    cleaned = cache.clean_expired()
    assert cleaned >= 0

    assert cache.clear() is True
    cleared_stats = cache.get_stats()
    assert cleared_stats["performance"]["total_requests"] >= 0


def test_multi_level_cache_disk_interaction(cache_without_redis, temp_cache_dir):
    cache = cache_without_redis

    payload = {"rows": [1, 2, 3]}
    assert cache.set("disk_key", payload)

    cache._set_to_memory("disk_key", payload, ttl=0)
    assert cache._get_from_memory("disk_key") is None

    disk_value = cache._get_from_disk("disk_key")
    assert disk_value == payload

    cache.delete("disk_key")
    assert cache._get_from_disk("disk_key") is None


def test_multi_level_cache_with_redis(monkeypatch, temp_cache_dir):
    fake_redis = DummyRedisAdapter()
    monkeypatch.setattr("src.data.cache.multi_level_cache.RedisCacheAdapter", lambda cfg: fake_redis)

    config = CacheConfig(
        memory_max_size=1,
        memory_ttl=60,
        disk_enabled=False,
        redis_enabled=True,
        redis_ttl=120,
    )
    cache = MultiLevelCache(config)
    assert cache.set("redis_key", {"order": 1})
    assert cache.get("redis_key") == {"order": 1}
    assert cache.get_stats()["redis_cache"]["enabled"] is True


def test_clean_expired_handles_disk_and_memory(cache_without_redis):
    cache = cache_without_redis
    cache.set("k_old", {"v": 1}, ttl=0)
    cleaned = cache.clean_expired()
    assert cleaned >= 1


def test_cache_handles_dataframe(cache_without_redis):
    cache = cache_without_redis
    df = pd.DataFrame({"col": [1, 2, 3]})
    cache.set("df_key", df)
    loaded = cache.get("df_key")
    assert isinstance(loaded, pd.DataFrame)
    assert loaded.equals(df)


def test_delete_removes_memory_disk_and_redis(monkeypatch, temp_cache_dir):
    fake_redis = DummyRedisAdapter()
    monkeypatch.setattr("src.data.cache.multi_level_cache.RedisCacheAdapter", lambda cfg: fake_redis)

    config = CacheConfig(
        redis_enabled=True,
        disk_enabled=True,
        disk_cache_dir=str(temp_cache_dir),
    )
    cache = MultiLevelCache(config)
    cache.set("del_key", {"value": 1}, ttl=60)
    assert cache.delete("del_key") is True
    assert cache.get("del_key") is None
    assert fake_redis.storage == {}
    assert not list(temp_cache_dir.glob("*.pkl"))


def test_clear_removes_all_layers(cache_without_redis, temp_cache_dir):
    cache = cache_without_redis
    cache.set("c1", {"value": 1})
    cache.set("c2", {"value": 2})
    cache.clear()
    assert cache.memory_cache == {}
    assert list(temp_cache_dir.glob("*.pkl")) == []


def test_get_promotes_from_redis(monkeypatch, temp_cache_dir):
    fake_redis = DummyRedisAdapter()
    monkeypatch.setattr("src.data.cache.multi_level_cache.RedisCacheAdapter", lambda cfg: fake_redis)
    config = CacheConfig(
        redis_enabled=True,
        disk_enabled=True,
        disk_cache_dir=str(temp_cache_dir),
    )
    cache = MultiLevelCache(config)
    fake_redis.storage["redis_key"] = 42

    result = cache.get("redis_key")
    assert result == 42
    assert cache.memory_cache["redis_key"] == 42
    assert list(temp_cache_dir.glob("*.pkl"))  # promoted to disk as well


def test_init_redis_cache_handles_failure(monkeypatch, temp_cache_dir):
    class FailingRedis:
        def __init__(self, cfg):
            raise RuntimeError("boom")

    monkeypatch.setattr("src.data.cache.multi_level_cache.RedisCacheAdapter", FailingRedis)
    config = CacheConfig(redis_enabled=True, disk_cache_dir=str(temp_cache_dir))
    cache = MultiLevelCache(config)
    assert cache.redis_cache is None


def test_memory_eviction_lru_path(temp_cache_dir):
    config = CacheConfig(
        memory_max_size=1,
        disk_enabled=False,
        redis_enabled=False,
    )
    cache = MultiLevelCache(config)
    cache.set("first", {"v": 1})
    cache.set("second", {"v": 2})
    assert "first" not in cache.memory_cache
    assert cache.get("second") == {"v": 2}


def test_set_to_disk_with_nonpositive_ttl_removes_file(temp_cache_dir):
    config = CacheConfig(disk_enabled=True, disk_cache_dir=str(temp_cache_dir))
    cache = MultiLevelCache(config)
    cache._set_to_disk("temp", {"v": 1}, ttl=10)
    cache_file = temp_cache_dir / "temp.pkl"
    assert cache_file.exists()
    cache._set_to_disk("temp", {"v": 1}, ttl=0)
    assert not cache_file.exists()


def test_get_from_disk_respects_size_limit(temp_cache_dir):
    config = CacheConfig(disk_enabled=True, disk_cache_dir=str(temp_cache_dir))
    cache = MultiLevelCache(config)
    cache._set_to_disk("huge", {"v": "x" * 1024}, ttl=60)
    cache.config.disk_max_size_mb = 0  # force oversize
    assert cache._get_from_disk("huge") is None
    assert not (temp_cache_dir / "huge.pkl").exists()


def test_set_to_redis_handles_negative_ttl(monkeypatch, temp_cache_dir):
    class DummyRedis:
        def __init__(self):
            self.deleted = []
            self.set_calls = []

        def delete(self, key):
            self.deleted.append(key)

        def set(self, key, value, ttl):
            self.set_calls.append((key, value, ttl))

    dummy = DummyRedis()
    config = CacheConfig(redis_enabled=False, disk_cache_dir=str(temp_cache_dir))
    cache = MultiLevelCache(config)
    cache.redis_cache = dummy
    cache._set_to_redis("k", {"v": 1}, ttl=0)
    assert dummy.deleted == ["k"]
    cache._set_to_redis("k2", {"v": 2}, ttl=10)
    assert dummy.set_calls == [("k2", {"v": 2}, 10)]


def test_cleanup_resets_disk_and_redis(temp_cache_dir):
    class DummyRedis:
        def __init__(self):
            self.cleared = False

        def clear(self):
            self.cleared = True

    config = CacheConfig(disk_enabled=True, disk_cache_dir=str(temp_cache_dir))
    cache = MultiLevelCache(config)
    cache.redis_cache = DummyRedis()

    cache._set_to_disk("temp", {"v": 1}, ttl=10)
    assert any(temp_cache_dir.glob("*.pkl"))

    cache.cleanup()

    assert cache.redis_cache.cleared is True
    assert list(temp_cache_dir.glob("*.pkl")) == []
    assert temp_cache_dir.exists()

