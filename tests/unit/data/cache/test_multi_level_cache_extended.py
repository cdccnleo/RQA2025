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
import time
from pathlib import Path

import pandas as pd
import pytest

from src.data.cache.multi_level_cache import CacheConfig, MultiLevelCache


class SpyRedisAdapter:
    def __init__(self):
        self.storage = {}
        self.deleted_keys = []
        self.clear_called = False

    def get(self, key):
        return self.storage.get(key)

    def set(self, key, value, ttl):
        self.storage[key] = (value, ttl)

    def delete(self, key):
        self.deleted_keys.append(key)
        self.storage.pop(key, None)

    def get_stats(self):
        return {"connected": True, "size": len(self.storage)}

    def clear(self):
        self.clear_called = True
        self.storage.clear()


@pytest.fixture
def temp_dir(tmp_path: Path):
    path = tmp_path / "mlc"
    path.mkdir()
    return path


def _create_cache(temp_dir: Path, **overrides) -> MultiLevelCache:
    params = {
        "memory_max_size": 2,
        "memory_ttl": 5,
        "disk_enabled": True,
        "disk_cache_dir": str(temp_dir),
        "disk_ttl": 1,
        "disk_max_size_mb": 1,
        "redis_enabled": False,
    }
    params.update(overrides)
    return MultiLevelCache(CacheConfig(**params))


def test_lru_eviction_prefers_least_frequently_used(temp_dir):
    cache = _create_cache(temp_dir)
    cache._set_to_memory("a", 1, ttl=10)
    cache._set_to_memory("b", 2, ttl=10)
    _ = cache._get_from_memory("a")
    cache._set_to_memory("c", 3, ttl=10)

    assert "b" not in cache.memory_cache
    assert "a" in cache.memory_cache
    assert "c" in cache.memory_cache


def test_get_from_disk_handles_size_limit_and_corruption(temp_dir):
    cache = _create_cache(temp_dir, disk_max_size_mb=0)
    cache.set("oversize", {"payload": "data"})

    assert cache._get_from_disk("oversize") is None
    assert not (Path(cache.disk_cache_dir) / "oversize.pkl").exists()

    corrupt_file = Path(cache.disk_cache_dir) / "corrupt.pkl"
    with open(corrupt_file, "wb") as file:
        file.write(b"not-a-pickle")

    assert cache._get_from_disk("corrupt") is None
    assert not corrupt_file.exists()


def test_clean_expired_removes_old_disk_entries(temp_dir):
    cache = _create_cache(temp_dir, disk_ttl=0)
    cache._set_to_disk("old", {"val": 1}, ttl=60)
    cache_file = Path(cache.disk_cache_dir) / "old.pkl"
    os.utime(cache_file, (time.time() - 10, time.time() - 10))

    removed = cache.clean_expired()
    assert removed >= 1
    assert not cache_file.exists()


def test_redis_paths_cover_stats_and_ttl(monkeypatch, temp_dir):
    spy = SpyRedisAdapter()
    monkeypatch.setattr(
        "src.data.cache.multi_level_cache.RedisCacheAdapter", lambda config: spy
    )

    cache = _create_cache(
        temp_dir, redis_enabled=True, redis_config={}, redis_ttl=5, disk_enabled=False
    )

    cache.set("redis_key", {"value": 1})
    cache._set_to_redis("redis_key", {"value": 2}, ttl=0)

    assert "redis_key" in spy.deleted_keys

    stats = cache.get_stats()
    assert stats["redis_cache"]["connected"] is True
    assert stats["redis_cache"]["size"] == 0


def test_cleanup_resets_all_layers(monkeypatch, temp_dir):
    spy = SpyRedisAdapter()
    monkeypatch.setattr(
        "src.data.cache.multi_level_cache.RedisCacheAdapter", lambda config: spy
    )

    cache = _create_cache(
        temp_dir, redis_enabled=True, redis_config={}, disk_enabled=True, disk_ttl=60
    )
    cache.set("key", pd.DataFrame({"c": [1, 2]}))

    dir_before = Path(cache.disk_cache_dir)
    cache.cleanup()

    assert dir_before.exists()
    assert list(dir_before.glob("*.pkl")) == []
    assert spy.clear_called is True
    assert cache.memory_cache == {}


def test_get_from_disk_respects_zero_ttl(temp_dir):
    cache = _create_cache(temp_dir)
    cache._set_to_disk("temp", {"x": 1}, ttl=0)
    assert not (Path(cache.disk_cache_dir) / "temp.pkl").exists()


