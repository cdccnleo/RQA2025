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

import pytest

import src.data.cache.multi_level_cache as multi_level_module
from src.data.cache.multi_level_cache import CacheConfig, MultiLevelCache


@pytest.fixture
def cache_manager(tmp_path):

    config = CacheConfig(
        memory_max_size=2,
        memory_ttl=5,
        disk_enabled=True,
        disk_cache_dir=str(tmp_path / "mlc"),
        disk_ttl=1,
        redis_enabled=False,
    )
    manager = MultiLevelCache(config)
    yield manager
    manager.cleanup()


def test_get_hits_memory_then_disk(cache_manager):

    assert cache_manager.set("alpha", {"price": 1})
    assert cache_manager.get("alpha") == {"price": 1}

    cache_manager.memory_cache.clear()
    cache_manager.memory_timestamps.clear()
    cache_manager.memory_access_count.clear()

    assert cache_manager.get("alpha") == {"price": 1}
    assert cache_manager.stats["disk_hits"] == 1


def test_memory_lru_eviction(cache_manager):

    cache_manager.set("k1", "v1")
    cache_manager.set("k2", "v2")
    cache_manager.get("k1")
    cache_manager.set("k3", "v3")

    assert "k2" not in cache_manager.memory_cache
    assert "k1" in cache_manager.memory_cache


def test_delete_removes_all_levels(cache_manager):

    cache_manager.set("temp", "value")
    assert cache_manager.delete("temp") is True
    assert cache_manager.get("temp") is None

    disk_file = Path(cache_manager.disk_cache_dir) / "temp.pkl"
    assert not disk_file.exists()


def test_clean_expired_counts_memory_and_disk(cache_manager):

    cache_manager.set("stale", "value")
    cache_manager.memory_timestamps["stale"] = time.time() - 10

    disk_file = Path(cache_manager.disk_cache_dir) / "stale.pkl"
    os.utime(disk_file, (time.time() - 3600, time.time() - 3600))

    cleaned = cache_manager.clean_expired()
    assert cleaned >= 2


def test_set_with_non_positive_ttl_expires_immediately(cache_manager):

    cache_manager.set("beta", "value", ttl=0)
    assert cache_manager.get("beta") is None

    disk_file = Path(cache_manager.disk_cache_dir) / "beta.pkl"
    assert not disk_file.exists()


def test_clear_resets_stats(cache_manager):

    cache_manager.set("x", 1)
    cache_manager.get("x")
    stats_before = cache_manager.get_stats()
    assert stats_before["performance"]["total_requests"] > 0

    assert cache_manager.clear() is True
    stats_after = cache_manager.get_stats()
    assert stats_after["performance"]["total_requests"] == 0
    assert stats_after["memory_cache"]["size"] == 0


def test_disk_disabled_skips_writes(tmp_path):

    config = CacheConfig(
        disk_enabled=False,
        redis_enabled=False,
        memory_max_size=1,
    )
    cache = MultiLevelCache(config)
    try:
        assert cache.disk_cache_dir is None
        cache.set("k", "v")
        assert cache.get("k") == "v"
    finally:
        cache.cleanup()


def test_get_from_disk_handles_large_and_corrupted_files(tmp_path):

    config = CacheConfig(
        disk_enabled=True,
        disk_cache_dir=str(tmp_path / "disk-large"),
        disk_ttl=10,
        disk_max_size_mb=0,
    )
    cache = MultiLevelCache(config)
    try:
        cache.set("huge", "v")
        cache_file = Path(cache.disk_cache_dir) / "huge.pkl"
        assert cache._get_from_disk("huge") is None
        assert not cache_file.exists()

        bad_config = CacheConfig(
            disk_enabled=True,
            disk_cache_dir=str(tmp_path / "disk-bad"),
            disk_ttl=10,
        )
        bad_cache = MultiLevelCache(bad_config)
        try:
            bad_file = Path(bad_cache.disk_cache_dir) / "bad.pkl"
            with open(bad_file, "wb") as fh:
                fh.write(b"not-pickle")
            assert bad_cache._get_from_disk("bad") is None
            assert not bad_file.exists()
        finally:
            bad_cache.cleanup()
    finally:
        cache.cleanup()


def test_redis_cache_flow_and_cleanup(monkeypatch, tmp_path):

    class DummyRedis:

        def __init__(self, config):

            self.store = {}

        def get(self, key):

            return self.store.get(key)

        def set(self, key, value, ttl):

            self.store[key] = value

        def delete(self, key):

            self.store.pop(key, None)

        def clear(self):

            self.store.clear()

        def get_stats(self):

            return {"entries": len(self.store)}

    monkeypatch.setattr(multi_level_module, "RedisCacheAdapter", DummyRedis)

    config = CacheConfig(
        disk_enabled=False,
        redis_enabled=True,
        redis_config={},
        memory_max_size=1,
        memory_ttl=1,
        redis_ttl=5,
    )
    cache = MultiLevelCache(config)
    try:
        cache.set("redis_key", "redis_value")
        cache.memory_cache.clear()
        cache.memory_timestamps.clear()
        cache.memory_access_count.clear()

        assert cache.get("redis_key") == "redis_value"
        assert cache.stats["redis_hits"] == 1
        stats = cache.get_stats()
        assert stats["redis_cache"]["entries"] == 1

        cache.set("redis_key", "redis_value", ttl=0)
        assert cache.get("redis_key") is None
    finally:
        cache.cleanup()


def test_redis_initialization_failure(monkeypatch):

    def boom(config):
        raise RuntimeError("boom")

    monkeypatch.setattr(multi_level_module, "RedisCacheAdapter", boom)

    config = CacheConfig(redis_enabled=True, redis_config={})
    cache = MultiLevelCache(config)
    try:
        assert cache.redis_cache is None
    finally:
        cache.cleanup()

