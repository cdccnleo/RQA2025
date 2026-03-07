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

import numpy as np
import pandas as pd
import pytest

import src.data.cache.enhanced_cache_manager as manager_module
from src.data.cache.enhanced_cache_manager import EnhancedCacheManager


@pytest.fixture
def cache_manager(tmp_path, monkeypatch):

    class DummyThread:

        def __init__(self, target=None, daemon=None):

            self.target = target
            self.daemon = daemon
            self._alive = False

        def start(self):

            self._alive = True

        def join(self, timeout=None):

            self._alive = False

        def is_alive(self):

            return self._alive

    monkeypatch.setattr(manager_module.threading, "Thread", DummyThread)

    cache_dir = tmp_path / "cache"
    manager = EnhancedCacheManager(
        cache_dir=str(cache_dir),
        max_memory_size=8 * 1024,
        max_disk_size=64 * 1024,
    )
    yield manager
    manager.shutdown()


def test_set_get_promotes_and_updates_stats(cache_manager):

    assert cache_manager.set("alpha", {"price": 1}, prefix="market", expire=10) is True
    assert cache_manager.get("alpha", "market") == {"price": 1}

    cache_key = cache_manager._generate_cache_key("alpha", "market")
    cache_item = cache_manager.memory_cache.pop(cache_key)
    cache_manager.memory_size -= cache_item["size"]

    disk_file = os.path.join(cache_manager.disk_cache_dir, f"{cache_key}.pkl")
    with open(disk_file, "wb") as fh:
        pickle.dump(cache_item, fh)

    value = cache_manager.get("alpha", "market")
    assert value == {"price": 1}

    stats = cache_manager.get_stats()
    assert stats["memory_hits"] >= 1
    assert stats["disk_hits"] >= 1
    assert stats["total_operations"] >= 3


def test_set_validates_inputs(cache_manager):

    with pytest.raises(ValueError):
        cache_manager.set("", "value")

    with pytest.raises(ValueError):
        cache_manager.set("key", None)

    with pytest.raises(ValueError):
        cache_manager.set("key", "value", expire=-1)


def test_clear_prefix_and_all(cache_manager):

    cache_manager.set("k1", "v1", prefix="group")
    cache_manager.set("k2", "v2", prefix="group")
    cache_manager.set("k3", "v3")

    cache_manager.clear(prefix="group")
    assert cache_manager.prefix_index.get("group") is None

    cache_manager.clear()
    assert cache_manager.memory_size == 0
    assert cache_manager.get("k3") is None


def test_cleanup_helpers_remove_stale_entries(cache_manager):

    stale_key = cache_manager._generate_cache_key("stale", "")
    cache_manager.memory_cache[stale_key] = {
        "value": "x",
        "expire_time": time.time() - 5,
        "size": 1,
        "access_count": 0,
        "created_time": time.time() - 10,
    }
    cache_manager.memory_size += 1

    cache_manager._cleanup_memory_cache()
    assert stale_key not in cache_manager.memory_cache

    disk_file = os.path.join(cache_manager.disk_cache_dir, f"{stale_key}.pkl")
    with open(disk_file, "wb") as fh:
        pickle.dump(
            {
                "value": "disk",
                "expire_time": time.time() - 5,
                "size": 1,
                "access_count": 0,
            },
            fh,
        )

    cache_manager._cleanup_disk_cache()
    assert not os.path.exists(disk_file)


def test_shutdown_resets_stats(cache_manager):

    cache_manager.set("temp", "value")
    assert cache_manager.stats["total_operations"] > 0

    cache_manager.shutdown()
    assert cache_manager.stats["total_operations"] == 0


def test_set_handles_storage_failures(cache_manager, monkeypatch):

    monkeypatch.setattr(cache_manager, "_try_memory_cache", lambda *a, **k: False)
    monkeypatch.setattr(cache_manager, "_try_disk_cache", lambda *a, **k: False)
    assert cache_manager.set("fail", "value") is False


def test_set_handles_internal_exception(cache_manager, monkeypatch):

    def boom(*_):
        raise RuntimeError("boom")

    monkeypatch.setattr(cache_manager, "_generate_cache_key", boom)
    assert cache_manager.set("err", "value") is False


def test_try_memory_cache_respects_limits(cache_manager, monkeypatch):

    cache_manager.max_memory_size = 1
    cache_manager.memory_size = 1
    monkeypatch.setattr(cache_manager, "_cleanup_memory_cache", lambda: None)

    result = cache_manager._try_memory_cache("heavy", {"size": 1})
    assert result is False


def test_try_disk_cache_cleanup_path(cache_manager, monkeypatch):

    cleanup_called = []

    def fake_size():
        return cache_manager.max_disk_size

    def fake_cleanup():
        cleanup_called.append(True)

    monkeypatch.setattr(cache_manager, "_get_disk_cache_size", fake_size)
    monkeypatch.setattr(cache_manager, "_cleanup_disk_cache", fake_cleanup)
    assert cache_manager._try_disk_cache("disk", {"size": 10}) is False
    assert cleanup_called


def test_get_removes_expired_memory_entries(cache_manager):

    cache_key = cache_manager._generate_cache_key("stale_mem", "")
    cache_manager.memory_cache[cache_key] = {
        "value": "x",
        "expire_time": time.time() - 1,
        "size": 1,
        "access_count": 0,
    }
    cache_manager.memory_size += 1
    assert cache_manager.get("stale_mem") is None
    assert cache_key not in cache_manager.memory_cache


def test_clear_supports_direct_cache_key(cache_manager):

    cache_manager.set("direct_key", "value")
    cache_manager.clear(prefix="direct_key")
    assert cache_manager.get("direct_key") is None


def test_get_stats_reports_zero_hit_rate(cache_manager):

    stats = cache_manager.get_stats()
    assert stats["hit_rate"] == 0.0
    assert stats["memory_hit_rate"] == 0.0
    assert stats["disk_hit_rate"] == 0.0


def test_cleanup_thread_worker_executes_once(cache_manager, monkeypatch):

    calls = []

    def fake_sleep(*_):
        return None

    def fake_cleanup_memory():
        calls.append("memory")
        cache_manager._stop_cleanup = True

    def fake_cleanup_disk():
        calls.append("disk")

    monkeypatch.setattr(manager_module.time, "sleep", fake_sleep)
    monkeypatch.setattr(cache_manager, "_cleanup_memory_cache", fake_cleanup_memory)
    monkeypatch.setattr(cache_manager, "_cleanup_disk_cache", fake_cleanup_disk)

    cache_manager._stop_cleanup = False
    cache_manager._cleanup_thread.target()
    assert calls == ["memory", "disk"]


def test_get_memory_size_supports_multiple_types(cache_manager):

    frame = pd.DataFrame({"v": [1, 2, 3]})
    assert cache_manager._get_memory_size(frame) >= frame.memory_usage(deep=True).sum()

    array = np.arange(5)
    assert cache_manager._get_memory_size(array) == array.nbytes

    mapping = {"values": [1, 2]}
    assert cache_manager._get_memory_size(mapping) >= len(pickle.dumps(mapping))

    unpicklable = {"callable": lambda x: x}
    assert cache_manager._get_memory_size(unpicklable) == len("callable")

    class BadStr:

        def __str__(self):

            raise RuntimeError("boom")

    assert cache_manager._get_memory_size(BadStr()) == 1024


def test_try_memory_cache_cleanup_allows_store(cache_manager):

    cache_manager.max_memory_size = 8
    stale_key = cache_manager._generate_cache_key("old", "")
    cache_manager.memory_cache[stale_key] = {
        "value": "stale",
        "expire_time": time.time() - 1,
        "size": 6,
        "access_count": 0,
    }
    cache_manager.memory_size = 6

    cache_item = {"size": 4, "expire_time": time.time() + 10}
    assert cache_manager._try_memory_cache("fresh", cache_item) is True


def test_try_disk_cache_cleanup_allows_store(cache_manager):

    cache_manager.max_disk_size = 256
    stale_key = cache_manager._generate_cache_key("disk_old", "")
    stale_file = os.path.join(cache_manager.disk_cache_dir, f"{stale_key}.pkl")
    with open(stale_file, "wb") as fh:
        pickle.dump(
            {
                "value": "disk",
                "expire_time": time.time() - 5,
                "size": 200,
                "access_count": 0,
            },
            fh,
        )

    cache_item = {"size": 32, "expire_time": time.time() + 5}
    assert cache_manager._try_disk_cache("disk_new", cache_item) is True


def test_cleanup_memory_cache_evicts_low_priority(cache_manager):

    cache_manager.max_memory_size = 10
    for idx in range(3):
        cache_key = cache_manager._generate_cache_key(f"obj{idx}", "")
        cache_manager.memory_cache[cache_key] = {
            "value": idx,
            "expire_time": time.time() + 100,
            "size": 5,
            "access_count": idx,
        }
        cache_manager.memory_size += 5

    cache_manager._cleanup_memory_cache()
    assert cache_manager.memory_size <= cache_manager.max_memory_size * 0.7

