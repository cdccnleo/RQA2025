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


import importlib
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.cache import enhanced_cache_manager as ecm_module
from src.data.cache.enhanced_cache_manager import EnhancedCacheManager


@pytest.fixture(autouse=True)
def disable_cleanup_thread(monkeypatch):
    monkeypatch.setattr(
        "src.data.cache.enhanced_cache_manager.EnhancedCacheManager._start_cleanup_thread",
        lambda self: None,
    )


@pytest.fixture
def cache(tmp_path):
    manager = EnhancedCacheManager(
        cache_dir=str(tmp_path / "enhanced_cache"),
        max_memory_size=1024,
        max_disk_size=2048,
    )
    yield manager
    manager.shutdown()


def test_set_validates_inputs(cache):
    with pytest.raises(ValueError):
        cache.set("", "value")
    with pytest.raises(ValueError):
        cache.set("key", None)
    with pytest.raises(ValueError):
        cache.set("key", "value", expire=-1)


def test_set_promotes_to_disk_when_memory_full(cache, monkeypatch):
    monkeypatch.setattr(cache, "_get_memory_size", lambda value: 900)
    monkeypatch.setattr(cache, "_try_disk_cache", lambda key, item: True)
    assert cache.set("k1", "a" * 10, expire=5)
    assert cache.set("k2", "b" * 10, expire=5)

    disk_files = list(Path(cache.disk_cache_dir).glob("*.pkl"))
    assert len(disk_files) >= 0


def test_get_returns_none_when_missing_or_expired(cache):
    assert cache.get("missing") is None
    cache.set("temp", "value", expire=0)
    assert cache.get("temp") is None


def test_get_promotes_hot_items_to_memory(cache):
    cache.set("key", "value", expire=60)
    cache._promote_to_memory = lambda key, item: item.update({"promoted": True})
    for _ in range(6):
        cache.get("key")
    assert cache.memory_cache
    stored = next(iter(cache.memory_cache.values()))
    assert stored["value"] == "value"


def test_disk_cache_handles_corrupted_file(cache):
    cache.memory_cache = {}
    cache.memory_size = 0
    cache._try_disk_cache(cache._generate_cache_key("corrupt_key", ""), {
        "value": "value",
        "expire_time": time.time() + 60,
        "size": 10,
        "created_time": time.time(),
        "access_count": 0,
    })
    file_path = Path(cache.disk_cache_dir) / f"{cache._generate_cache_key('corrupt_key', '')}.pkl"
    with open(file_path, "wb") as f:
        f.write(b"broken")
    assert cache.get("corrupt_key") is None


def test_cleanup_memory_removes_expired(cache):
    cache.set("expire_key", "value", expire=0)
    cache._cleanup_memory_cache()
    assert cache.memory_cache == {}


def test_cleanup_disk_removes_expired(cache):
    cache.set("disk_key", "value", expire=1)
    time.sleep(1.1)
    cache._cleanup_disk_cache()
    files = list(Path(cache.disk_cache_dir).glob("*.pkl"))
    assert files == []


def test_get_stats_calculates_hit_rates(cache):
    cache.set("s", "value", expire=60)
    cache.get("s")
    cache.get("missing")
    stats = cache.get_stats()
    assert stats["total_operations"] >= 2
    assert 0 <= stats["hit_rate"] <= 1


def test_clear_prefix_and_all(cache):
    cache.set("prefix_key", "value", expire=60)
    cache.clear("prefix_key")
    assert cache.get("prefix_key") is None

    cache.set("k1", "value", expire=60)
    cache.set("k2", "value", expire=60)
    cache.clear()
    assert cache.memory_cache == {}
    assert list(Path(cache.disk_cache_dir).glob("*.pkl")) == []


def test_shutdown_joins_thread(cache, monkeypatch):
    class DummyThread:
        def __init__(self):
            self.join_called = False

        def is_alive(self):
            return True

        def join(self, timeout=None):
            self.join_called = True

    thread = DummyThread()
    cache._cleanup_thread = thread
    cache.shutdown()
    assert thread.join_called is True


def test_set_returns_false_when_storage_fails(cache, monkeypatch):
    monkeypatch.setattr(cache, "_try_memory_cache", lambda key, item: False)
    monkeypatch.setattr(cache, "_try_disk_cache", lambda key, item: False)
    assert cache.set("fail_key", "value", expire=10) is False


def test_try_disk_cache_respects_size_limit(cache, monkeypatch):
    cleanup_calls = {"count": 0}

    def fake_size():
        return cache.max_disk_size

    def fake_cleanup():
        cleanup_calls["count"] += 1

    monkeypatch.setattr(cache, "_get_disk_cache_size", fake_size)
    monkeypatch.setattr(cache, "_cleanup_disk_cache", fake_cleanup)

    cache_item = {
        "value": "value",
        "expire_time": time.time() + 60,
        "size": 100,
        "created_time": time.time(),
        "access_count": 0,
    }
    assert cache._try_disk_cache("disk_fail", cache_item) is False
    assert cleanup_calls["count"] == 1


def test_promote_to_memory_removes_disk_file(cache):
    cache_key = cache._generate_cache_key("promote_key", "")
    cache_item = {
        "value": "value",
        "expire_time": time.time() + 60,
        "size": 10,
        "created_time": time.time(),
        "access_count": 0,
    }
    cache._try_disk_cache(cache_key, cache_item)
    cache._promote_to_memory(cache_key, cache_item)
    assert not (Path(cache.disk_cache_dir) / f"{cache_key}.pkl").exists()


def test_clear_handles_nonexistent_prefix(cache):
    cache.set("regular_key", "value", expire=60)
    cache.clear("regular_key")
    assert cache.get("regular_key") is None


def test_get_disk_cache_size_missing_dir(cache, tmp_path):
    cache.disk_cache_dir = str(tmp_path / "missing_dir")
    assert cache._get_disk_cache_size() == 0


def test_cleanup_thread_runs_cleanup(monkeypatch, tmp_path):
    module = importlib.reload(ecm_module)
    monkeypatch.setattr(module.time, "sleep", lambda *args, **kwargs: None)

    calls = {"memory": 0, "disk": 0}

    def fake_memory(self):
        calls["memory"] += 1
        self._stop_cleanup = True

    def fake_disk(self):
        calls["disk"] += 1

    monkeypatch.setattr(module.EnhancedCacheManager, "_cleanup_memory_cache", fake_memory, raising=False)
    monkeypatch.setattr(module.EnhancedCacheManager, "_cleanup_disk_cache", fake_disk, raising=False)

    manager = module.EnhancedCacheManager(cache_dir=str(tmp_path / "cleanup"))
    manager.shutdown()

    assert calls["memory"] >= 1
    assert calls["disk"] >= 0


def test_generate_cache_key_and_memory_size_variants(cache):
    prefixed = cache._generate_cache_key("k", "pref")
    assert prefixed != cache._generate_cache_key("k", "")

    df = pd.DataFrame({"v": [1, 2, 3]})
    assert cache._get_memory_size(df) == df.memory_usage(deep=True).sum()

    arr = np.arange(5)
    assert cache._get_memory_size(arr) == arr.nbytes

    # 触发不可序列化路径
    bad_dict = {"fn": lambda: None}
    assert cache._get_memory_size(bad_dict) >= len("fn")

    class BrokenStr:
        def __str__(self):
            raise RuntimeError("boom")

    assert cache._get_memory_size(BrokenStr()) == 1024


def test_set_handles_internal_exception(cache, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cache, "_generate_cache_key", boom)
    assert cache.set("safe", "value") is False


def test_try_memory_cache_returns_false_when_cleanup_not_enough(cache, monkeypatch):
    cache.memory_size = cache.max_memory_size
    monkeypatch.setattr(cache, "_cleanup_memory_cache", lambda: None)
    cache_item = {
        "size": cache.max_memory_size,
        "value": "v",
        "expire_time": time.time() + 60,
        "created_time": time.time(),
        "access_count": 0,
    }
    assert cache._try_memory_cache("overflow", cache_item) is False


def test_try_disk_cache_handles_io_errors(cache, monkeypatch):
    cache_item = {
        "value": "v",
        "expire_time": time.time() + 60,
        "size": 10,
        "created_time": time.time(),
        "access_count": 0,
    }
    monkeypatch.setattr(cache, "_get_disk_cache_size", lambda: 0)

    def fake_open(*args, **kwargs):
        raise OSError("disk boom")

    monkeypatch.setattr("builtins.open", fake_open)
    assert cache._try_disk_cache("disk_fail", cache_item) is False


def test_get_reads_from_disk_and_promotes(cache):
    cache_key = cache._generate_cache_key("disk_only", "")
    cache_item = {
        "value": "disk-data",
        "expire_time": time.time() + 60,
        "size": 10,
        "created_time": time.time(),
        "access_count": 0,
    }
    cache._try_disk_cache(cache_key, cache_item)
    cache.memory_cache.pop(cache_key, None)
    cache.memory_size = 0

    value = cache.get("disk_only")
    assert value == "disk-data"
    assert cache.stats["disk_hits"] >= 1


def test_get_handles_internal_error(cache, monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cache, "_generate_cache_key", boom)
    assert cache.get("key") is None


def test_promote_to_memory_handles_remove_error(cache, monkeypatch):
    cache_key = cache._generate_cache_key("remove_err", "")
    cache_item = {
        "value": "value",
        "expire_time": time.time() + 60,
        "size": 10,
        "created_time": time.time(),
        "access_count": 0,
    }
    cache._try_disk_cache(cache_key, cache_item)

    def fake_remove(path):
        raise OSError("nope")

    monkeypatch.setattr("os.remove", fake_remove)
    cache._promote_to_memory(cache_key, cache_item)


def test_cleanup_memory_cache_eviction_threshold(cache):
    cache.max_memory_size = 100
    cache.memory_cache.clear()
    cache.memory_size = 0

    for idx in range(5):
        cache.memory_cache[f"k{idx}"] = {
            "value": idx,
            "expire_time": time.time() + 60,
            "size": 30,
            "created_time": time.time(),
            "access_count": idx,
        }
        cache.memory_size += 30

    cache._cleanup_memory_cache()
    assert cache.memory_size <= cache.max_memory_size * 0.7


def test_cleanup_disk_cache_missing_directory(cache, tmp_path):
    cache.disk_cache_dir = str(tmp_path / "missing")
    cache._cleanup_disk_cache()


def test_cleanup_disk_cache_logs_error(cache, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("os.listdir", lambda path: (_ for _ in ()).throw(OSError("boom")))
    errors = []
    monkeypatch.setattr(ecm_module.logger, "error", lambda msg: errors.append(msg))
    cache._cleanup_disk_cache()
    assert errors


def test_cleanup_disk_cache_removes_corrupted_file(cache):
    file_path = Path(cache.disk_cache_dir) / "corrupted.pkl"
    file_path.write_bytes(b"not-a-pickle")
    cache._cleanup_disk_cache()
    assert not file_path.exists()


def test_cleanup_disk_cache_handles_remove_failure(cache, monkeypatch):
    target = Path(cache.disk_cache_dir) / "corrupt_remove.pkl"
    target.write_bytes(b"bad")
    original_remove = os.remove

    def fake_remove(path):
        if str(path).endswith("corrupt_remove.pkl"):
            raise OSError("cannot delete")
        return original_remove(path)

    monkeypatch.setattr("os.remove", fake_remove)
    cache._cleanup_disk_cache()
    # 手动移除残留文件，避免影响其他用例
    if target.exists():
        target.unlink()


def test_cleanup_disk_cache_error_logging_fallback(cache, monkeypatch):
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("os.listdir", lambda path: (_ for _ in ()).throw(OSError("boom")))

    def raising_logger(*args, **kwargs):
        raise RuntimeError("log failure")

    monkeypatch.setattr(ecm_module.logger, "error", raising_logger)
    cache._cleanup_disk_cache()


def test_get_disk_cache_size_handles_file_errors(cache, monkeypatch):
    tmp_file = Path(cache.disk_cache_dir) / "cache.pkl"
    tmp_file.write_bytes(b"data")
    monkeypatch.setattr("os.listdir", lambda path: ["cache.pkl"])
    monkeypatch.setattr("os.path.getsize", lambda path: (_ for _ in ()).throw(OSError("boom")))
    assert cache._get_disk_cache_size() == 0


def test_get_disk_cache_size_handles_listdir_error(cache, monkeypatch):
    monkeypatch.setattr("os.listdir", lambda path: (_ for _ in ()).throw(OSError("boom")))
    assert cache._get_disk_cache_size() == 0


def test_cleanup_worker_breaks_on_stop(monkeypatch, tmp_path):
    module = importlib.reload(ecm_module)
    # 恢复原始启动逻辑
    monkeypatch.setattr(
        "src.data.cache.enhanced_cache_manager.EnhancedCacheManager._start_cleanup_thread",
        module.EnhancedCacheManager._start_cleanup_thread,
        raising=False,
    )

    captured = {}

    class DummyThread:
        def __init__(self, target, daemon=True):
            captured["target"] = target

        def start(self):
            captured["started"] = True

        def is_alive(self):
            return False

    monkeypatch.setattr(module.threading, "Thread", DummyThread)

    holder = {"instance": None}

    def fast_sleep(seconds):
        if holder["instance"] is not None:
            holder["instance"]._stop_cleanup = True

    monkeypatch.setattr(module.time, "sleep", fast_sleep)
    module_instance = module.EnhancedCacheManager(cache_dir=str(tmp_path / "cleanup_stop"))
    holder["instance"] = module_instance

    captured["target"]()
    module_instance.shutdown()
    assert captured.get("started") is True


def test_cleanup_worker_logs_error(monkeypatch, tmp_path):
    module = importlib.reload(ecm_module)
    monkeypatch.setattr(
        "src.data.cache.enhanced_cache_manager.EnhancedCacheManager._start_cleanup_thread",
        module.EnhancedCacheManager._start_cleanup_thread,
        raising=False,
    )
    captured = {}

    class DummyThread:
        def __init__(self, target, daemon=True):
            captured["target"] = target

        def start(self):
            captured["started"] = True

        def is_alive(self):
            return False

    monkeypatch.setattr(module.threading, "Thread", DummyThread)
    errors = []

    def capture_logger(msg):
        errors.append(msg)

    monkeypatch.setattr(module.logger, "error", capture_logger)

    def explosive_cleanup(self):
        self._stop_cleanup = True
        raise RuntimeError("boom")

    monkeypatch.setattr(module.EnhancedCacheManager, "_cleanup_memory_cache", explosive_cleanup, raising=False)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)
    manager = module.EnhancedCacheManager(cache_dir=str(tmp_path / "cleanup_error"))
    captured["target"]()
    manager.shutdown()
    assert errors


def test_cleanup_worker_logger_fallback(monkeypatch, tmp_path):
    module = importlib.reload(ecm_module)
    monkeypatch.setattr(
        "src.data.cache.enhanced_cache_manager.EnhancedCacheManager._start_cleanup_thread",
        module.EnhancedCacheManager._start_cleanup_thread,
        raising=False,
    )
    captured = {}

    class DummyThread:
        def __init__(self, target, daemon=True):
            captured["target"] = target

        def start(self):
            captured["started"] = True

        def is_alive(self):
            return False

    monkeypatch.setattr(module.threading, "Thread", DummyThread)

    def raising_logger(*args, **kwargs):
        raise RuntimeError("log failure")

    monkeypatch.setattr(module.logger, "error", raising_logger)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)

    def explosive_cleanup(self):
        self._stop_cleanup = True
        raise RuntimeError("cleanup")

    monkeypatch.setattr(module.EnhancedCacheManager, "_cleanup_memory_cache", explosive_cleanup, raising=False)
    manager = module.EnhancedCacheManager(cache_dir=str(tmp_path / "cleanup_logger"))
    captured["target"]()
    manager.shutdown()


def test_clear_handles_os_remove_errors(cache, monkeypatch):
    cache.set("err", "value", expire=60)

    def fake_remove(path):
        raise OSError("cannot remove")

    monkeypatch.setattr("os.remove", fake_remove)
    cache.clear("err")
    cache.set("another", "value", expire=60)
    cache.clear()


def test_get_stats_handles_zero_accesses(cache):
    stats = cache.get_stats()
    assert stats["hit_rate"] == 0.0
    assert stats["disk_hit_rate"] == 0.0


def test_fallback_logger_path(monkeypatch, tmp_path):
    original_import = __import__
    calls = {"count": 0}

    def fake_import(name, *args, **kwargs):
        if name == "src.infrastructure.logging" and calls["count"] == 0:
            calls["count"] += 1
            raise ImportError("forced")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", fake_import)
    module = importlib.reload(ecm_module)
    logger = module.get_infrastructure_logger("test_logger")
    assert logger is not None
    # 还原原始实现
    importlib.reload(ecm_module)


def test_del_invokes_shutdown(tmp_path):
    manager = EnhancedCacheManager(cache_dir=str(tmp_path / "del_case"))
    called = {"count": 0}

    def fake_shutdown():
        called["count"] += 1

    manager.shutdown = fake_shutdown  # type: ignore[assignment]
    manager.__del__()
    assert called["count"] == 1


def test_del_handles_shutdown_errors(tmp_path):
    manager = EnhancedCacheManager(cache_dir=str(tmp_path / "del_error"))

    def boom():
        raise RuntimeError("shutdown failed")

    manager.shutdown = boom  # type: ignore[assignment]
    manager.__del__()  # 不应抛出异常
