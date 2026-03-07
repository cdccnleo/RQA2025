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


import time
from typing import Optional

import pytest

from src.data.cache.cache_manager import CacheManager, CacheConfig, CacheEntry


class FakeDiskCache:
    """最小磁盘缓存桩，便于模拟磁盘读取/删除行为。"""

    def __init__(self, entry: Optional[CacheEntry] = None):
        self._entry = entry
        self.deleted_keys: list[str] = []

    def get_entry(self, key: str, update_metadata: bool = False):
        return self._entry

    def delete(self, key: str) -> None:
        self.deleted_keys.append(key)

    def set(self, key: str, value, ttl: int):
        self._entry = CacheEntry(key=key, value=value, ttl=ttl)

    def clear(self):
        self._entry = None

    def stop(self):
        pass

    def close(self):
        pass

    def get_stats(self):
        return {"disk_cache": {}}


@pytest.fixture
def cache_manager(tmp_path):
    config = CacheConfig(
        max_size=3,
        ttl=5,
        enable_disk_cache=False,
        enable_stats=False,
        disk_cache_dir=str(tmp_path),
    )
    manager = CacheManager(config=config)
    yield manager
    manager.close()


def test_get_uses_disk_cache_when_memory_miss(cache_manager):
    entry = CacheEntry(key="foo", value={"answer": 42}, ttl=30)
    fake_disk = FakeDiskCache(entry=entry)
    cache_manager.disk_cache = fake_disk

    value = cache_manager.get("foo")

    assert value == {"answer": 42}
    assert "foo" in cache_manager.memory_cache
    assert cache_manager._stats.hits == 1


def test_get_ignores_stale_disk_entries_after_clear(cache_manager):
    entry = CacheEntry(key="foo", value="stale", ttl=30)
    entry.created_at = time.time() - 100
    fake_disk = FakeDiskCache(entry=entry)
    cache_manager.disk_cache = fake_disk
    cache_manager._last_clear_time = time.time()

    value = cache_manager.get("foo")

    assert value is None
    assert fake_disk.deleted_keys == ["foo"]
    assert cache_manager._stats.misses == 1


def test_cleanup_expired_removes_entries_and_updates_stats(cache_manager):
    cache_manager.set("old", "value", ttl=1)
    cache_manager.memory_cache["old"].created_at -= 5

    removed = cache_manager.cleanup_expired()

    assert removed == 1
    assert "old" not in cache_manager.memory_cache
    assert cache_manager._stats.evictions == 1


def test_health_check_returns_error_when_set_fails(cache_manager, monkeypatch):
    monkeypatch.setattr(cache_manager, "set", lambda *args, **kwargs: False)

    result = cache_manager.health_check()

    assert result["status"] == "error"
    assert result["message"] == "Set operation failed"

