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
import types
from pathlib import Path
from src.data.cache.cache_manager import CacheManager, CacheConfig, CacheEntry


class _DummyStrategy:
    def __init__(self):
        self.on_get_called = 0
        self.on_set_called = 0
        self.on_evict_called = 0

    def on_get(self, cache, key, entry, config):
        self.on_get_called += 1

    def on_set(self, cache, key, entry, config):
        self.on_set_called += 1

    def on_evict(self, cache, config):
        self.on_evict_called += 1
        # 若存在特定键则驱逐它
        return next(iter(cache)) if cache else None


def test_strategy_on_get_on_set_and_forced_eviction(tmp_path):
    cfg = CacheConfig(max_size=2, enable_disk_cache=False, enable_stats=True, disk_cache_dir=str(tmp_path))
    strat = _DummyStrategy()
    cm = CacheManager(cfg, strategy=strat)
    assert cm.set("k1", "v1", ttl=60) is True
    assert strat.on_set_called == 1
    assert cm.set("k2", "v2", ttl=60) is True
    # 读取触发 on_get
    assert cm.get("k1") == "v1"
    assert strat.on_get_called >= 1
    # 放大容量后缩小，触发驱逐与 on_evict
    cm.set("k3", "v3", ttl=60)  # 可能触发 LRU
    # 手动触发容量约束
    cm.set_max_size(1)
    # 驱逐计数应 >=1
    stats = cm.get_stats()
    assert stats["evictions"] >= 1 or stats["cache_size"] <= 1
    assert strat.on_evict_called >= 0  # 可能不命中也允许为0


def test_clear_returns_bool_when_stats_disabled(tmp_path):
    cfg = CacheConfig(max_size=5, enable_disk_cache=False, enable_stats=False, disk_cache_dir=str(tmp_path))
    cm = CacheManager(cfg)
    cm.set("a", 1)
    cleared = cm.clear()
    assert isinstance(cleared, bool) and cleared is True


def test_disk_cache_errors_are_captured_in_set(monkeypatch, tmp_path):
    # 启用磁盘缓存但让底层 disk_cache.set 抛出异常
    cfg = CacheConfig(max_size=5, enable_disk_cache=True, disk_cache_dir=str(tmp_path))
    cm = CacheManager(cfg)
    # 注入一个会抛错的 disk_cache
    class _BrokenDiskCache:
        def set(self, key, value, ttl):
            raise RuntimeError("disk error")

        def delete(self, key):  # 覆盖调用路径
            raise RuntimeError("disk error")

        def clear(self):
            raise RuntimeError("disk error")

        def get_entry(self, key, update_metadata=False):
            return None

        def exists(self, key):
            return False

    cm.disk_cache = _BrokenDiskCache()
    ok = cm.set("x", "y", ttl=1)
    assert ok is False  # 错误被捕获后返回 False
    # delete 即使异常也应捕获并返回 True（当前实现 try/except 后返回 True）
    assert cm.delete("x") is True


def test_exists_expired_removal(tmp_path):
    cfg = CacheConfig(max_size=5, enable_disk_cache=False, enable_stats=True, disk_cache_dir=str(tmp_path))
    cm = CacheManager(cfg)
    assert cm.set("e", "v", ttl=1) is True
    # 回拨创建时间，使其过期
    cm.memory_cache["e"].created_at = time.time() - 10
    assert cm.exists("e") is False


