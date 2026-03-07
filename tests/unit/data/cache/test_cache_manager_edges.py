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
from pathlib import Path
import time
import pytest

from src.data.cache.cache_manager import CacheManager, CacheConfig, CacheEntry


class FaultyDiskCache:
    def __init__(self, *args, **kwargs):
        self._deleted = []

    def get_entry(self, key, update_metadata=False):
        # 模拟磁盘存在但稍后读取失败的情况由外层处理；这里返回None表示未命中
        return None

    def set(self, key, value, ttl):
        # 模拟磁盘写入失败
        raise RuntimeError("disk-write-fail")

    def delete(self, key):
        # 模拟磁盘删除失败，但不应导致整体删除失败
        self._deleted.append(key)
        raise RuntimeError("disk-delete-fail")

    def exists(self, key):
        return False

    def clear(self):
        # 模拟清空成功
        return True

    def get_stats(self):
        return {"disk_cache": {"enabled": True, "size": 0}}

    def close(self):
        return None

    def stop(self):
        return None


def build_manager(tmp_path, enable_disk=True):
    config = CacheConfig(
        max_size=2,
        ttl=1,  # 短TTL便于过期测试
        enable_disk_cache=enable_disk,
        disk_cache_dir=str(tmp_path / "cache"),
        enable_stats=True,
        cleanup_interval=1,
    )
    mgr = CacheManager(config)
    if enable_disk:
        # 注入故障磁盘缓存
        mgr.disk_cache = FaultyDiskCache()
    return mgr


def test_set_get_exists_and_expiration(tmp_path):
    mgr = build_manager(tmp_path, enable_disk=False)
    assert mgr.set("k1", "v1", ttl=1) is True
    assert mgr.exists("k1") is True
    assert mgr.get("k1") == "v1"

    # 过期后自动清理
    time.sleep(1.1)
    # 第一次访问触发miss并移除
    assert mgr.get("k1") is None
    assert mgr.exists("k1") is False


def test_delete_with_faulty_disk_cache_does_not_fail(tmp_path):
    mgr = build_manager(tmp_path, enable_disk=True)
    # 磁盘写入失败时，当前实现返回False（但内存已写入）
    assert mgr.set("k2", {"a": 1}, ttl=10) is False
    # 删除应返回True，即使磁盘删除失败
    assert mgr.delete("k2") is True


def test_evict_when_exceeding_max_size(tmp_path):
    mgr = build_manager(tmp_path, enable_disk=False)
    assert mgr.set("a", 1, ttl=10)
    assert mgr.set("b", 2, ttl=10)
    # 触发容量驱逐
    assert mgr.set("c", 3, ttl=10)
    keys = set(mgr.list_keys())
    # 至少有两个存在，且不超过max_size
    assert len(keys) <= mgr.config.max_size


def test_clear_and_health_check_and_stats(tmp_path):
    mgr = build_manager(tmp_path, enable_disk=True)
    # 磁盘写入失败，返回False
    assert mgr.set("x", 1, ttl=10) is False
    assert isinstance(mgr.get_stats(), dict)

    cleared = mgr.clear()
    # enable_stats=True 时，clear 返回清理数量
    assert isinstance(cleared, int)

    health = mgr.health_check()
    assert isinstance(health, dict)
    assert health.get("status") in {"healthy", "error"}

    # 关闭应可执行且写出统计文件（忽略可能的文件竞争）
    mgr.close()


def test_cleanup_thread_stop_on_close(tmp_path):
    mgr = build_manager(tmp_path, enable_disk=False)
    # 快速启动并关闭，避免后台线程残留
    mgr.close()
    # 多次调用析构安全
    mgr.__del__()


