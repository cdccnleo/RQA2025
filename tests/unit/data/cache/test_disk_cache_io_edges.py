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
import io
import time
import pickle
import tempfile
from pathlib import Path

import pytest

from src.data.cache.disk_cache import DiskCache, DiskCacheConfig
from src.data.cache.cache_manager import CacheEntry, CacheStats


def _mk_cache(tmpdir: Path, max_file_size: int = 1024 * 1024) -> DiskCache:
    cfg = DiskCacheConfig(cache_dir=str(tmpdir), max_file_size=max_file_size)
    cache = DiskCache(cfg)
    return cache


def test_get_entry_deletes_too_large_file(tmp_path):
    cache = _mk_cache(tmp_path, max_file_size=32)  # 32 bytes
    key = "k1"

    # 人工创建超大文件
    file_path = cache._get_file_path(key)  # type: ignore[attr-defined]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(b"x" * 128)

    # 触发读取逻辑：应当删除并返回 None，且统计记为 error
    before_errors = cache.stats.errors
    entry = cache.get_entry(key)
    assert entry is None
    assert not file_path.exists()
    assert cache.stats.errors == before_errors + 1

    cache.stop()


def test_get_entry_corrupted_pickle_is_deleted(tmp_path):
    cache = _mk_cache(tmp_path)
    key = "k2"
    file_path = cache._get_file_path(key)  # type: ignore[attr-defined]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(b"\x80\x04not-a-pickle")

    before_errors = cache.stats.errors
    res = cache.get_entry(key)
    assert res is None
    assert not file_path.exists()
    assert cache.stats.errors == before_errors + 1

    cache.stop()


def test_get_entry_read_ioerror_graceful(tmp_path, monkeypatch):
    cache = _mk_cache(tmp_path)
    key = "k3"
    file_path = cache._get_file_path(key)  # type: ignore[attr-defined]
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "wb") as f:
        # 写入一个看似正常的内容以确保先走到 open 分支
        f.write(b"\x80\x04K.")  # minimal pickle int

    def _bad_open(path, mode="rb", *args, **kwargs):
        if str(path) == str(file_path) and "rb" in mode:
            raise IOError("perm denied")
        return open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", _bad_open, raising=True)

    before_errors = cache.stats.errors
    res = cache.get_entry(key)
    assert res is None
    assert not file_path.exists()  # 发生IOError后会删除文件
    assert cache.stats.errors == before_errors + 1

    cache.stop()


