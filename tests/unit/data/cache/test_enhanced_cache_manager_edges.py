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

from src.data.cache.enhanced_cache_manager import EnhancedCacheManager


def _mk(cache_dir: Path) -> EnhancedCacheManager:
    return EnhancedCacheManager(cache_dir=str(cache_dir), max_memory_size=1024 * 1024, max_disk_size=1024 * 1024)


def test_set_invalid_inputs_raise(tmp_path):
    ecm = _mk(tmp_path)
    with pytest.raises(ValueError):
        ecm.set("", 1)
    with pytest.raises(ValueError):
        ecm.set("k", None)  # type: ignore
    with pytest.raises(ValueError):
        ecm.set("k", 1, expire=-1)
    ecm.shutdown()


def test_set_disk_write_failure_falls_back_to_memory(tmp_path, monkeypatch):
    ecm = _mk(tmp_path)
    # 通过模拟 open 写磁盘失败
    def _bad_open(path, mode="rb", *args, **kwargs):
        if isinstance(path, str) and path.endswith(".pkl") and "wb" in mode:
            raise OSError("disk readonly")
        return open(path, mode, *args, **kwargs)
    monkeypatch.setattr("builtins.open", _bad_open, raising=True)
    ok = ecm.set("k1", {"v": 1}, expire=60)
    # 设计为内存或磁盘任一成功即 True；此处磁盘失败，但应成功存入内存
    assert ok is True
    # 验证磁盘未写入
    cache_key = ecm._generate_cache_key("k1", "")  # type: ignore[attr-defined]
    cache_file = Path(ecm.disk_cache_dir) / f"{cache_key}.pkl"  # type: ignore[attr-defined]
    assert not cache_file.exists()
    ecm.shutdown()


def test_get_expired_item_returns_none_and_counts_miss(tmp_path):
    ecm = _mk(tmp_path)
    assert ecm.set("k2", "v2", expire=0) is True  # 立即过期
    v = ecm.get("k2")
    assert v is None
    stats = ecm.get_stats()
    assert stats["misses"] >= 1
    ecm.shutdown()


def test_promote_from_disk_and_clear_prefix(tmp_path):
    ecm = _mk(tmp_path)
    # 正常 set（写入内存并尝试磁盘）
    assert ecm.set("k3", {"n": 3}, expire=60, prefix="px")
    # 命中前置索引
    cache_key = ecm._generate_cache_key("k3", "px")  # type: ignore[attr-defined]
    # 确保磁盘文件存在（若只进内存，手动写入以覆盖 get-from-disk/提升路径）
    cache_file = Path(ecm.disk_cache_dir) / f"{cache_key}.pkl"  # type: ignore[attr-defined]
    if not cache_file.exists():
        with open(cache_file, "wb") as f:
            pickle.dump({
                "value": {"n": 3},
                "expire_time": time.time() + 60,
                "size": 10,
                "created_time": time.time(),
                "access_count": 0
            }, f)
    # 首次 get 应命中磁盘并提升到内存
    v = ecm.get("k3", prefix="px")
    assert isinstance(v, dict) and v["n"] == 3
    # 清理 prefix，确保内存与磁盘清除
    ecm.clear("px")
    assert ecm.get("k3", prefix="px") is None
    ecm.shutdown()


