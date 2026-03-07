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

from src.data.cache.disk_cache import DiskCache, DiskCacheConfig


def _build_disk_cache(tmp_path, **overrides):
    config = DiskCacheConfig(
        cache_dir=str(tmp_path),
        max_file_size=overrides.get("max_file_size", 1024),
        cleanup_interval=overrides.get("cleanup_interval", 0),
    )
    return DiskCache(config=config)


def test_disk_cache_handles_corrupted_file(tmp_path):
    disk_cache = _build_disk_cache(tmp_path)
    try:
        corrupt_path = disk_cache._get_file_path("corrupt-key")
        corrupt_path.write_bytes(b"not a pickle payload")
        assert disk_cache.get("corrupt-key") is None
        assert not corrupt_path.exists()
    finally:
        disk_cache.stop()


def test_disk_cache_removes_oversized_file(tmp_path):
    disk_cache = _build_disk_cache(tmp_path, max_file_size=4)
    try:
        large_path = disk_cache._get_file_path("large-key")
        large_path.write_bytes(b"a" * 10)
        assert disk_cache.get("large-key") is None
        assert not large_path.exists()
    finally:
        disk_cache.stop()


def test_disk_cache_health_check_permission_error(tmp_path, monkeypatch):
    disk_cache = _build_disk_cache(tmp_path)
    try:
        monkeypatch.setattr(os, "access", lambda *args, **kwargs: False)
        status = disk_cache.health_check()
        assert status["status"] == "error"
        assert "permission" in status["message"].lower()
    finally:
        disk_cache.stop()


def test_disk_cache_list_keys(tmp_path):
    disk_cache = _build_disk_cache(tmp_path)
    try:
        disk_cache.set("k1", "v1")
        disk_cache.set("k2", "v2")
        keys = disk_cache.list_keys()
        assert len(keys) == 2
    finally:
        disk_cache.stop()


def test_disk_cache_exists_invokes_get(tmp_path):
    disk_cache = _build_disk_cache(tmp_path)
    try:
        disk_cache.set("existing", "value")
        assert disk_cache.exists("existing") is True
        disk_cache.delete("existing")
        assert disk_cache.exists("existing") is False
    finally:
        disk_cache.stop()

