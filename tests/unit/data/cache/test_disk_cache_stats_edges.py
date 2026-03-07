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
from src.data.cache.disk_cache import DiskCache


def test_collect_disk_usage_on_empty_dir(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # 直接构造对象并注入 cache_dir，避免依赖完整配置
    dc = object.__new__(DiskCache)  # type: ignore[misc]
    dc.cache_dir = cache_dir  # type: ignore[attr-defined]
    file_count, total_size = dc._collect_disk_usage()
    assert file_count == 0 and total_size == 0


def test_collect_disk_usage_with_files(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    # create two fake cache files
    f1 = cache_dir / "a.cache"
    f2 = cache_dir / "b.cache"
    f1.write_bytes(b"12345")
    f2.write_bytes(b"67890")
    dc = object.__new__(DiskCache)  # type: ignore[misc]
    dc.cache_dir = cache_dir  # type: ignore[attr-defined]
    file_count, total_size = dc._collect_disk_usage()
    assert file_count >= 2 and total_size >= 10


