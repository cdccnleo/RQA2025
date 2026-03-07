"""
补充测试：cache_manager.py
补充边界测试以提升覆盖率
"""
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


import pytest
from unittest.mock import Mock
from pathlib import Path
import tempfile

from src.data.cache.cache_manager import (
    CacheConfig,
    CacheEntry,
    CacheManager,
)


def test_cache_manager_strategy_on_evict_returns_none():
    """测试 CacheStrategy（on_evict 返回 None，覆盖 255 行）"""
    # 导入 approach 类
    from src.data.cache.cache_manager import approach as CacheStrategy
    
    class TestStrategy(CacheStrategy):
        pass
    
    strategy = TestStrategy()
    cache = {}
    config = CacheConfig()
    
    # on_evict 应该返回 None（覆盖 255 行）
    result = strategy.on_evict(cache, config)
    assert result is None


def test_cache_manager_save_to_disk_disabled():
    """测试 CacheManager（保存到磁盘，禁用磁盘缓存，覆盖 388 行）"""
    config = CacheConfig(enable_disk_cache=False)
    manager = CacheManager(config=config)
    
    try:
        # 尝试保存到磁盘，应该返回 False（覆盖 388 行）
        entry = CacheEntry(key="key1", value="value1", ttl=3600)
        result = manager._save_to_disk("key1", entry)
        assert result is False
    finally:
        manager.stop()


def test_cache_manager_load_from_disk_disabled():
    """测试 CacheManager（从磁盘加载，禁用磁盘缓存，覆盖 411 行）"""
    config = CacheConfig(enable_disk_cache=False)
    manager = CacheManager(config=config)
    
    try:
        # 尝试从磁盘加载，应该返回 None（覆盖 411 行）
        result = manager._load_from_disk("key1")
        assert result is None
    finally:
        manager.stop()


def test_cache_manager_load_from_disk_file_not_exists(tmp_path):
    """测试 CacheManager（从磁盘加载，文件不存在，覆盖 416 行）"""
    config = CacheConfig(
        enable_disk_cache=True,
        disk_cache_dir=str(tmp_path)
    )
    manager = CacheManager(config=config)
    
    try:
        # 尝试加载不存在的文件，应该返回 None（覆盖 416 行）
        result = manager._load_from_disk("nonexistent_key")
        assert result is None
    finally:
        manager.stop()

