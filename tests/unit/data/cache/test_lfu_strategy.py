# -*- coding: utf-8 -*-
"""
LFU缓存策略测试
测试最少访问淘汰策略的正确性
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
from datetime import datetime, timedelta
from unittest.mock import Mock
from src.data.cache.lfu_strategy import LFUStrategy


class MockCacheEntry:
    """模拟缓存条目"""

    def __init__(self, key: str, access_count: int = 0, created_at: datetime = None):
        self.key = key
        self.access_count = access_count
        self.created_at = created_at or datetime.now()


class MockCacheConfig:
    """模拟缓存配置"""

    def __init__(self):
        self.max_size = 100
        self.ttl = None


class TestLFUStrategy:
    """测试LFU缓存策略"""

    def setup_method(self):
        """测试前准备"""
        self.strategy = LFUStrategy()
        self.config = MockCacheConfig()

    def test_on_set_no_operation(self):
        """测试on_set方法不执行任何操作"""
        cache = {}
        entry = MockCacheEntry("test_key")

        # on_set不应该修改缓存或条目
        self.strategy.on_set(cache, "test_key", entry, self.config)

        assert len(cache) == 0

    def test_on_get_no_operation(self):
        """测试on_get方法不执行任何操作"""
        cache = {"test_key": MockCacheEntry("test_key")}
        entry = cache["test_key"]

        # on_get不应该修改缓存或条目
        self.strategy.on_get(cache, "test_key", entry, self.config)

        assert len(cache) == 1
        assert "test_key" in cache

    def test_on_evict_empty_cache(self):
        """测试空缓存的淘汰"""
        cache = {}

        result = self.strategy.on_evict(cache, self.config)

        assert result is None

    def test_on_evict_single_entry(self):
        """测试单个条目的淘汰"""
        cache = {
            "key1": MockCacheEntry("key1", access_count=5)
        }

        result = self.strategy.on_evict(cache, self.config)

        assert result == "key1"

    def test_on_evict_multiple_different_access_counts(self):
        """测试多个不同访问次数的淘汰"""
        cache = {
            "key1": MockCacheEntry("key1", access_count=10),
            "key2": MockCacheEntry("key2", access_count=5),
            "key3": MockCacheEntry("key3", access_count=15)
        }

        result = self.strategy.on_evict(cache, self.config)

        assert result == "key2"  # 最少访问的key2应该被淘汰

    def test_on_evict_multiple_same_access_count(self):
        """测试多个相同访问次数的淘汰（选择最早创建的）"""
        early_time = datetime.now() - timedelta(hours=1)
        later_time = datetime.now()

        cache = {
            "key1": MockCacheEntry("key1", access_count=5, created_at=later_time),
            "key2": MockCacheEntry("key2", access_count=5, created_at=early_time),
            "key3": MockCacheEntry("key3", access_count=10)
        }

        result = self.strategy.on_evict(cache, self.config)

        assert result == "key2"  # key2访问次数相同但创建时间最早

    def test_on_evict_multiple_candidates_earliest_creation(self):
        """测试多个候选者中选择最早创建的"""
        early_time = datetime.now() - timedelta(hours=2)
        middle_time = datetime.now() - timedelta(hours=1)
        late_time = datetime.now()

        cache = {
            "key1": MockCacheEntry("key1", access_count=1, created_at=middle_time),
            "key2": MockCacheEntry("key2", access_count=1, created_at=early_time),
            "key3": MockCacheEntry("key3", access_count=1, created_at=late_time),
            "key4": MockCacheEntry("key4", access_count=2)
        }

        result = self.strategy.on_evict(cache, self.config)

        assert result == "key2"  # key2创建时间最早

    def test_on_evict_zero_access_count(self):
        """测试访问次数为0的条目淘汰"""
        cache = {
            "key1": MockCacheEntry("key1", access_count=0),
            "key2": MockCacheEntry("key2", access_count=0),
            "key3": MockCacheEntry("key3", access_count=1)
        }

        result = self.strategy.on_evict(cache, self.config)

        # 应该淘汰访问次数为0的条目中最早创建的
        assert result in ["key1", "key2"]

    def test_strategy_initialization(self):
        """测试策略初始化"""
        strategy = LFUStrategy()
        assert strategy is not None
        assert hasattr(strategy, 'on_set')
        assert hasattr(strategy, 'on_get')
        assert hasattr(strategy, 'on_evict')
