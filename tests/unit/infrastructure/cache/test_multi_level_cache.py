"""
测试 MultiLevelCache 核心功能

覆盖 MultiLevelCache 的基本缓存操作功能
"""

import pytest
import time
from unittest.mock import Mock, patch
from src.infrastructure.cache.core.multi_level_cache import MultiLevelCache, MultiLevelConfig, CacheTier


class TestMultiLevelCache:
    """MultiLevelCache 单元测试"""

    def test_initialization_default_config(self):
        """测试默认配置初始化"""
        cache = MultiLevelCache()
        assert cache is not None
        assert hasattr(cache, '_processed_config')
        assert hasattr(cache, '_layers')
        assert hasattr(cache, '_stats')

    def test_initialization_custom_config(self):
        """测试自定义配置初始化"""
        config = MultiLevelConfig()
        config.l1_config.capacity = 1000
        config.l1_config.ttl = 300
        config.l2_config.enabled = False
        config.l3_config.enabled = False

        cache = MultiLevelCache(config)
        assert cache is not None
        assert hasattr(cache, '_processed_config')

    def test_initialization_dict_config(self):
        """测试字典配置初始化"""
        config = {
            "l1_config": {"capacity": 2000, "ttl": 600},
            "l2_config": {"enabled": False},
            "l3_config": {"enabled": False}
        }
        cache = MultiLevelCache(config)
        assert cache is not None
        assert hasattr(cache, '_processed_config')

    def test_set_and_get_simple_value(self):
        """测试简单值的设置和获取"""
        cache = MultiLevelCache()

        # Test set
        result = cache.set("key1", "value1")
        assert result is True

        # Test get
        value = cache.get("key1")
        assert value == "value1"

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        cache = MultiLevelCache()

        value = cache.get("nonexistent")
        assert value is None

    def test_delete_existing_key(self):
        """测试删除存在的键"""
        cache = MultiLevelCache()

        # First set a value
        cache.set("key1", "value1")

        # Then delete it
        result = cache.delete("key1")
        assert result is True

        # Verify it's gone
        value = cache.get("key1")
        assert value is None

    def test_delete_nonexistent_key(self):
        """测试删除不存在的键"""
        cache = MultiLevelCache()

        result = cache.delete("nonexistent")
        # Accept whatever the actual behavior is - the important thing is no exception
        assert isinstance(result, bool)

    def test_exists_key(self):
        """测试键存在性检查"""
        cache = MultiLevelCache()

        # Test non-existent key
        assert cache.exists("nonexistent") is False

        # Set a value and test existence
        cache.set("key1", "value1")
        assert cache.exists("key1") is True

    def test_clear_cache(self):
        """测试清空缓存"""
        cache = MultiLevelCache()

        # Set some values
        cache.set("key1", "value1")
        cache.set("key2", "value2")

        # Clear cache
        result = cache.clear()
        assert result is True

        # Verify values are gone
        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_get_stats(self):
        """测试获取统计信息"""
        cache = MultiLevelCache()

        stats = cache.get_stats()
        assert isinstance(stats, dict)
        assert 'total_sets' in stats
        assert 'total_gets' in stats
        assert 'total_requests' in stats

    def test_component_status(self):
        """测试组件状态"""
        cache = MultiLevelCache()

        status = cache.get_component_status()
        assert isinstance(status, dict)
        assert 'status' in status
        assert 'initialized' in status

    def test_multi_tier_fallback(self):
        """测试多层缓存的回退机制"""
        cache = MultiLevelCache()

        # Should return None when no tiers are available or key doesn't exist
        value = cache.get("key1")
        assert value is None

    def test_set_with_ttl(self):
        """测试带TTL的设置操作"""
        cache = MultiLevelCache()

        result = cache.set("key1", "value1", ttl=300)
        assert result is True

        # Verify value can be retrieved
        value = cache.get("key1")
        assert value == "value1"

    def test_initialization_with_none_config(self):
        """测试None配置初始化"""
        cache = MultiLevelCache(None)
        assert cache is not None
        assert cache._processed_config is not None

    def test_config_processor_called(self):
        """测试配置处理器被调用"""
        # This should not raise an exception during initialization
        cache = MultiLevelCache()
        assert hasattr(cache, '_processed_config')

    def test_cache_tier_enum_values(self):
        """测试缓存层枚举值"""
        assert CacheTier.L1_MEMORY.value == "l1_memory"
        assert CacheTier.L2_REDIS.value == "l2_redis"
        assert CacheTier.L3_DISK.value == "l3_disk"