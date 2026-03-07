"""
测试 UnifiedCacheManager 核心功能

覆盖 UnifiedCacheManager 的基本缓存管理功能
"""

import pytest
import time
from unittest.mock import Mock
from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager


class TestUnifiedCacheManager:
    """UnifiedCacheManager 单元测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        manager = UnifiedCacheManager()
        assert manager is not None
        assert hasattr(manager, '_cache')
        assert hasattr(manager, '_ttl_cache')
        assert hasattr(manager, '_stats')

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = Mock()
        config.basic = Mock()
        config.basic.max_size = 2000
        config.basic.ttl = 600
        config.multi_level = Mock()
        config.multi_level.memory_ttl = 600

        manager = UnifiedCacheManager(config)
        assert manager._max_size == 2000
        # The _default_ttl gets overridden by memory_ttl
        assert manager._default_ttl == 600

    def test_set_and_get_simple_value(self):
        """测试简单值的设置和获取"""
        manager = UnifiedCacheManager()

        # Test set
        result = manager.set("key1", "value1")
        assert result is True

        # Test get
        value = manager.get("key1")
        assert value == "value1"

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        manager = UnifiedCacheManager()

        value = manager.get("nonexistent")
        assert value is None

    def test_set_with_ttl(self):
        """测试带TTL的设置"""
        manager = UnifiedCacheManager()

        result = manager.set("key1", "value1", ttl=1)
        assert result is True

        # Immediately should still be available
        value = manager.get("key1")
        assert value == "value1"

        # Wait for expiration (in a real test, this would be mocked)
        # For now, just verify it was stored
        assert manager.exists("key1") is True

    def test_delete_existing_key(self):
        """测试删除存在的键"""
        manager = UnifiedCacheManager()

        # Set a value
        manager.set("key1", "value1")

        # Delete it
        result = manager.delete("key1")
        assert result is True

        # Verify it's gone
        assert manager.get("key1") is None
        assert manager.exists("key1") is False

    def test_delete_nonexistent_key(self):
        """测试删除不存在的键"""
        manager = UnifiedCacheManager()

        result = manager.delete("nonexistent")
        assert result is False

    def test_exists_key(self):
        """测试键存在性检查"""
        manager = UnifiedCacheManager()

        # Test non-existent
        assert manager.exists("nonexistent") is False

        # Set and test existence
        manager.set("key1", "value1")
        assert manager.exists("key1") is True

    def test_has_key_alias(self):
        """测试has_key方法（exists的别名）"""
        manager = UnifiedCacheManager()

        # Test non-existent
        assert manager.has_key("nonexistent") is False

        # Set and test existence
        manager.set("key1", "value1")
        assert manager.has_key("key1") is True

    def test_clear_cache(self):
        """测试清空缓存"""
        manager = UnifiedCacheManager()

        # Set some values
        manager.set("key1", "value1")
        manager.set("key2", "value2")

        # Clear
        result = manager.clear()
        assert result is True

        # Verify all gone
        assert manager.get("key1") is None
        assert manager.get("key2") is None

    def test_keys_method(self):
        """测试获取所有键"""
        manager = UnifiedCacheManager()

        # Initially empty
        keys = list(manager.keys())
        assert len(keys) == 0

        # Add some keys
        manager.set("key1", "value1")
        manager.set("key2", "value2")

        keys = list(manager.keys())
        assert len(keys) == 2
        assert "key1" in keys
        assert "key2" in keys

    def test_size_method(self):
        """测试获取缓存大小"""
        manager = UnifiedCacheManager()

        # Initially empty
        assert manager.size() == 0

        # Add items
        manager.set("key1", "value1")
        assert manager.size() == 1

        manager.set("key2", "value2")
        assert manager.size() == 2

        # Delete one
        manager.delete("key1")
        assert manager.size() == 1

    def test_get_stats(self):
        """测试获取统计信息"""
        manager = UnifiedCacheManager()

        stats = manager.get_stats()
        assert isinstance(stats, dict)
        assert "hits" in stats
        assert "misses" in stats
        assert "evictions" in stats

    def test_get_cache_stats(self):
        """测试获取缓存统计"""
        manager = UnifiedCacheManager()

        stats = manager.get_cache_stats()
        assert isinstance(stats, dict)

    def test_health_check(self):
        """测试健康检查"""
        manager = UnifiedCacheManager()

        health = manager.health_check()
        assert isinstance(health, dict)
        assert "status" in health

    def test_get_health_status(self):
        """测试获取健康状态"""
        manager = UnifiedCacheManager()

        status = manager.get_health_status()
        assert isinstance(status, dict)

    def test_get_monitoring_status(self):
        """测试获取监控状态"""
        manager = UnifiedCacheManager()

        status = manager.get_monitoring_status()
        assert isinstance(status, dict)

    def test_start_monitoring(self):
        """测试启动监控"""
        manager = UnifiedCacheManager()

        # Should not raise exception
        manager.start_monitoring(True)
        manager.start_monitoring(False)

    def test_invalid_key_validation(self):
        """测试无效键验证"""
        manager = UnifiedCacheManager()

        # Test with None key
        result = manager.get(None)
        assert result is None

        # Test set with None key
        result = manager.set(None, "value")
        assert result is False

    def test_key_type_coercion(self):
        """测试键类型强制转换"""
        manager = UnifiedCacheManager()

        # Test with integer key (should be converted to string)
        manager.set(123, "value")
        value = manager.get("123")
        assert value == "value"

    def test_performance_metrics_tracking(self):
        """测试性能指标跟踪"""
        manager = UnifiedCacheManager()

        # Initial metrics
        assert manager.performance_metrics["total_requests"] == 0

        # Perform some operations
        manager.get("nonexistent")  # Miss
        manager.set("key1", "value1")  # Set
        manager.get("key1")  # Hit

        # Check that metrics were updated
        assert manager.performance_metrics["total_requests"] >= 2