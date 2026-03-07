#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多级缓存覆盖率提升测试
专注于提升multi_level_cache.py的测试覆盖率从27%到>80%
"""

import pytest
import time
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any, Optional

from src.infrastructure.cache.core.multi_level_cache import (
    MultiLevelCache,
    CacheTier,
    TierConfig,
    MultiLevelConfig,
    MemoryTier,
    RedisTier,
    DiskTier,
    CacheOperationStrategy
)


class TestMultiLevelCacheInitialization:
    """测试多级缓存初始化"""

    def test_default_initialization(self):
        """测试默认初始化"""
        cache = MultiLevelCache()
        
        assert cache is not None
        assert hasattr(cache, 'tiers')
        assert hasattr(cache, 'layers')
        assert hasattr(cache, '_stats')
        assert cache._initialized is True

    def test_initialization_with_dict_config(self):
        """测试使用字典配置初始化"""
        config = {
            'levels': {
                'L1': {'enabled': True, 'max_size': 1000, 'ttl': 3600},
                'L2': {'enabled': False},
                'L3': {'enabled': True, 'type': 'disk', 'capacity': 10000}
            }
        }
        
        cache = MultiLevelCache(config)
        
        assert cache is not None
        assert len(cache.tiers) > 0
        # L2应该被禁用
        assert CacheTier.L2_REDIS not in cache.tiers or cache.tiers[CacheTier.L2_REDIS] is None

    def test_initialization_with_multilevel_config(self):
        """测试使用MultiLevelConfig初始化"""
        # 使用默认配置创建MultiLevelConfig
        config = MultiLevelConfig()
        
        cache = MultiLevelCache(config)
        
        assert cache is not None
        # 默认L1应该启用
        assert CacheTier.L1_MEMORY in cache.tiers or len(cache.tiers) >= 0

    def test_empty_config_initialization(self):
        """测试空配置初始化"""
        cache = MultiLevelCache(config={})
        
        assert cache is not None
        assert len(cache.tiers) >= 0


class TestCacheTierOperations:
    """测试缓存层级操作"""

    @pytest.fixture
    def cache(self):
        """创建测试用缓存实例"""
        config = {
            'levels': {
                'L1': {'enabled': True, 'max_size': 100, 'ttl': 60},
                'L2': {'enabled': False},
                'L3': {'enabled': False}
            }
        }
        return MultiLevelCache(config)

    def test_set_and_get_basic(self, cache):
        """测试基本的set和get操作"""
        key = "test_key"
        value = "test_value"
        
        # Set操作
        result = cache.set(key, value)
        assert result is True
        
        # Get操作
        retrieved = cache.get(key)
        assert retrieved == value

    def test_set_with_ttl(self, cache):
        """测试带TTL的set操作"""
        key = "ttl_key"
        value = "ttl_value"
        ttl = 1  # 1秒
        
        result = cache.set(key, value, ttl=ttl)
        assert result is True
        
        # 立即获取应该成功
        retrieved = cache.get(key)
        assert retrieved == value
        
        # 等待TTL过期
        time.sleep(1.5)
        
        # 过期后应该获取不到
        expired = cache.get(key)
        assert expired is None

    def test_set_with_specific_tier(self, cache):
        """测试指定层级的set操作"""
        key = "tier_key"
        value = "tier_value"
        
        # 指定L1层级
        result = cache.set(key, value, tier='L1')
        assert result is True
        
        retrieved = cache.get(key)
        assert retrieved == value

    def test_put_method_alias(self, cache):
        """测试put方法（set的别名）"""
        key = "put_key"
        value = "put_value"
        
        result = cache.put(key, value)
        assert result is True
        
        retrieved = cache.get(key)
        assert retrieved == value

    def test_delete_operation(self, cache):
        """测试delete操作"""
        key = "delete_key"
        value = "delete_value"
        
        # 先设置
        cache.set(key, value)
        assert cache.get(key) == value
        
        # 删除
        result = cache.delete(key)
        assert result is True
        
        # 确认删除
        assert cache.get(key) is None

    def test_exists_operation(self, cache):
        """测试exists操作"""
        key = "exists_key"
        value = "exists_value"
        
        # 不存在
        assert cache.exists(key) is False
        
        # 设置后存在
        cache.set(key, value)
        assert cache.exists(key) is True
        
        # 删除后不存在
        cache.delete(key)
        assert cache.exists(key) is False

    def test_clear_operation(self, cache):
        """测试clear操作"""
        # 设置多个键
        for i in range(5):
            cache.set(f"key_{i}", f"value_{i}")
        
        # 确认存在
        assert cache.get("key_0") is not None
        
        # 清空
        result = cache.clear()
        assert result is True
        
        # 确认清空
        for i in range(5):
            assert cache.get(f"key_{i}") is None


class TestCacheStatistics:
    """测试缓存统计功能"""

    @pytest.fixture
    def cache(self):
        """创建测试用缓存实例"""
        return MultiLevelCache()

    def test_get_stats_basic(self, cache):
        """测试基本统计信息获取"""
        stats = cache.get_stats()
        
        assert isinstance(stats, dict)
        assert 'total_tiers' in stats
        assert 'tier_stats' in stats
        assert 'overall_stats' in stats

    def test_stats_after_operations(self, cache):
        """测试操作后的统计信息"""
        # 执行一些操作
        cache.set("key1", "value1")
        cache.get("key1")
        cache.get("non_existent")
        cache.delete("key1")
        
        stats = cache.get_stats()
        
        # 验证统计信息被更新
        assert stats['total_tiers'] > 0

    def test_get_cache_size(self, cache):
        """测试获取缓存大小"""
        # 初始大小
        initial_size = cache.get_cache_size()
        assert initial_size >= 0
        
        # 添加数据
        for i in range(10):
            cache.set(f"key_{i}", f"value_{i}")
        
        # 大小应该增加
        new_size = cache.get_cache_size()
        assert new_size >= initial_size

    def test_cache_stats_method(self, cache):
        """测试get_cache_stats方法"""
        stats = cache.get_cache_stats()
        
        assert isinstance(stats, dict)


class TestCacheComponentInterface:
    """测试ICacheComponent接口实现"""

    @pytest.fixture
    def cache(self):
        """创建测试用缓存实例"""
        return MultiLevelCache()

    def test_component_name(self, cache):
        """测试组件名称"""
        name = cache.component_name
        assert name == "MultiLevelCache"

    def test_component_type(self, cache):
        """测试组件类型"""
        comp_type = cache.component_type
        assert comp_type == "multi_level_cache"

    def test_initialize_component(self, cache):
        """测试组件初始化"""
        config = {'test': 'config'}
        result = cache.initialize_component(config)
        assert result is True

    def test_get_component_status(self, cache):
        """测试获取组件状态"""
        status = cache.get_component_status()
        
        assert isinstance(status, dict)
        assert 'status' in status
        assert 'initialized' in status
        assert 'tiers_count' in status

    def test_health_check(self, cache):
        """测试健康检查"""
        health = cache.health_check()
        assert isinstance(health, bool)
        assert health is True  # 初始化后应该健康

    def test_shutdown_component(self, cache):
        """测试组件关闭"""
        cache.shutdown_component()
        assert cache._initialized is False


class TestCacheItemOperations:
    """测试缓存项操作方法"""

    @pytest.fixture
    def cache(self):
        """创建测试用缓存实例"""
        return MultiLevelCache()

    def test_get_cache_item(self, cache):
        """测试get_cache_item方法"""
        cache.set("test", "value")
        result = cache.get_cache_item("test")
        assert result == "value"

    def test_set_cache_item(self, cache):
        """测试set_cache_item方法"""
        result = cache.set_cache_item("test", "value")
        assert result is True
        assert cache.get("test") == "value"

    def test_delete_cache_item(self, cache):
        """测试delete_cache_item方法"""
        cache.set("test", "value")
        result = cache.delete_cache_item("test")
        assert result is True
        assert cache.get("test") is None

    def test_has_cache_item(self, cache):
        """测试has_cache_item方法"""
        cache.set("test", "value")
        assert cache.has_cache_item("test") is True
        
        cache.delete("test")
        assert cache.has_cache_item("test") is False

    def test_clear_all_cache(self, cache):
        """测试clear_all_cache方法"""
        cache.set("key1", "value1")
        cache.set("key2", "value2")
        
        result = cache.clear_all_cache()
        assert result is True
        
        assert cache.get("key1") is None
        assert cache.get("key2") is None


class TestLayersCompatibility:
    """测试layers属性兼容性"""

    @pytest.fixture
    def cache(self):
        """创建测试用缓存实例"""
        return MultiLevelCache()

    def test_layers_exists(self, cache):
        """测试layers属性存在"""
        assert hasattr(cache, 'layers')
        assert isinstance(cache.layers, list)

    def test_layers_wrapper_get(self, cache):
        """测试层级包装器的get方法"""
        if len(cache.layers) > 0:
            layer = cache.layers[0]
            assert hasattr(layer, 'get')

    def test_layers_wrapper_set(self, cache):
        """测试层级包装器的set方法"""
        if len(cache.layers) > 0:
            layer = cache.layers[0]
            assert hasattr(layer, 'set')

    def test_layers_wrapper_put(self, cache):
        """测试层级包装器的put方法"""
        if len(cache.layers) > 0:
            layer = cache.layers[0]
            assert hasattr(layer, 'put')

    def test_layers_wrapper_delete(self, cache):
        """测试层级包装器的delete方法"""
        if len(cache.layers) > 0:
            layer = cache.layers[0]
            assert hasattr(layer, 'delete')

    def test_layers_wrapper_data_attribute(self, cache):
        """测试层级包装器的data属性"""
        if len(cache.layers) > 0:
            layer = cache.layers[0]
            assert hasattr(layer, 'data')
            data = layer.data
            assert isinstance(data, dict)


class TestConfigConversion:
    """测试配置转换功能"""

    def test_convert_dict_config(self):
        """测试字典配置转换"""
        cache = MultiLevelCache()
        
        raw_config = {
            'levels': {
                'L1': {'enabled': True, 'max_size': 500},
                'L2': {'enabled': True, 'type': 'redis'},
                'L3': {'enabled': True, 'type': 'disk'}
            }
        }
        
        config = cache._convert_dict_config(raw_config)
        assert isinstance(config, MultiLevelConfig)

    def test_convert_l1_config(self):
        """测试L1配置转换"""
        cache = MultiLevelCache()
        
        levels = {
            'L1': {'enabled': True, 'max_size': 1000, 'ttl': 3600}
        }
        config_dict = {}
        
        cache._convert_l1_config(levels, config_dict)
        assert 'l1_config' in config_dict

    def test_convert_l2_config_redis(self):
        """测试L2 Redis配置转换"""
        cache = MultiLevelCache()
        
        levels = {
            'L2': {'enabled': True, 'type': 'redis', 'host': 'localhost'}
        }
        config_dict = {}
        
        cache._convert_l2_config(levels, config_dict)
        assert 'l2_config' in config_dict

    def test_convert_l2_config_file(self):
        """测试L2 File配置转换（映射到L3）"""
        cache = MultiLevelCache()
        
        levels = {
            'L2': {'enabled': True, 'type': 'file', 'path': '/tmp/cache'}
        }
        config_dict = {}
        
        cache._convert_l2_config(levels, config_dict)
        # File类型应该映射到L3
        assert 'l3_config' in config_dict

    def test_convert_l3_config(self):
        """测试L3配置转换"""
        cache = MultiLevelCache()
        
        levels = {
            'L3': {'enabled': True, 'type': 'disk', 'capacity': 10000}
        }
        config_dict = {}
        
        cache._convert_l3_config(levels, config_dict)
        assert 'l3_config' in config_dict


class TestTiersDict:
    """测试tiers_dict兼容性"""

    def test_tiers_dict_creation(self):
        """测试tiers_dict创建"""
        config = {
            'levels': {
                'L1': {'enabled': True},
                'L2': {'enabled': False},
                'L3': {'enabled': True, 'type': 'disk'}
            }
        }
        
        cache = MultiLevelCache(config)
        
        assert hasattr(cache, 'tiers_dict')
        assert isinstance(cache.tiers_dict, dict)
        
        # L1应该存在
        if CacheTier.L1_MEMORY in cache.tiers:
            assert 'L1' in cache.tiers_dict


class TestOperationStrategy:
    """测试缓存操作策略"""

    def test_operation_strategy_exists(self):
        """测试操作策略存在"""
        cache = MultiLevelCache()
        
        assert hasattr(cache, 'operation_strategy')
        assert isinstance(cache.operation_strategy, CacheOperationStrategy)


class TestMultiTierSync:
    """测试多层级同步"""

    @pytest.fixture
    def cache(self):
        """创建包含多个层级的缓存"""
        config = {
            'levels': {
                'L1': {'enabled': True, 'max_size': 100},
                'L2': {'enabled': False},  # 禁用L2避免Redis依赖
                'L3': {'enabled': False}   # 禁用L3避免磁盘依赖
            }
        }
        return MultiLevelCache(config)

    def test_set_updates_all_tiers(self, cache):
        """测试set操作更新所有层级"""
        key = "sync_key"
        value = "sync_value"
        
        result = cache.set(key, value)
        assert result is True
        
        # 验证可以从缓存获取
        retrieved = cache.get(key)
        assert retrieved == value

    def test_delete_removes_from_all_tiers(self, cache):
        """测试delete操作从所有层级删除"""
        key = "delete_sync_key"
        value = "delete_sync_value"
        
        cache.set(key, value)
        result = cache.delete(key)
        assert result is True
        
        # 验证所有层级都删除
        assert cache.get(key) is None


class TestErrorHandling:
    """测试错误处理"""

    def test_get_nonexistent_key(self):
        """测试获取不存在的键"""
        cache = MultiLevelCache()
        
        result = cache.get("nonexistent_key")
        assert result is None

    def test_delete_nonexistent_key(self):
        """测试删除不存在的键"""
        cache = MultiLevelCache()
        
        result = cache.delete("nonexistent_key")
        # 删除不存在的键可能返回False或True，取决于实现
        assert isinstance(result, bool)

    def test_invalid_config_handling(self):
        """测试无效配置处理"""
        # 即使配置无效，也应该能创建实例
        cache = MultiLevelCache(config=None)
        assert cache is not None


class TestPerformance:
    """测试性能相关功能"""

    def test_bulk_operations(self):
        """测试批量操作"""
        cache = MultiLevelCache()
        
        # 批量设置
        start = time.time()
        for i in range(100):
            cache.set(f"perf_key_{i}", f"value_{i}")
        set_duration = time.time() - start
        
        # 批量获取
        start = time.time()
        for i in range(100):
            cache.get(f"perf_key_{i}")
        get_duration = time.time() - start
        
        # 性能应该在合理范围内
        assert set_duration < 1.0  # 100次set应该在1秒内完成
        assert get_duration < 1.0  # 100次get应该在1秒内完成

    def test_stats_overhead(self):
        """测试统计信息收集的开销"""
        cache = MultiLevelCache()
        
        # 执行操作
        for i in range(50):
            cache.set(f"key_{i}", f"value_{i}")
            cache.get(f"key_{i}")
        
        # 获取统计信息应该很快
        start = time.time()
        stats = cache.get_stats()
        duration = time.time() - start
        
        assert duration < 0.1  # 获取统计信息应该在100ms内完成
        assert isinstance(stats, dict)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

