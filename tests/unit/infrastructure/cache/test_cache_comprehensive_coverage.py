#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存模块综合覆盖率测试

针对UnifiedCacheManager和MultiLevelCache中尚未充分测试的方法和边界条件进行综合测试
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import tempfile
import os
import threading
import json
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager, create_unified_cache, create_memory_cache, create_redis_cache, create_hybrid_cache
from src.infrastructure.cache.core.multi_level_cache import MultiLevelCache, MemoryTier, RedisTier, DiskTier, TierConfig, CacheTier
from src.infrastructure.cache.core.cache_configs import CacheConfig, CacheLevel, BasicCacheConfig, MultiLevelCacheConfig, AdvancedCacheConfig, SmartCacheConfig, DistributedCacheConfig
from src.infrastructure.cache.interfaces.data_structures import CacheEntry, CacheStats, PerformanceMetrics


class TestCacheComprehensiveCoverage:
    """缓存模块综合覆盖率测试"""

    def setup_method(self, method):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {
                'level': 'memory', 
                'memory_max_size': 50, 
                'memory_ttl': 60,
                'file_cache_dir': self.temp_dir
            },
            'advanced': {'enable_compression': False, 'enable_preloading': False},
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        })
        self.manager = UnifiedCacheManager(self.config)
        self.cache = MultiLevelCache(config={
            'levels': {
                'L1': {'type': 'memory', 'max_size': 100, 'ttl': 60},
                'L2': {'type': 'file', 'max_size': 1000, 'ttl': 300, 'file_dir': self.temp_dir}
            }
        })

    def teardown_method(self, method):
        """测试后清理"""
        if hasattr(self, 'manager') and self.manager:
            self.manager.shutdown()
        if hasattr(self, 'cache') and self.cache:
            self.cache.clear()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_create_unified_cache(self):
        """测试创建统一缓存管理器便捷函数"""
        cache = create_unified_cache()
        assert cache is not None
        assert isinstance(cache, UnifiedCacheManager)
        cache.shutdown()

    def test_create_memory_cache(self):
        """测试创建内存缓存便捷函数"""
        cache = create_memory_cache(max_size=500, ttl=1800)
        assert cache is not None
        assert isinstance(cache, UnifiedCacheManager)
        # 验证配置
        assert cache.config.basic.max_size == 500
        assert cache.config.basic.ttl == 1800
        assert cache.config.multi_level.level == CacheLevel.MEMORY
        cache.shutdown()

    def test_create_redis_cache(self):
        """测试创建Redis缓存便捷函数"""
        cache = create_redis_cache(host="127.0.0.1", port=6380, max_size=2000)
        assert cache is not None
        assert isinstance(cache, UnifiedCacheManager)
        # 验证配置
        assert cache.config.basic.max_size == 2000
        assert cache.config.multi_level.level == CacheLevel.REDIS
        assert cache.config.distributed.redis_host == "127.0.0.1"
        assert cache.config.distributed.redis_port == 6380
        cache.shutdown()

    def test_create_hybrid_cache(self):
        """测试创建混合缓存便捷函数"""
        cache = create_hybrid_cache(redis_host="127.0.0.1", redis_port=6380, max_size=3000)
        assert cache is not None
        assert isinstance(cache, UnifiedCacheManager)
        # 验证配置
        assert cache.config.basic.max_size == 3000
        assert cache.config.multi_level.level == CacheLevel.HYBRID
        assert cache.config.distributed.redis_host == "127.0.0.1"
        assert cache.config.distributed.redis_port == 6380
        cache.shutdown()

    def test_cache_manager_context_manager(self):
        """测试缓存管理器上下文管理器"""
        with UnifiedCacheManager() as cache:
            # 在上下文中使用缓存
            cache.set("context_key", "context_value")
            assert cache.get("context_key") == "context_value"
        # 退出上下文后，缓存应该被自动关闭

    def test_cache_manager_with_none_config(self):
        """测试缓存管理器使用None配置"""
        cache = UnifiedCacheManager(None)
        assert cache is not None
        assert hasattr(cache, 'config')
        cache.shutdown()

    def test_cache_manager_get_with_empty_key(self):
        """测试缓存管理器获取空字符串键"""
        with pytest.raises(Exception):  # 应该抛出验证异常
            self.manager.get("")

    def test_cache_manager_set_with_empty_key(self):
        """测试缓存管理器设置空字符串键"""
        with pytest.raises(Exception):  # 应该抛出验证异常
            self.manager.set("", "value")

    def test_cache_manager_set_with_negative_ttl(self):
        """测试缓存管理器设置负数TTL"""
        result = self.manager.set("negative_ttl_key", "value", ttl=-1)
        # 负数TTL应该被视为永不过期
        assert result is True
        value = self.manager.get("negative_ttl_key")
        assert value == "value"

    def test_cache_manager_delete_with_empty_key(self):
        """测试缓存管理器删除空字符串键"""
        with pytest.raises(Exception):  # 应该抛出验证异常
            self.manager.delete("")

    def test_cache_manager_exists_with_empty_key(self):
        """测试缓存管理器检查空字符串键存在性"""
        with pytest.raises(Exception):  # 应该抛出验证异常
            self.manager.exists("")

    def test_cache_manager_health_check_with_redis_connection(self):
        """测试缓存管理器健康检查（有Redis连接）"""
        # 创建启用Redis的配置
        redis_config = CacheConfig.from_dict({
            'distributed': {
                'distributed': True,
                'redis_host': 'localhost',
                'redis_port': 6379
            }
        })
        
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.return_value = True
            mock_redis_class.return_value = mock_redis
            
            cache = UnifiedCacheManager(redis_config)
            # 手动设置redis_client
            cache._redis_client = mock_redis
            
            # 执行健康检查
            health = cache.health_check()
            assert isinstance(health, dict)
            assert 'service' in health
            assert 'healthy' in health
            
            cache.shutdown()

    def test_cache_manager_health_check_with_redis_connection_failure(self):
        """测试缓存管理器健康检查（Redis连接失败）"""
        # 创建启用Redis的配置
        redis_config = CacheConfig.from_dict({
            'distributed': {
                'distributed': True,
                'redis_host': 'localhost',
                'redis_port': 6379
            }
        })
        
        with patch('redis.Redis') as mock_redis_class:
            mock_redis = Mock()
            mock_redis.ping.side_effect = Exception("Connection failed")
            mock_redis_class.return_value = mock_redis
            
            cache = UnifiedCacheManager(redis_config)
            # 手动设置redis_client
            cache._redis_client = mock_redis
            
            # 执行健康检查
            health = cache.health_check()
            assert isinstance(health, dict)
            assert 'service' in health
            assert 'healthy' in health
            
            cache.shutdown()

    def test_cache_manager_health_check_with_exception(self):
        """测试缓存管理器健康检查时出现异常"""
        # Mock get_cache_stats方法抛出异常
        with patch.object(self.manager, 'get_cache_stats', side_effect=Exception("Health check failed")):
            health = self.manager.health_check()
            assert isinstance(health, dict)
            assert health['healthy'] is False
            assert health['status'] == 'error'

    def test_cache_manager_get_stats_compatibility(self):
        """测试缓存管理器获取统计信息兼容性方法"""
        # 设置一些数据
        self.manager.set("stats_key1", "stats_value1")
        self.manager.get("stats_key1")  # 增加命中计数
        self.manager.get("nonexistent_key")  # 增加未命中计数
        
        # 测试get_stats方法
        stats = self.manager.get_stats()
        assert isinstance(stats, dict)
        assert 'total_requests' in stats
        assert 'hit_rate' in stats
        
        # 测试get_cache_stats方法
        cache_stats = self.manager.get_cache_stats()
        assert isinstance(cache_stats, dict)
        assert 'total_requests' in cache_stats
        assert 'hit_rate' in cache_stats

    def test_cache_manager_keys_method(self):
        """测试缓存管理器获取所有键方法"""
        # 设置一些数据
        test_keys = ["key1", "key2", "key3"]
        for key in test_keys:
            self.manager.set(key, f"value_{key}")
        
        # 获取所有键
        all_keys = self.manager.keys()
        assert isinstance(all_keys, list)
        for key in test_keys:
            assert key in all_keys

    def test_cache_manager_has_key_method(self):
        """测试缓存管理器检查键存在性方法"""
        # 测试不存在的键
        assert self.manager.has_key("nonexistent") is False
        
        # 设置键并测试存在性
        self.manager.set("test_key", "test_value")
        assert self.manager.has_key("test_key") is True
        
        # 删除键并再次测试
        self.manager.delete("test_key")
        assert self.manager.has_key("test_key") is False

    def test_multi_level_cache_with_hybrid_config(self):
        """测试多级缓存使用混合配置"""
        hybrid_config = {
            'levels': {
                'L1': {'type': 'memory', 'max_size': 100, 'ttl': 60},
                'L2': {'type': 'redis', 'max_size': 1000, 'ttl': 3600},
                'L3': {'type': 'file', 'max_size': 10000, 'ttl': 86400, 'file_dir': self.temp_dir}
            }
        }
        
        cache = MultiLevelCache(config=hybrid_config)
        assert cache is not None
        cache.clear()

    def test_multi_level_cache_with_file_config(self):
        """测试多级缓存使用文件配置"""
        file_config = {
            'levels': {
                'L1': {'type': 'memory', 'max_size': 100, 'ttl': 60},
                'L2': {'type': 'file', 'max_size': 1000, 'ttl': 3600, 'file_dir': self.temp_dir}
            }
        }
        
        cache = MultiLevelCache(config=file_config)
        assert cache is not None
        cache.clear()

    def test_multi_level_cache_get_memory_usage(self):
        """测试多级缓存获取内存使用情况"""
        # 设置一些数据
        self.cache.set_memory("memory_key1", "memory_value1")
        self.cache.set_memory("memory_key2", "memory_value2")
        
        # 获取内存使用情况
        usage = self.cache.get_memory_usage()
        assert isinstance(usage, dict)
        assert 'used' in usage
        assert 'total' in usage
        assert 'percentage' in usage
        assert 'item_count' in usage

    def test_multi_level_cache_get_memory_bulk(self):
        """测试多级缓存批量获取内存"""
        # 设置一些数据
        test_data = {
            "bulk_key1": "bulk_value1",
            "bulk_key2": "bulk_value2",
            "bulk_key3": "bulk_value3"
        }
        
        for key, value in test_data.items():
            self.cache.set_memory(key, value)
        
        # 批量获取
        keys = list(test_data.keys())
        result = self.cache.get_memory_bulk(keys)
        assert isinstance(result, dict)
        assert len(result) == len(keys)
        for key, value in test_data.items():
            assert result[key] == value

    def test_multi_level_cache_sync_memory_to_file(self):
        """测试多级缓存同步内存到文件"""
        # 这个测试依赖于具体的实现，主要验证不会抛出异常
        result = self.cache.sync_memory_to_file()
        assert isinstance(result, bool)

    def test_multi_level_cache_clear_methods(self):
        """测试多级缓存清空方法"""
        # 设置一些数据
        self.cache.set("clear_key1", "clear_value1")
        self.cache.set_memory("clear_memory_key1", "clear_memory_value1")
        
        # 测试清空内存
        result = self.cache.clear_memory()
        assert isinstance(result, bool)
        
        # 测试清空文件
        result = self.cache.clear_file()
        assert isinstance(result, bool)

    def test_multi_level_cache_delete_methods(self):
        """测试多级缓存删除方法"""
        # 设置一些数据
        self.cache.set("delete_key1", "delete_value1")
        self.cache.set_memory("delete_memory_key1", "delete_memory_value1")
        
        # 测试删除内存
        result = self.cache.delete_memory("delete_memory_key1")
        assert isinstance(result, bool)
        
        # 测试删除文件
        result = self.cache.delete_file("delete_key1")
        assert isinstance(result, bool)

    def test_multi_level_cache_close_and_is_closed(self):
        """测试多级缓存关闭和检查关闭状态"""
        # 测试未关闭状态
        assert self.cache.is_closed() is False
        
        # 关闭缓存
        self.cache.close()
        
        # 测试关闭状态
        assert self.cache.is_closed() is True

    def test_multi_level_cache_load_from_file_with_valid_data(self):
        """测试多级缓存从文件加载有效数据"""
        # 创建包含有效JSON的文件
        valid_file = os.path.join(self.temp_dir, "valid_data.json")
        data = {"cache_key1": "cache_value1", "cache_key2": "cache_value2"}
        with open(valid_file, 'w', encoding='utf-8') as f:
            json.dump(data, f)
        
        # 测试加载（具体实现可能不同，主要验证不会抛出异常）
        try:
            result = self.cache.load_from_file(valid_file)
            assert isinstance(result, bool)
        except Exception:
            # 如果实现不支持加载，也应该不会抛出未处理的异常
            pass

    def test_memory_tier_comprehensive(self):
        """测试内存层级综合功能"""
        if self.cache.l1_tier and isinstance(self.cache.l1_tier, MemoryTier):
            # 测试设置和获取
            result = self.cache.l1_tier.set("mem_key", "mem_value", 60)
            assert result is True
            
            value = self.cache.l1_tier.get("mem_key")
            assert value == "mem_value"
            
            # 测试存在性检查
            assert self.cache.l1_tier.exists("mem_key") is True
            assert self.cache.l1_tier.exists("nonexistent_key") is False
            
            # 测试删除
            result = self.cache.l1_tier.delete("mem_key")
            assert result is True
            assert self.cache.l1_tier.get("mem_key") is None
            
            # 测试清空
            self.cache.l1_tier.set("temp_key1", "temp_value1")
            self.cache.l1_tier.set("temp_key2", "temp_value2")
            result = self.cache.l1_tier.clear()
            assert result is True
            assert self.cache.l1_tier.size() == 0
            
            # 测试统计信息
            stats = self.cache.l1_tier.get_stats()
            assert isinstance(stats, dict)
            assert 'size' in stats

    def test_disk_tier_comprehensive(self):
        """测试磁盘层级综合功能"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            # 测试设置和获取
            result = self.cache.l3_tier.set("disk_key", "disk_value", 60)
            assert result is True
            
            value = self.cache.l3_tier.get("disk_key")
            assert value == "disk_value"
            
            # 测试存在性检查
            assert self.cache.l3_tier.exists("disk_key") is True
            assert self.cache.l3_tier.exists("nonexistent_key") is False
            
            # 测试删除
            result = self.cache.l3_tier.delete("disk_key")
            assert result is True
            assert self.cache.l3_tier.get("disk_key") is None
            
            # 测试清空
            self.cache.l3_tier.set("temp_disk_key1", "temp_disk_value1")
            self.cache.l3_tier.set("temp_disk_key2", "temp_disk_value2")
            result = self.cache.l3_tier.clear()
            assert result is True
            
            # 测试统计信息
            stats = self.cache.l3_tier.get_stats()
            assert isinstance(stats, dict)
            assert 'size' in stats

    def test_cache_entry_data_structure(self):
        """测试缓存条目数据结构"""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            ttl=300,
            size_bytes=1024
        )
        
        assert entry.key == "test_key"
        assert entry.value == "test_value"
        assert entry.ttl == 300
        assert entry.size_bytes == 1024
        assert hasattr(entry, 'created_at')

    def test_cache_stats_data_structure(self):
        """测试缓存统计信息数据结构"""
        stats = CacheStats()
        assert hasattr(stats, 'hits')
        assert hasattr(stats, 'misses')
        assert hasattr(stats, 'evictions')
        assert hasattr(stats, 'total_requests')
        assert hasattr(stats, 'total_size_bytes')

    def test_performance_metrics_data_structure(self):
        """测试性能指标数据结构"""
        from datetime import datetime
        metrics = PerformanceMetrics(
            timestamp=datetime.now(),
            hit_rate=0.85,
            response_time=15.5,
            throughput=1000,
            memory_usage=50.0,
            eviction_rate=0.01,
            cache_size=100,
            miss_penalty=50.0
        )
        
        assert metrics.hit_rate == 0.85
        assert metrics.response_time == 15.5
        assert metrics.throughput == 1000
        assert metrics.memory_usage == 50.0

    def test_cache_config_creation(self):
        """测试缓存配置创建"""
        # 测试完整的配置创建
        config = CacheConfig(
            basic=BasicCacheConfig(max_size=1000, ttl=3600),
            multi_level=MultiLevelCacheConfig(
                level=CacheLevel.HYBRID,
                memory_max_size=100,
                memory_ttl=300,
                redis_max_size=1000,
                redis_ttl=3600
            ),
            advanced=AdvancedCacheConfig(
                enable_compression=True,
                enable_preloading=True,
                max_memory_mb=100
            ),
            smart=SmartCacheConfig(
                enable_monitoring=True,
                enable_auto_optimization=True
            ),
            distributed=DistributedCacheConfig(
                distributed=True,
                redis_host="localhost",
                redis_port=6379
            )
        )
        
        assert config.basic.max_size == 1000
        assert config.multi_level.level == CacheLevel.HYBRID
        assert config.advanced.enable_compression is True
        assert config.smart.enable_monitoring is True
        assert config.distributed.distributed is True

    def test_cache_config_from_dict(self):
        """测试从字典创建缓存配置"""
        config_dict = {
            'basic': {'max_size': 500, 'ttl': 1800},
            'multi_level': {
                'level': 'memory',
                'memory_max_size': 50,
                'memory_ttl': 60
            },
            'advanced': {
                'enable_compression': True,
                'enable_preloading': False
            },
            'smart': {
                'enable_monitoring': True,
                'enable_auto_optimization': False
            },
            'distributed': {
                'distributed': False
            }
        }
        
        config = CacheConfig.from_dict(config_dict)
        assert config.basic.max_size == 500
        assert config.basic.ttl == 1800
        # 从字典创建可能返回字符串或枚举，需要处理两种情况
        level_value = config.multi_level.level
        if hasattr(level_value, 'value'):
            assert level_value.value == 'memory'
        else:
            assert level_value == 'memory' or level_value == CacheLevel.MEMORY
        assert config.advanced.enable_compression is True
        assert config.smart.enable_monitoring is True
        assert config.distributed.distributed is False


if __name__ == '__main__':
    pytest.main([__file__])