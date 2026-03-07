#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多级缓存增强测试

专门针对MultiLevelCache中未充分测试的方法和边界条件进行测试
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import tempfile
import os
import threading
import json
import pickle
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.cache.core.multi_level_cache import (
    MultiLevelCache, MemoryTier, RedisTier, DiskTier, 
    TierConfig, CacheTier, CacheOperationStrategy, 
    CachePerformanceOptimizer, BaseCacheTier
)
from src.infrastructure.cache.core.cache_configs import CacheLevel


class TestMultiLevelCacheEnhancement:
    """多级缓存增强测试"""

    def setup_method(self, method):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            'levels': {
                'L1': {'type': 'memory', 'max_size': 100, 'ttl': 60},
                'L2': {'type': 'file', 'max_size': 1000, 'ttl': 300, 'file_dir': self.temp_dir}
            }
        }
        self.cache = MultiLevelCache(config=self.config)

    def teardown_method(self, method):
        """测试后清理"""
        if hasattr(self, 'cache') and self.cache:
            self.cache.clear()
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_base_cache_tier_abstract_methods(self):
        """测试基础缓存层级抽象方法"""
        # BaseCacheTier不是真正的抽象类，但未实现的方法会抛出NotImplementedError
        base_tier = BaseCacheTier(TierConfig(tier=CacheTier.L1_MEMORY))
        
        # 测试未实现的方法抛出NotImplementedError
        with pytest.raises(NotImplementedError):
            base_tier.get("test_key")
        
        with pytest.raises(NotImplementedError):
            base_tier.set("test_key", "test_value")
            
        with pytest.raises(NotImplementedError):
            base_tier.delete("test_key")
            
        with pytest.raises(NotImplementedError):
            base_tier.clear()
            
        with pytest.raises(NotImplementedError):
            base_tier.get_stats()

    def test_base_cache_tier_concrete_methods(self):
        """测试基础缓存层级具体方法"""
        # 创建一个具体的子类来测试基础方法
        class ConcreteCacheTier(BaseCacheTier):
            def get(self, key):
                return "test_value"
            
            def set(self, key, value, ttl=None):
                return True
                
            def delete(self, key):
                return True
                
            def clear(self):
                return True
                
            def get_stats(self):
                return {"size": 1}
        
        config = TierConfig(tier=CacheTier.L1_MEMORY)
        tier = ConcreteCacheTier(config)
        
        # 测试exists方法
        assert tier.exists("test_key") is True
        
        # 测试size方法
        assert tier.size() == 1
        
        # 测试_estimate_size方法
        size = tier._estimate_size("test_string")
        assert isinstance(size, int)
        assert size > 0

    def test_memory_tier_is_expired_with_missing_key(self):
        """测试内存层级检查过期（键不存在）"""
        if self.cache.l1_tier and isinstance(self.cache.l1_tier, MemoryTier):
            result = self.cache.l1_tier._is_expired("nonexistent_key")
            assert result is True

    def test_memory_tier_remove_expired(self):
        """测试内存层级移除过期项"""
        if self.cache.l1_tier and isinstance(self.cache.l1_tier, MemoryTier):
            # 添加一个项
            self.cache.l1_tier.cache["test_key"] = "test_value"
            self.cache.l1_tier.metadata["test_key"] = {
                "created_at": time.time() - 100,  # 100秒前创建
                "ttl": 1,  # 1秒TTL，已过期
                "size": 100
            }
            self.cache.l1_tier.access_times["test_key"] = time.time() - 100
            
            # 移除过期项
            self.cache.l1_tier._remove_expired("test_key")
            
            # 验证项已被移除
            assert "test_key" not in self.cache.l1_tier.cache
            assert "test_key" not in self.cache.l1_tier.metadata
            assert "test_key" not in self.cache.l1_tier.access_times

    def test_memory_tier_evict_oldest_empty_cache(self):
        """测试内存层级驱逐最旧项（空缓存）"""
        if self.cache.l1_tier and isinstance(self.cache.l1_tier, MemoryTier):
            self.cache.l1_tier.cache.clear()
            # 应该不会抛出异常
            self.cache.l1_tier._evict_oldest()

    def test_redis_tier_get_with_none_client(self):
        """测试Redis层级获取（无客户端）"""
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = None
        result = redis_tier.get("test_key")
        assert result is None

    def test_redis_tier_set_with_none_client(self):
        """测试Redis层级设置（无客户端）"""
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = None
        result = redis_tier.set("test_key", "test_value")
        assert result is False

    def test_redis_tier_delete_with_none_client(self):
        """测试Redis层级删除（无客户端）"""
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = None
        result = redis_tier.delete("test_key")
        assert result is False

    def test_redis_tier_exists_with_none_client(self):
        """测试Redis层级检查存在（无客户端）"""
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = None
        result = redis_tier.exists("test_key")
        assert result is False

    def test_redis_tier_clear_with_none_client(self):
        """测试Redis层级清空（无客户端）"""
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = None
        result = redis_tier.clear()
        assert result is False

    def test_redis_tier_get_stats_with_none_client(self):
        """测试Redis层级获取统计（无客户端）"""
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = None
        stats = redis_tier.get_stats()
        assert stats['status'] == 'unavailable'

    def test_redis_tier_get_with_non_bytes_value(self):
        """测试Redis层级获取非字节值"""
        mock_redis = Mock()
        mock_redis.get.return_value = "string_value"
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = mock_redis
        
        result = redis_tier.get("test_key")
        assert result == "string_value"

    def test_redis_tier_get_with_exception(self):
        """测试Redis层级获取时出现异常"""
        mock_redis = Mock()
        mock_redis.get.side_effect = Exception("Redis error")
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = mock_redis
        
        result = redis_tier.get("test_key")
        assert result is None

    def test_redis_tier_set_with_exception(self):
        """测试Redis层级设置时出现异常"""
        mock_redis = Mock()
        mock_redis.setex.side_effect = Exception("Redis error")
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = mock_redis
        
        result = redis_tier.set("test_key", "test_value")
        assert result is False

    def test_redis_tier_delete_with_exception(self):
        """测试Redis层级删除时出现异常"""
        mock_redis = Mock()
        mock_redis.delete.side_effect = Exception("Redis error")
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = mock_redis
        
        result = redis_tier.delete("test_key")
        assert result is False

    def test_redis_tier_exists_with_exception(self):
        """测试Redis层级检查存在时出现异常"""
        mock_redis = Mock()
        mock_redis.exists.side_effect = Exception("Redis error")
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = mock_redis
        
        result = redis_tier.exists("test_key")
        assert result is False

    def test_redis_tier_clear_with_exception(self):
        """测试Redis层级清空时出现异常"""
        mock_redis = Mock()
        mock_redis.flushdb.side_effect = Exception("Redis error")
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = mock_redis
        
        result = redis_tier.clear()
        assert result is False

    def test_redis_tier_get_stats_with_exception(self):
        """测试Redis层级获取统计时出现异常"""
        mock_redis = Mock()
        mock_redis.info.side_effect = Exception("Redis error")
        redis_tier = RedisTier(TierConfig(tier=CacheTier.L2_REDIS))
        redis_tier.redis_client = mock_redis
        
        stats = redis_tier.get_stats()
        assert stats['status'] == 'error'

    def test_disk_tier_key_exists_with_missing_key(self):
        """测试磁盘层级检查键存在（键不存在）"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            result = self.cache.l3_tier._key_exists("nonexistent_key")
            assert result is False

    def test_disk_tier_is_expired_with_missing_key(self):
        """测试磁盘层级检查过期（键不存在）"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            result = self.cache.l3_tier._is_expired("nonexistent_key")
            assert result is True

    def test_disk_tier_remove_key_with_missing_key(self):
        """测试磁盘层级移除键（键不存在）"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            result = self.cache.l3_tier._remove_key("nonexistent_key")
            assert result is False

    def test_disk_tier_evict_oldest_empty_cache(self):
        """测试磁盘层级驱逐最旧项（空缓存）"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            self.cache.l3_tier.metadata.clear()
            # 应该不会抛出异常
            self.cache.l3_tier._evict_oldest()

    def test_disk_tier_get_disk_usage_with_exception(self):
        """测试磁盘层级获取磁盘使用量时出现异常"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            with patch('os.path.getsize', side_effect=Exception("File error")):
                usage = self.cache.l3_tier._get_disk_usage()
                assert usage == 0.0

    def test_disk_tier_load_metadata_with_exception(self):
        """测试磁盘层级加载元数据时出现异常"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            with patch('os.path.exists', return_value=True):
                with patch('builtins.open', side_effect=Exception("File error")):
                    # 应该不会抛出异常
                    self.cache.l3_tier._load_metadata()
                    assert self.cache.l3_tier.metadata == {}

    def test_disk_tier_save_metadata_with_exception(self):
        """测试磁盘层级保存元数据时出现异常"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            with patch('builtins.open', side_effect=Exception("File error")):
                # 应该不会抛出异常
                self.cache.l3_tier._save_metadata()

    def test_disk_tier_get_with_missing_file(self):
        """测试磁盘层级获取（文件不存在）"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            # 添加元数据但不创建文件
            self.cache.l3_tier.metadata["test_key"] = {
                "created_at": time.time(),
                "ttl": 3600,
                "file_path": "/nonexistent/path"
            }
            
            result = self.cache.l3_tier.get("test_key")
            assert result is None
            # 验证元数据被清理
            assert "test_key" not in self.cache.l3_tier.metadata

    def test_disk_tier_get_with_expired_key(self):
        """测试磁盘层级获取（键已过期）"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            # 添加已过期的元数据
            self.cache.l3_tier.metadata["expired_key"] = {
                "created_at": time.time() - 100,
                "ttl": 1,  # 1秒TTL，已过期
                "file_path": "/tmp/test"
            }
            
            result = self.cache.l3_tier.get("expired_key")
            assert result is None
            # 验证元数据被清理
            assert "expired_key" not in self.cache.l3_tier.metadata

    def test_disk_tier_set_with_exception(self):
        """测试磁盘层级设置时出现异常"""
        if self.cache.l3_tier and isinstance(self.cache.l3_tier, DiskTier):
            with patch('os.makedirs', side_effect=Exception("Permission denied")):
                result = self.cache.l3_tier.set("test_key", "test_value")
                assert result is False

    def test_cache_operation_strategy_execute_get_operation(self):
        """测试缓存操作策略执行获取操作"""
        strategy = CacheOperationStrategy(self.cache)
        
        # 测试成功获取
        with patch.object(self.cache, 'l1_tier') as mock_tier:
            mock_tier.get.return_value = "test_value"
            result = strategy.execute_get_operation("test_key", "l1")
            assert result == "test_value"
        
        # 测试异常处理
        with patch.object(self.cache, 'l1_tier') as mock_tier:
            mock_tier.get.side_effect = Exception("Tier error")
            result = strategy.execute_get_operation("test_key", "l1")
            assert result is None

    def test_cache_operation_strategy_execute_set_operation(self):
        """测试缓存操作策略执行设置操作"""
        strategy = CacheOperationStrategy(self.cache)
        
        # 测试成功设置
        with patch.object(self.cache, 'l1_tier') as mock_tier:
            mock_tier.set.return_value = True
            result = strategy.execute_set_operation("test_key", "test_value", 3600, "l1")
            assert result is True
        
        # 测试异常处理
        with patch.object(self.cache, 'l1_tier') as mock_tier:
            mock_tier.set.side_effect = Exception("Tier error")
            result = strategy.execute_set_operation("test_key", "test_value", 3600, "l1")
            assert result is False

    def test_cache_operation_strategy_execute_delete_operation(self):
        """测试缓存操作策略执行删除操作"""
        strategy = CacheOperationStrategy(self.cache)
        
        # 测试成功删除
        with patch.object(self.cache, 'l1_tier') as mock_tier:
            mock_tier.delete.return_value = True
            result = strategy.execute_delete_operation("test_key", "l1")
            assert result is True
        
        # 测试异常处理
        with patch.object(self.cache, 'l1_tier') as mock_tier:
            mock_tier.delete.side_effect = Exception("Tier error")
            result = strategy.execute_delete_operation("test_key", "l1")
            assert result is False

    def test_cache_performance_optimizer(self):
        """测试缓存性能优化器"""
        optimizer = CachePerformanceOptimizer()
        result = optimizer.optimize_cache_strategy()
        assert isinstance(result, dict)
        assert result["status"] == "completed"

    def test_multi_level_cache_init_with_dict_config(self):
        """测试多级缓存使用字典配置初始化"""
        config = {
            'levels': {
                'L1': {'type': 'memory', 'max_size': 50, 'ttl': 30},
                'L2': {'type': 'redis', 'max_size': 500, 'ttl': 300, 'host': 'localhost', 'port': 6379},
                'L3': {'type': 'file', 'max_size': 5000, 'ttl': 3600, 'file_dir': self.temp_dir}
            }
        }
        
        cache = MultiLevelCache(config=config)
        assert cache is not None
        cache.clear()

    def test_multi_level_cache_init_with_none_config(self):
        """测试多级缓存使用None配置初始化"""
        cache = MultiLevelCache(config=None)
        assert cache is not None
        cache.clear()

    def test_multi_level_cache_init_with_exception(self):
        """测试多级缓存初始化时出现异常"""
        with patch('src.infrastructure.cache.core.multi_level_cache.CacheConfigProcessor.process_config', side_effect=Exception("Config error")):
            # 期望异常传播，因为MultiLevelCache没有异常处理
            with pytest.raises(Exception, match="Config error"):
                MultiLevelCache(config=self.config)

    def test_multi_level_cache_get_with_exception(self):
        """测试多级缓存获取时出现异常"""
        from src.infrastructure.cache.core.multi_level_cache import CacheTier
        
        # 设置tiers字典以正确模拟
        mock_tier = StandardMockBuilder.create_cache_mock()
        mock_tier.get.side_effect = Exception("Tier error")
        self.cache.tiers = {CacheTier.L1_MEMORY: mock_tier}
        
        # 根据实际实现，异常会直接传播
        with pytest.raises(Exception, match="Tier error"):
            self.cache.get("test_key")

    def test_multi_level_cache_set_with_invalid_key(self):
        """测试多级缓存设置无效键"""
        # 根据实际实现，只有None被拒绝，空字符串是有效的key
        result = self.cache.set(None, "test_value")
        assert result is False
        
        # 空字符串实际上是有效的key
        result_empty = self.cache.set("", "test_value")
        assert result_empty is True

    def test_multi_level_cache_set_with_invalid_tier(self):
        """测试多级缓存设置无效层级"""
        result = self.cache.set("test_key", "test_value", tier="invalid")
        assert result is False

    def test_multi_level_cache_delete_with_exception(self):
        """测试多级缓存删除时出现异常"""
        with patch.object(self.cache, 'tiers') as mock_tiers:
            mock_tier = StandardMockBuilder.create_cache_mock()
            mock_tier.delete.side_effect = Exception("Tier error")
            mock_tiers.__getitem__.return_value = mock_tier
            mock_tiers.__contains__.return_value = True
            
            result = self.cache.delete("test_key")
            # 即使一个层级失败，也应该继续处理其他层级
            assert isinstance(result, bool)

    def test_multi_level_cache_exists_with_exception(self):
        """测试多级缓存检查存在时出现异常"""
        with patch.object(self.cache, 'tiers') as mock_tiers:
            mock_tier = StandardMockBuilder.create_cache_mock()
            mock_tier.exists.side_effect = Exception("Tier error")
            mock_tiers.__getitem__.return_value = mock_tier
            mock_tiers.__contains__.return_value = True
            
            result = self.cache.exists("test_key")
            # 即使一个层级失败，也应该继续处理其他层级
            assert isinstance(result, bool)

    def test_multi_level_cache_clear_with_exception(self):
        """测试多级缓存清空时出现异常"""
        with patch.object(self.cache, 'tiers') as mock_tiers:
            mock_tier = StandardMockBuilder.create_cache_mock()
            mock_tier.clear.side_effect = Exception("Tier error")
            mock_tiers.__getitem__.return_value = mock_tier
            mock_tiers.__contains__.return_value = True
            
            result = self.cache.clear()
            # 即使一个层级失败，也应该继续处理其他层级
            assert isinstance(result, bool)

    def test_multi_level_cache_get_stats_with_exception(self):
        """测试多级缓存获取统计时出现异常"""
        with patch.object(self.cache, 'tiers') as mock_tiers:
            mock_tier = StandardMockBuilder.create_cache_mock()
            mock_tier.get_stats.side_effect = Exception("Tier error")
            mock_tiers.__getitem__.return_value = mock_tier
            mock_tiers.__contains__.return_value = True
            
            stats = self.cache.get_stats()
            # 应该仍然返回统计信息，但可能缺少某些层级的数据
            assert isinstance(stats, dict)
            assert 'tier_stats' in stats

    def test_multi_level_cache_propagate_to_faster_tiers(self):
        """测试多级缓存传播到更快层级"""
        # 测试从L3传播到L1
        with patch.object(self.cache, 'tiers') as mock_tiers:
            mock_l1_tier = Mock()
            mock_l3_tier = Mock()
            mock_tiers.__contains__.side_effect = lambda tier: tier in [CacheTier.L1_MEMORY, CacheTier.L3_DISK]
            mock_tiers.__getitem__.side_effect = lambda tier: mock_l1_tier if tier == CacheTier.L1_MEMORY else mock_l3_tier
            
            self.cache._propagate_to_faster_tiers("test_key", "test_value", CacheTier.L3_DISK)
            mock_l1_tier.set.assert_called_once_with("test_key", "test_value", ttl=300)

    def test_multi_level_cache_get_optimal_ttl(self):
        """测试多级缓存获取最优TTL"""
        ttl = self.cache._get_optimal_ttl("test_key", CacheTier.L1_MEMORY)
        assert ttl == 300  # 内存层级默认5分钟
        
        ttl = self.cache._get_optimal_ttl("test_key", CacheTier.L2_REDIS)
        assert ttl == 3600  # Redis层级默认1小时
        
        ttl = self.cache._get_optimal_ttl("test_key", CacheTier.L3_DISK)
        assert ttl == 86400  # 磁盘层级默认24小时

    def test_multi_level_cache_set_memory_with_ttl(self):
        """测试多级缓存设置带TTL的内存缓存"""
        result = self.cache.set_memory_with_ttl("test_key", "test_value", 60)
        # 结果取决于实际实现，但不应该抛出异常
        assert isinstance(result, bool)

    def test_multi_level_cache_set_memory_bulk(self):
        """测试多级缓存批量设置内存缓存"""
        data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        result = self.cache.set_memory_bulk(data)
        # 结果取决于实际实现，但不应该抛出异常
        assert isinstance(result, bool)

    def test_multi_level_cache_set_memory_compressed(self):
        """测试多级缓存设置压缩内存缓存"""
        result = self.cache.set_memory_compressed("test_key", "test_value", 60)
        # 结果取决于实际实现，但不应该抛出异常
        assert isinstance(result, bool)

    def test_multi_level_cache_set_with_promotion(self):
        """测试多级缓存设置并提升"""
        result = self.cache.set_with_promotion("test_key", "test_value", 60)
        # 结果取决于实际实现，但不应该抛出异常
        assert isinstance(result, bool)

    def test_multi_level_cache_operate_on_tier_with_none_key(self):
        """测试多级缓存在层级上操作（键为None）"""
        result = self.cache._operate_on_tier(CacheTier.L1_MEMORY, 'get', None, "default")
        # None键会被转换为空字符串，get操作返回None（空字符串键不存在）
        # 只有在tier不存在时才返回default_value
        assert result is None

    def test_multi_level_cache_operate_on_tier_with_invalid_operation(self):
        """测试多级缓存在层级上操作（无效操作）"""
        result = self.cache._operate_on_tier(CacheTier.L1_MEMORY, 'invalid', "test_key", "default")
        # 对于无效操作，应该返回默认值
        assert result == "default"

    def test_multi_level_cache_sync_memory_to_file_with_exception(self):
        """测试多级缓存同步内存到文件时出现异常"""
        result = self.cache.sync_memory_to_file()
        # 结果取决于实际实现，但不应该抛出异常
        assert isinstance(result, bool)

    def test_multi_level_cache_get_memory_bulk(self):
        """测试多级缓存批量获取内存缓存"""
        keys = ["key1", "key2", "key3"]
        result = self.cache.get_memory_bulk(keys)
        # 应该返回字典
        assert isinstance(result, dict)

    def test_multi_level_cache_get_memory_usage(self):
        """测试多级缓存获取内存使用情况"""
        result = self.cache.get_memory_usage()
        # 应该返回字典
        assert isinstance(result, dict)
        assert 'used' in result
        assert 'total' in result
        assert 'percentage' in result

    def test_multi_level_cache_close(self):
        """测试多级缓存关闭"""
        # 应该不会抛出异常
        self.cache.close()

    def test_multi_level_cache_is_closed(self):
        """测试多级缓存是否已关闭"""
        result = self.cache.is_closed()
        # 结果取决于实现
        assert isinstance(result, bool)

    def test_multi_level_cache_load_from_file_with_nonexistent_file(self):
        """测试多级缓存从文件加载（文件不存在）"""
        with pytest.raises(FileNotFoundError):
            self.cache.load_from_file("/nonexistent/file")

    def test_multi_level_cache_load_from_file_with_empty_file(self):
        """测试多级缓存从文件加载（空文件）"""
        # 创建空文件
        empty_file = os.path.join(self.temp_dir, "empty.json")
        with open(empty_file, 'w') as f:
            f.write("")
        
        # 应该不会抛出异常
        result = self.cache.load_from_file(empty_file)
        assert result is True

    def test_multi_level_cache_load_from_file_with_invalid_json(self):
        """测试多级缓存从文件加载（无效JSON）"""
        # 创建包含无效JSON的文件
        invalid_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_file, 'w') as f:
            f.write("invalid json content")
        
        with pytest.raises(ValueError):
            self.cache.load_from_file(invalid_file)

    def test_multi_level_cache_load_from_file_with_valid_json(self):
        """测试多级缓存从文件加载（有效JSON）"""
        # 创建包含有效JSON的文件
        valid_file = os.path.join(self.temp_dir, "valid.json")
        data = {"key1": "value1", "key2": "value2"}
        with open(valid_file, 'w') as f:
            json.dump(data, f)
        
        # 应该不会抛出异常
        result = self.cache.load_from_file(valid_file)
        assert result is True


if __name__ == '__main__':
    pytest.main([__file__])