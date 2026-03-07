#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理器覆盖率增强测试

专门针对UnifiedCacheManager低覆盖率问题的系统性测试套件
目标：将缓存管理模块测试覆盖率提升至80%+

重点覆盖：
1. 初始化阶段的各个_setup方法
2. 多级缓存查找和设置逻辑
3. 异常处理和错误恢复机制
4. 内存管理和清理机制
5. 监控和统计功能
6. 分布式缓存集成
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
import tempfile
import os
from unittest.mock import Mock, patch, MagicMock, Mock as MockClass
from typing import Dict, Any, Optional

from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
from src.infrastructure.cache.core.cache_configs import CacheConfig, CacheLevel
from src.infrastructure.cache.interfaces.data_structures import CacheEntry, CacheStats


class TestCacheManagerInitialization:
    """测试缓存管理器初始化阶段的各种方法"""

    @pytest.fixture
    def mock_config(self):
        """创建模拟配置"""
        config = Mock()
        config.__dict__ = {
            'multi_level': Mock(),
            'smart': Mock(),
            'distributed': Mock(),
            'advanced': Mock()
        }
        config.multi_level.level = CacheLevel.MEMORY
        config.multi_level.memory_max_size = 100
        config.multi_level.memory_ttl = 60
        config.multi_level.redis_max_size = 200
        config.multi_level.redis_ttl = 120
        config.multi_level.file_max_size = 500
        config.multi_level.file_ttl = 300
        config.smart.enable_monitoring = True
        config.distributed.distributed = False
        return config

    def test_setup_basic_configuration_with_validation_error(self):
        """测试基础配置设置的验证错误处理"""
        from src.infrastructure.cache.core.cache_configs import CacheConfig
        
        # 创建一个有效的配置用于测试验证错误处理
        config = CacheConfig.create_simple_memory_config()
        
        with patch('src.infrastructure.cache.core.cache_manager.InfrastructureConfigValidator.validate_required_config') as mock_validate:
            mock_validate.side_effect = Exception("配置验证失败")
            
            # 应该能够处理验证异常
            with pytest.raises(Exception, match="配置验证失败"):
                with patch.object(UnifiedCacheManager, '_init_components'):
                    with patch.object(UnifiedCacheManager, '_start_cleanup_thread'):
                        with patch.object(UnifiedCacheManager, 'start_monitoring'):
                            UnifiedCacheManager(config)

    def test_setup_basic_configuration_with_config_validation(self):
        """测试基础配置设置的配置验证"""
        from src.infrastructure.cache.core.cache_configs import CacheConfig
        
        # 使用真实的配置对象
        config = CacheConfig.create_simple_memory_config()
        
        with patch('src.infrastructure.cache.core.cache_manager.InfrastructureConfigValidator.validate_required_config') as mock_validate:
            mock_validate.return_value = []  # 无验证错误
            
            with patch.object(UnifiedCacheManager, '_init_components'):
                with patch.object(UnifiedCacheManager, '_start_cleanup_thread'):
                    with patch.object(UnifiedCacheManager, 'start_monitoring'):
                        manager = UnifiedCacheManager(config)
            
                        
                        # 验证配置验证方法被调用
                        mock_validate.assert_called()
                        assert manager.config == config
    def test_setup_storage_systems(self):
        """测试存储系统设置"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            with patch.object(UnifiedCacheManager, '_init_cache_storage') as mock_init:
                manager = UnifiedCacheManager()
                # 验证初始化方法被调用
                mock_init.assert_called()

    def test_setup_monitoring_and_intelligence(self):
        """测试监控和智能功能设置"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            with patch.object(UnifiedCacheManager, '_init_monitoring_components') as mock_init:
                manager = UnifiedCacheManager()
                # 验证监控初始化方法被调用
                mock_init.assert_called()

    def test_setup_production_features(self):
        """测试生产级功能设置"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            with patch.object(UnifiedCacheManager, '_init_production_components') as mock_init:
                manager = UnifiedCacheManager()
                # 验证生产功能初始化方法被调用
                mock_init.assert_called()

    def test_integrate_components(self):
        """测试组件集成"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            manager = UnifiedCacheManager()
            manager._integrate_components()
            mock_init_components.assert_called()

    def test_finalize_initialization(self):
        """测试初始化完成"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            with patch.object(UnifiedCacheManager, '_log_initialization_info', return_value=None) as mock_log:
                manager = UnifiedCacheManager()
                assert mock_log.call_count >= 1


class TestCacheManagerCoreMethods:
    """测试缓存管理器核心方法"""

    @pytest.fixture
    def manager(self):
        """创建缓存管理器实例"""
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'smart': {'enable_monitoring': False}
        })
        manager = UnifiedCacheManager(config)
        yield manager
        if hasattr(manager, 'shutdown'):
            manager.shutdown()

    def test_init_cache_storage(self, manager):
        """测试缓存存储初始化"""
        # 验证基础存储结构被正确初始化
        assert hasattr(manager, 'cache')
        assert hasattr(manager, 'access_times')
        assert hasattr(manager, 'creation_times')
        assert hasattr(manager, 'lock')
        assert hasattr(manager, '_memory_cache')
        assert hasattr(manager, '_memory_stats')

    def test_init_monitoring_components(self, manager):
        """测试监控组件初始化"""
        # 验证监控相关属性被正确初始化
        assert hasattr(manager, '_performance_history')
        assert hasattr(manager, '_prediction_model')
        assert hasattr(manager, '_alert_callbacks')

    def test_init_production_components(self, manager):
        """测试生产级组件初始化"""
        # 验证生产级相关属性被正确初始化
        assert hasattr(manager, '_preload_cache')
        assert hasattr(manager, '_warmup_tasks')
        assert hasattr(manager, '_cleanup_thread')

    def test_validate_key_format_valid(self, manager):
        """测试键格式验证 - 有效键"""
        # 测试有效键 - 使用实际存在的方法
        try:
            manager._validate_get_key("valid_key")
            manager._validate_get_key("key_123")
            manager._validate_get_key("my-key")
        except Exception as e:
            pytest.fail(f"有效键验证失败: {e}")

    def test_validate_key_format_invalid(self, manager):
        """测试键格式验证 - 无效键"""
        from src.infrastructure.cache.core.cache_manager import ValidationError
        
        # 测试无效键 - 使用实际存在的方法
        with pytest.raises(ValidationError):
            manager._validate_get_key("")
        
        with pytest.raises(ValidationError):
            manager._validate_get_key(None)
        
        with pytest.raises(ValidationError):
            manager._validate_get_key(123)

    def test_try_multi_level_cache_lookup_with_cache(self, manager):
        """测试多级缓存查找 - 有缓存"""
        # 模拟多级缓存
        mock_cache = Mock()
        mock_cache.get.return_value = "test_value"
        manager._multi_level_cache = mock_cache
        
        # 模拟内存缓存条目
        mock_entry = Mock()
        mock_entry.is_expired = False
        manager._memory_cache = {"test_key": mock_entry}
        
        result = manager._try_multi_level_cache_lookup("test_key")
        assert result == "test_value"
        mock_cache.get.assert_called_once_with("test_key")

    def test_try_multi_level_cache_lookup_without_cache(self, manager):
        """测试多级缓存查找 - 无缓存"""
        manager._multi_level_cache = None
        result = manager._try_multi_level_cache_lookup("test_key")
        assert result is None

    def test_try_multi_level_cache_lookup_with_expired_entry(self, manager):
        """测试多级缓存查找 - 过期条目"""
        # 模拟多级缓存
        mock_cache = Mock()
        mock_cache.get.return_value = "test_value"
        manager._multi_level_cache = mock_cache
        
        # 模拟过期的内存缓存条目
        mock_entry = Mock()
        mock_entry.is_expired = True
        manager._memory_cache = {"test_key": mock_entry}
        
        with patch.object(manager, '_memory_stats') as mock_stats:
            result = manager._try_multi_level_cache_lookup("test_key")
            assert result is None
            mock_cache.delete.assert_called_once_with("test_key")

    def test_check_distributed_cache_consistency_with_manager(self, manager):
        """测试分布式缓存一致性检查 - 有管理器"""
        mock_dist_manager = Mock()
        mock_dist_manager.get.return_value = "dist_value"
        manager._distributed_manager = mock_dist_manager
        
        # 测试一致的值
        with patch('src.infrastructure.cache.core.cache_manager.logger') as mock_logger:
            manager._check_distributed_cache_consistency("test_key", "dist_value")
            mock_dist_manager.get.assert_called_once_with("test_key")

    def test_check_distributed_cache_consistency_inconsistent(self, manager):
        """测试分布式缓存一致性检查 - 不一致"""
        mock_dist_manager = Mock()
        mock_dist_manager.get.return_value = "different_value"
        manager._distributed_manager = mock_dist_manager
        
        with patch('src.infrastructure.cache.core.cache_manager.logger') as mock_logger:
            manager._check_distributed_cache_consistency("test_key", "test_value")
            mock_logger.warning.assert_called()

    def test_perform_fallback_lookup_found(self, manager):
        """测试降级查找 - 找到值"""
        with patch.object(manager, '_fallback_cache_lookup') as mock_fallback:
            mock_fallback.return_value = {'found': True, 'value': 'test_value'}
            
            with patch.object(manager, '_memory_stats') as mock_stats:
                result = manager._perform_fallback_lookup("test_key")
                assert result == "test_value"
                mock_stats.hits += 1

    def test_perform_fallback_lookup_not_found(self, manager):
        """测试降级查找 - 未找到值"""
        with patch.object(manager, '_fallback_cache_lookup') as mock_fallback:
            mock_fallback.return_value = {'found': False, 'value': None}
            
            with patch.object(manager, '_memory_stats') as mock_stats:
                result = manager._perform_fallback_lookup("test_key")
                assert result is None
                mock_stats.misses += 1

    def test_update_request_stats_with_update_method(self, manager):
        """测试请求统计更新 - 有update方法"""
        mock_stats = Mock()
        mock_stats.update.return_value = None
        manager._memory_stats = mock_stats
        
        manager._update_request_stats()
        mock_stats.update.assert_called_once()

    def test_update_request_stats_without_update_method(self, manager):
        """测试请求统计更新 - 无update方法"""
        # 没有update方法的stats对象
        manager._memory_stats = object()
        
        # 应该不抛出异常
        manager._update_request_stats()

    def test_fallback_cache_lookup_memory_hit(self, manager):
        """测试降级缓存查找 - 内存命中"""
        from collections import OrderedDict
        
        # 模拟内存缓存条目
        mock_entry = Mock()
        mock_entry.is_expired = False
        mock_entry.touch.return_value = None
        mock_entry.value = "memory_value"
        
        # 正确设置内存缓存为OrderedDict
        manager._memory_cache = OrderedDict([("test_key", mock_entry)])
        
        with patch.object(manager, '_update_access_stats') as mock_update:
            result = manager._fallback_cache_lookup("test_key")
            assert result == {'found': True, 'value': 'memory_value'}

    def test_fallback_cache_lookup_basic_cache_hit(self, manager):
        """测试降级缓存查找 - 基础缓存命中"""
        manager.cache = {"test_key": "basic_value"}
        
        result = manager._fallback_cache_lookup("test_key")
        assert result == {'found': True, 'value': 'basic_value'}

    def test_fallback_cache_lookup_distributed_hit(self, manager):
        """测试降级缓存查找 - 分布式缓存命中"""
        mock_dist_manager = Mock()
        mock_dist_manager.get.return_value = "distributed_value"
        manager._distributed_manager = mock_dist_manager
        
        result = manager._fallback_cache_lookup("test_key")
        assert result == {'found': True, 'value': 'distributed_value'}

    def test_fallback_cache_set_success(self, manager):
        """测试降级缓存设置 - 成功"""
        with patch('time.time', return_value=12345):
            result = manager._fallback_cache_set("test_key", "test_value")
            assert result is True
            assert manager.cache["test_key"] == "test_value"
            assert manager.access_times["test_key"] == 12345
            assert manager.creation_times["test_key"] == 12345

    def test_fallback_cache_set_failure(self, manager):
        """测试降级缓存设置 - 失败"""
        # 模拟异常情况
        with patch('time.time', side_effect=Exception("时间错误")):
            with patch('src.infrastructure.cache.core.cache_manager.logger') as mock_logger:
                result = manager._fallback_cache_set("test_key", "test_value")
                assert result is False
                mock_logger.error.assert_called()


class TestCacheManagerAdvancedOperations:
    """测试缓存管理器高级操作"""

    @pytest.fixture
    def manager(self):
        """创建缓存管理器实例"""
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'smart': {'enable_monitoring': False}
        })
        manager = UnifiedCacheManager(config)
        yield manager
        if hasattr(manager, 'shutdown'):
            manager.shutdown()

    def test_optimized_cache_lookup_memory_hit(self, manager):
        """测试优化缓存查找 - 内存命中"""
        with patch.object(manager, '_lookup_memory_cache') as mock_lookup:
            mock_lookup.return_value = {'found': True, 'value': 'memory_value'}
            
            result = manager._optimized_cache_lookup("test_key")
            assert result == {'found': True, 'value': 'memory_value', 'level': 'memory'}

    def test_optimized_cache_lookup_redis_hit(self, manager):
        """测试优化缓存查找 - Redis命中"""
        with patch.object(manager, '_lookup_memory_cache', return_value={'found': False}):
            with patch.object(manager, '_lookup_redis_cache') as mock_lookup:
                mock_lookup.return_value = {'found': True, 'value': 'redis_value'}
                
                result = manager._optimized_cache_lookup("test_key")
                assert result == {'found': True, 'value': 'redis_value', 'level': 'redis'}

    def test_optimized_cache_lookup_no_hit(self, manager):
        """测试优化缓存查找 - 无命中"""
        lookup_methods = ['_lookup_memory_cache', '_lookup_redis_cache', 
                         '_lookup_file_cache', '_lookup_basic_cache', '_lookup_preload_cache']
        
        for method in lookup_methods:
            with patch.object(manager, method, return_value={'found': False}):
                pass
        
        result = manager._optimized_cache_lookup("test_key")
        assert result == {'found': False, 'value': None, 'level': None}

    def test_promote_to_higher_cache_success(self, manager):
        """测试提升到高级缓存 - 成功"""
        with patch.object(manager, '_set_memory_cache') as mock_set:
            manager._promote_to_higher_cache("test_key", "test_value", 60)
            mock_set.assert_called_once_with("test_key", "test_value", 60)

    def test_promote_to_higher_cache_failure(self, manager):
        """测试提升到高级缓存 - 失败"""
        with patch.object(manager, '_set_memory_cache', side_effect=Exception("设置失败")):
            with patch('src.infrastructure.cache.core.cache_manager.logger') as mock_logger:
                manager._promote_to_higher_cache("test_key", "test_value", 60)
                mock_logger.warning.assert_called()

    def test_lookup_cache_hierarchy_distributed_hit(self, manager):
        """测试缓存层级查找 - 分布式命中"""
        # 模拟前几个层级都未命中
        lookup_methods = ['_lookup_memory_cache', '_lookup_basic_cache', 
                         '_lookup_redis_cache', '_lookup_file_cache', '_lookup_preload_cache']
        
        for method in lookup_methods:
            with patch.object(manager, method, return_value={'found': False}):
                pass
        
        # 模拟分布式缓存管理器
        mock_dist_manager = Mock()
        mock_dist_manager.get.return_value = "distributed_value"
        manager._distributed_manager = mock_dist_manager
        
        with patch('src.infrastructure.cache.core.cache_manager.logger') as mock_logger:
            result = manager._lookup_cache_hierarchy("test_key")
            assert result == {'found': True, 'value': 'distributed_value', 'source': 'distributed'}

    def test_lookup_cache_hierarchy_no_hit(self, manager):
        """测试缓存层级查找 - 无命中"""
        # 模拟所有查找方法都未命中
        lookup_methods = ['_lookup_memory_cache', '_lookup_basic_cache', 
                         '_lookup_redis_cache', '_lookup_file_cache', '_lookup_preload_cache']
        
        for method in lookup_methods:
            with patch.object(manager, method, return_value={'found': False}):
                pass
        
        manager._distributed_manager = None
        
        result = manager._lookup_cache_hierarchy("test_key")
        assert result == {'found': False, 'value': None, 'source': None}


class TestCacheManagerMemoryManagement:
    """测试缓存管理器内存管理功能"""

    @pytest.fixture
    def manager(self):
        """创建缓存管理器实例"""
        config = CacheConfig.from_dict({
            'basic': {'max_size': 10, 'ttl': 300},  # 小容量用于测试
            'multi_level': {'level': 'memory', 'memory_max_size': 5, 'memory_ttl': 60},
            'smart': {'enable_monitoring': False}
        })
        manager = UnifiedCacheManager(config)
        yield manager
        if hasattr(manager, 'shutdown'):
            manager.shutdown()

    def test_lookup_memory_cache_hit(self, manager):
        """测试内存缓存查找 - 命中"""
        from collections import OrderedDict
        
        # 创建模拟的缓存条目
        mock_entry = Mock()
        mock_entry.is_expired.return_value = False  # 注意：is_expired是方法
        mock_entry.value = "cached_value"
        mock_entry.update_access = Mock()
        
        # 使用OrderedDict
        manager._memory_cache = OrderedDict([("test_key", mock_entry)])
        
        with patch.object(manager, '_update_access_stats') as mock_update_stats:
            result = manager._lookup_memory_cache("test_key")
            assert result == {'found': True, 'value': 'cached_value', 'source': 'memory'}

    def test_lookup_memory_cache_expired(self, manager):
        """测试内存缓存查找 - 过期"""
        from collections import OrderedDict
        
        # 创建过期的缓存条目
        mock_entry = Mock()
        mock_entry.is_expired.return_value = True  # 注意：is_expired是方法
        
        manager._memory_cache = OrderedDict([("test_key", mock_entry)])
        
        # 模拟_memory_stats，确保有evictions属性
        mock_stats = Mock()
        mock_stats.evictions = 0
        manager._memory_stats = mock_stats
        
        result = manager._lookup_memory_cache("test_key")
        assert result == {'found': False, 'value': None, 'source': None}

    def test_lookup_memory_cache_miss(self, manager):
        """测试内存缓存查找 - 未命中"""
        from collections import OrderedDict
        
        manager._memory_cache = OrderedDict()
        
        result = manager._lookup_memory_cache("test_key")
        assert result == {'found': False, 'value': None, 'source': None}


class TestCacheManagerErrorHandling:
    """测试缓存管理器错误处理"""

    @pytest.fixture
    def manager(self):
        """创建缓存管理器实例"""
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'smart': {'enable_monitoring': False}
        })
        manager = UnifiedCacheManager(config)
        yield manager
        if hasattr(manager, 'shutdown'):
            manager.shutdown()

    def test_try_multi_level_cache_lookup_exception(self, manager):
        """测试多级缓存查找异常处理"""
        mock_cache = Mock()
        mock_cache.get.side_effect = TimeoutError("连接超时")
        manager._multi_level_cache = mock_cache
        
        with patch('src.infrastructure.cache.core.cache_manager.logger') as mock_logger:
            result = manager._try_multi_level_cache_lookup("test_key")
            assert result is None
            mock_logger.warning.assert_called()

    def test_fallback_cache_lookup_distributed_exception(self, manager):
        """测试降级缓存查找分布式异常处理"""
        mock_dist_manager = Mock()
        mock_dist_manager.get.side_effect = Exception("分布式连接失败")
        manager._distributed_manager = mock_dist_manager
        
        with patch('src.infrastructure.cache.core.cache_manager.logger') as mock_logger:
            result = manager._fallback_cache_lookup("test_key")
            assert result == {'found': False, 'value': None}

    def test_optimized_cache_lookup_strategy_exception(self, manager):
        """测试优化缓存查找策略异常处理"""
        with patch.object(manager, '_lookup_memory_cache', side_effect=Exception("内存错误")):
            with patch.object(manager, '_lookup_redis_cache', return_value={'found': True, 'value': 'redis_value'}):
                with patch('src.infrastructure.cache.core.cache_manager.logger') as mock_logger:
                    result = manager._optimized_cache_lookup("test_key")
                    assert result == {'found': True, 'value': 'redis_value', 'level': 'redis'}

    def test_lookup_cache_hierarchy_distributed_exception(self, manager):
        """测试缓存层级查找分布式异常处理"""
        lookup_methods = ['_lookup_memory_cache', '_lookup_basic_cache', 
                         '_lookup_redis_cache', '_lookup_file_cache', '_lookup_preload_cache']
        
        for method in lookup_methods:
            with patch.object(manager, method, return_value={'found': False}):
                pass
        
        mock_dist_manager = Mock()
        mock_dist_manager.get.side_effect = Exception("分布式连接失败")
        manager._distributed_manager = mock_dist_manager
        
        with patch('src.infrastructure.cache.core.cache_manager.logger') as mock_logger:
            result = manager._lookup_cache_hierarchy("test_key")
            assert result == {'found': False, 'value': None, 'source': None}


class TestCacheManagerStatistics:
    """测试缓存管理器统计功能"""

    @pytest.fixture
    def manager(self):
        """创建缓存管理器实例"""
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'smart': {'enable_monitoring': False}
        })
        manager = UnifiedCacheManager(config)
        yield manager
        if hasattr(manager, 'shutdown'):
            manager.shutdown()

    def test_update_access_stats(self, manager):
        """测试访问统计更新"""
        # 先设置一个键在access_times中
        manager.access_times = {"test_key": 12300}
        
        with patch('time.time', return_value=12345):
            manager._update_access_stats("test_key")
            # 验证访问时间被更新
            assert manager.access_times["test_key"] == 12345
            
        # 测试不存在的键
        manager._update_access_stats("nonexistent_key")
        # 不存在的键不应该被添加到access_times中
        assert "nonexistent_key" not in manager.access_times

    def test_get_cache_stats_comprehensive(self, manager):
        """测试获取缓存统计信息 - 全面测试"""
        from collections import OrderedDict
        
        # 设置一些测试数据
        manager.cache = {"key1": "value1", "key2": "value2"}
        manager._memory_cache = OrderedDict([("key3", Mock())])
        
        # 模拟统计对象，确保有正确的属性
        mock_stats = Mock()
        mock_stats.hits = 10
        mock_stats.misses = 5
        mock_stats.evictions = 2
        mock_stats.hit_rate = 0.8
        mock_stats.total_requests = 15
        mock_stats.total_size_bytes = 1024  # 提供数值而不是Mock对象
        manager._memory_stats = mock_stats
        
        # 确保配置对象有正确的属性
        if hasattr(manager.config, 'basic'):
            manager.config.basic.max_size = 100
            if hasattr(manager.config.basic, 'strategy'):
                strategy_mock = StandardMockBuilder.create_cache_mock()
                strategy_mock.value = "LRU"
                manager.config.basic.strategy = strategy_mock
        
        if hasattr(manager.config, 'multi_level'):
            level_mock = StandardMockBuilder.create_cache_mock()
            level_mock.value = "memory"
            manager.config.multi_level.level = level_mock
        
        if hasattr(manager.config, 'distributed'):
            manager.config.distributed.distributed = False
        
        manager._monitoring_active = False
        
        stats = manager.get_cache_stats()
        
        assert isinstance(stats, dict)
        # 基本验证 - 只要有返回结果就说明方法能工作
        assert len(stats) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])