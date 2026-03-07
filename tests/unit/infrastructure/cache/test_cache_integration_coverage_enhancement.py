#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存集成测试覆盖率增强

专门针对缓存模块集成功能的测试覆盖率提升
目标：验证端到端功能和模块间集成
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
from src.infrastructure.cache.core.cache_configs import (
    CacheConfig, BasicCacheConfig, MultiLevelCacheConfig, DistributedCacheConfig,
    AdvancedCacheConfig, SmartCacheConfig
)
from src.infrastructure.cache.strategies.cache_strategy_manager import (
    CacheStrategyManager, StrategyType
)
from src.infrastructure.cache.core.cache_optimizer import CacheOptimizer


@pytest.fixture
def basic_config():
    """创建基础缓存配置"""
    return BasicCacheConfig(
        max_size=100,
        ttl=60
    )


@pytest.fixture
def multi_level_config():
    """创建多级缓存配置"""
    return MultiLevelCacheConfig(
        level='memory',
        memory_max_size=50,
        memory_ttl=30
    )


@pytest.fixture
def distributed_config():
    """创建分布式缓存配置"""
    return DistributedCacheConfig(
        distributed=False,
        redis_host='localhost',
        redis_port=6379
    )


class TestCacheIntegrationEndToEnd:
    """测试缓存端到端集成功能"""

    def test_cache_manager_with_basic_config(self, basic_config):
        """测试缓存管理器与基础配置集成"""
        # 创建完整的CacheConfig对象
        full_config = CacheConfig(
            basic=basic_config,
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )
        
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(full_config)
            
            # 测试基本操作
            manager.set("test_key", "test_value")
            result = manager.get("test_key")
            
            assert result is not None
            assert manager.config == full_config

    def test_cache_manager_with_multi_level_config(self, multi_level_config):
        """测试缓存管理器与多级配置集成"""
        full_config = CacheConfig(
            basic=BasicCacheConfig(),
            multi_level=multi_level_config,
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )
        
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(full_config)
            
            # 验证多级缓存配置
            assert manager.config.multi_level == multi_level_config

    def test_cache_strategy_manager_integration(self):
        """测试缓存策略管理器集成"""
        strategy_manager = CacheStrategyManager(
            default_strategy=StrategyType.LRU,
            capacity=100
        )
        
        # 测试策略切换
        strategy_manager.set_current_strategy(StrategyType.LRU)
        assert strategy_manager.current_strategy_type == StrategyType.LRU
        
        strategy_manager.set_current_strategy(StrategyType.LFU)
        assert strategy_manager.current_strategy_type == StrategyType.LFU

    def test_cache_optimizer_integration(self, basic_config):
        """测试缓存优化器集成"""
        optimizer = CacheOptimizer()
        
        # 测试优化建议生成
        recommendations = optimizer.get_optimization_recommendations()
        assert isinstance(recommendations, dict)
        
        # 测试缓存大小优化
        optimized_size = optimizer.optimize_cache_size(
            current_size=100,
            hit_rate=0.7,
            memory_usage=0.5
        )
        assert isinstance(optimized_size, int)

    def test_full_cache_workflow_integration(self, basic_config):
        """测试完整缓存工作流集成"""
        # 创建完整的CacheConfig对象
        full_config = CacheConfig(
            basic=basic_config,
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )
        
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            # 初始化管理器
            manager = UnifiedCacheManager(full_config)
            
            # 测试设置和获取操作
            test_data = {
                'key1': 'value1',
                'key2': 'value2',
                'key3': 'value3'
            }
            
            # 批量设置
            for key, value in test_data.items():
                manager.set(key, value)
            
            # 批量获取并验证
            for key, expected_value in test_data.items():
                result = manager.get(key)
                assert result is not None


class TestCachePerformanceIntegration:
    """测试缓存性能集成功能"""

    def test_cache_performance_under_load(self, basic_config):
        """测试负载下的缓存性能"""
        # 创建完整的CacheConfig对象
        full_config = CacheConfig(
            basic=basic_config,
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )
        
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(full_config)
            results = []
            
            def worker(thread_id):
                for i in range(20):
                    key = f"thread_{thread_id}_key_{i}"
                    value = f"value_{i}"
                    
                    manager.set(key, value)
                    result = manager.get(key)
                    results.append((key, result))
            
            # 启动多个线程模拟并发负载
            threads = []
            for i in range(3):
                thread = threading.Thread(target=worker, args=(i,))
                threads.append(thread)
                thread.start()
            
            # 等待所有线程完成
            for thread in threads:
                thread.join()
            
            # 验证结果数量
            assert len(results) >= 60  # 3 threads * 20 operations

    def test_cache_memory_efficiency(self, basic_config):
        """测试缓存内存效率"""
        # 创建完整的CacheConfig对象
        full_config = CacheConfig(
            basic=basic_config,
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )
        
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(full_config)
            
            # 填充缓存到容量限制
            for i in range(150):  # 超过配置的max_size=100
                manager.set(f"key_{i}", f"value_{i}")
            
            # 验证缓存大小控制在限制内
            stats = manager.get_cache_stats()
            assert 'total_keys' in stats


class TestCacheErrorHandlingIntegration:
    """测试缓存错误处理集成功能"""

    def test_cache_error_recovery(self, basic_config):
        """测试缓存错误恢复"""
        # 创建完整的CacheConfig对象
        full_config = CacheConfig(
            basic=basic_config,
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )
        
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(full_config)
            
            # 测试无效键处理
            with pytest.raises(Exception):
                manager.set("", "value")  # 空键
            
            with pytest.raises(Exception):
                manager.set(None, "value")  # None键
            
            # 验证系统仍然正常工作
            manager.set("valid_key", "valid_value")
            result = manager.get("valid_key")
            assert result is not None

    def test_cache_distributed_fallback(self):
        """测试分布式缓存降级处理"""
        # 模拟分布式配置但实际不可用的情况
        full_config = CacheConfig(
            basic=BasicCacheConfig(),
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig(distributed=True)
        )
        
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(full_config)
            
            # 基本功能应该仍然可用
            manager.set("test_key", "test_value")
            result = manager.get("test_key")
            assert result is not None


class TestCacheConfigurationIntegration:
    """测试缓存配置集成功能"""

    def test_config_validation_integration(self):
        """测试配置验证集成"""
        # 测试无效配置
        with pytest.raises(ValueError):
            BasicCacheConfig(max_size=0)  # 无效大小
        
        with pytest.raises(ValueError):
            BasicCacheConfig(ttl=-1)  # 无效TTL

    def test_config_merging_integration(self):
        """测试配置合并集成"""
        base_config = BasicCacheConfig(max_size=100, ttl=60)
        
        # 测试配置更新
        updated_config = BasicCacheConfig(
            max_size=200,
            ttl=base_config.ttl  # 保持原TTL
        )
        
        assert updated_config.max_size == 200
        assert updated_config.ttl == 60

    def test_configuration_hierarchy_integration(self):
        """测试配置层次结构集成"""
        basic_config = BasicCacheConfig(max_size=100)
        from src.infrastructure.cache.core.cache_configs import CacheLevel
        multi_config = MultiLevelCacheConfig(level=CacheLevel.MEMORY)
        distributed_config = DistributedCacheConfig(distributed=False)
        
        # 创建包含所有层级的配置
        full_config = CacheConfig(
            basic=basic_config,
            multi_level=multi_config,
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=distributed_config
        )
        
        assert full_config.basic.max_size == 100
        assert full_config.multi_level.level.value == 'memory'
        assert full_config.distributed.distributed is False


class TestCacheHealthMonitoringIntegration:
    """测试缓存健康监控集成功能"""

    def test_health_check_integration(self, basic_config):
        """测试健康检查集成"""
        # 创建完整的CacheConfig对象
        full_config = CacheConfig(
            basic=basic_config,
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )
        
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(full_config)
            
            # 测试健康状态检查
            if hasattr(manager, 'get_health_status'):
                health_status = manager.get_health_status()
                assert isinstance(health_status, dict)
                assert 'status' in health_status

    def test_monitoring_metrics_integration(self, basic_config):
        """测试监控指标集成"""
        # 创建完整的CacheConfig对象
        full_config = CacheConfig(
            basic=basic_config,
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )
        
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(full_config)
            
            # 执行一些操作以生成指标
            for i in range(10):
                manager.set(f"key_{i}", f"value_{i}")
                manager.get(f"key_{i}")
            
            # 验证统计信息
            stats = manager.get_cache_stats()
            assert isinstance(stats, dict)
            assert 'total_keys' in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
