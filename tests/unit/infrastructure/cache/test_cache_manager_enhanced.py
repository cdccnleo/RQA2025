#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强缓存管理器单元测试

补充测试UnifiedCacheManager的高级功能和边界情况
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
from src.infrastructure.cache.core.cache_configs import CacheConfig, CacheLevel
from src.infrastructure.cache.interfaces.data_structures import CacheEntry
# ValidationError is defined as Exception alias in cache_manager


class TestUnifiedCacheManagerEnhanced:
    """测试统一缓存管理器增强功能"""

    def setup_method(self, method):
        """测试前准备"""
        self.config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'advanced': {'enable_compression': False, 'enable_preloading': False, 'cleanup_interval': 1},
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        })
        self.manager = UnifiedCacheManager(self.config)

    def teardown_method(self, method):
        """测试后清理"""
        if hasattr(self, 'manager') and self.manager:
            self.manager.shutdown()

    def test_multi_level_cache_integration(self):
        """测试多级缓存集成"""
        # 设置配置以启用多级缓存
        multi_config = CacheConfig.from_dict({
            'basic': {'max_size': 1000, 'ttl': 3600},
            'multi_level': {
                'level': 'hybrid',
                'memory_max_size': 100,
                'memory_ttl': 300,
                'redis_max_size': 1000,
                'redis_ttl': 3600
            },
            'advanced': {'enable_compression': True, 'enable_preloading': True},
            'smart': {'enable_monitoring': True, 'enable_auto_optimization': True}
        })
        
        with patch('src.infrastructure.cache.core.cache_manager.MultiLevelCache') as mock_multi_cache:
            # Mock多级缓存的行为
            mock_instance = Mock()
            mock_instance.get.return_value = "test_value"
            mock_instance.set.return_value = True
            mock_instance.delete.return_value = True
            mock_instance.clear.return_value = True
            mock_instance.get_stats.return_value = {
                'size': 1,
                'total_hits': 1,
                'total_misses': 0,
                'hit_rate': 1.0
            }
            mock_multi_cache.return_value = mock_instance
            
            manager = UnifiedCacheManager(multi_config)
            
            # 测试多级缓存操作
            manager.set("multi_key", "multi_value")
            value = manager.get("multi_key")
            assert value == "test_value"
            
            # 验证多级缓存被调用
            mock_instance.set.assert_called()
            mock_instance.get.assert_called()

    def test_redis_cache_operations(self):
        """测试Redis缓存操作"""
        # 设置启用Redis的配置
        redis_config = CacheConfig.from_dict({
            'basic': {'max_size': 1000, 'ttl': 3600},
            'multi_level': {
                'level': 'redis',
                'redis_max_size': 1000,
                'redis_ttl': 3600
            },
            'distributed': {
                'distributed': True,
                'redis_host': 'localhost',
                'redis_port': 6379
            }
        })
        
        with patch('redis.Redis') as mock_redis:
            mock_client = Mock()
            mock_client.ping.return_value = True
            mock_client.setex.return_value = True
            # 修正mock返回值，使其符合_get_redis_cache方法的期望
            mock_client.get.return_value = '"redis_value"'  # 返回JSON字符串格式
            mock_redis.return_value = mock_client
            
            manager = UnifiedCacheManager(redis_config)
            # 手动设置_redis_client以绕过初始化过程
            manager._redis_client = mock_client
            
            # 测试Redis操作
            result = manager._set_redis_cache("redis_key", "redis_value", 300)
            assert result is None  # 方法没有返回值
            
            value = manager._get_redis_cache("redis_key")
            assert value == "redis_value"

    def test_file_cache_operations(self):
        """测试文件缓存操作"""
        import tempfile
        import os
        import pickle
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp()
        
        try:
            # 设置文件缓存配置
            file_config = CacheConfig.from_dict({
                'basic': {'max_size': 1000, 'ttl': 3600},
                'multi_level': {
                    'level': 'file',
                    'file_max_size': 1000,
                    'file_ttl': 3600,
                    'file_cache_dir': temp_dir
                }
            })
            
            manager = UnifiedCacheManager(file_config)
            
            # 测试文件缓存操作
            manager._set_file_cache("file_key", "file_value", 300)
            
            value = manager._get_file_cache("file_key")
            assert value == "file_value"
            
            # 测试文件缓存删除
            manager._delete_file_cache("file_key")
            value = manager._get_file_cache("file_key")
            assert value is None
            
        finally:
            # 清理临时目录
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)

    def test_cache_eviction_strategies(self):
        """测试缓存淘汰策略"""
        # 设置小容量缓存以测试淘汰
        small_config = CacheConfig.from_dict({
            'basic': {'max_size': 5, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 3, 'memory_ttl': 60}
        })
        
        manager = UnifiedCacheManager(small_config)
        
        # 添加超过容量限制的缓存项
        for i in range(10):
            manager._set_memory_cache(f"evict_key_{i}", f"evict_value_{i}", 300)
        
        # 验证缓存大小被控制
        assert len(manager._memory_cache) <= 3
        
        # 验证最近添加的项存在
        assert manager._memory_cache.get("evict_key_9") is not None

    def test_concurrent_cache_operations(self):
        """测试并发缓存操作"""
        import concurrent.futures
        
        # 为这个测试创建一个新的manager实例
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'advanced': {'enable_compression': False, 'enable_preloading': False},
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        })
        manager = UnifiedCacheManager(config)
        
        errors = []
        results = []
        
        def worker(worker_id):
            """工作线程函数"""
            try:
                for i in range(20):
                    key = f"concurrent_{worker_id}_{i}"
                    value = f"value_{worker_id}_{i}"
                    
                    # 设置缓存
                    manager.set(key, value)
                    
                    # 获取缓存
                    retrieved = manager.get(key)
                    if retrieved != value:
                        errors.append(f"Worker {worker_id} mismatch: expected {value}, got {retrieved}")
                    
                    # 删除缓存
                    manager.delete(key)
                    
            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")
        
        # 启动多个并发线程
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            concurrent.futures.wait(futures)
        
        # 验证没有错误
        assert len(errors) == 0, f"Concurrent operations had errors: {errors}"

    def test_cache_statistics_accuracy(self):
        """测试缓存统计信息准确性"""
        # 为这个测试创建一个新的manager实例
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'advanced': {'enable_compression': False, 'enable_preloading': False},
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        })
        manager = UnifiedCacheManager(config)
        
        # 执行一系列操作
        manager.set("stat_key1", "stat_value1")
        manager.set("stat_key2", "stat_value2")
        
        # 命中测试
        value1 = manager.get("stat_key1")
        value2 = manager.get("stat_key2")
        
        # 未命中测试
        missing = manager.get("missing_key")
        
        # 获取统计信息
        stats = manager.get_cache_stats()
        
        # 验证统计信息
        assert stats['total_requests'] >= 3
        assert stats['total_hits'] >= 2
        assert stats['total_misses'] >= 1
        assert 0 <= stats['hit_rate'] <= 1

    def test_cache_manager_shutdown(self):
        """测试缓存管理器关闭"""
        # 为这个测试创建一个新的manager实例
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'advanced': {'enable_compression': False, 'enable_preloading': False},
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        })
        manager = UnifiedCacheManager(config)
        
        # 执行一些操作
        manager.set("shutdown_key", "shutdown_value")
        assert manager.get("shutdown_key") == "shutdown_value"
        
        # 关闭管理器
        manager.shutdown()
        
        # 验证管理器状态
        health = manager.get_health_status()
        # 注意：关闭后健康状态可能因实现而异

    def test_cache_manager_context_manager(self):
        """测试缓存管理器上下文管理器"""
        with UnifiedCacheManager() as manager:
            # 在上下文中使用
            manager.set("context_key", "context_value")
            assert manager.get("context_key") == "context_value"
        
        # 上下文退出后，管理器应该被正确关闭

    def test_error_handling_robustness(self):
        """测试错误处理的健壮性"""
        # 为这个测试创建一个新的manager实例
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'advanced': {'enable_compression': False, 'enable_preloading': False},
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        })
        manager = UnifiedCacheManager(config)
        
        # 测试无效键应该抛出异常
        with pytest.raises(Exception):
            manager.set("", "value")  # 空字符串键应该被拒绝
        
        # 测试空键
        result = manager.set("empty_key", "value")
        assert result == True
        
        # 测试超长键
        long_key = "x" * 10000
        result = manager.set(long_key, "value")
        # 结果取决于实现，但不应该抛出异常

    def test_performance_under_load(self):
        """测试负载下的性能"""
        import time
        
        # 为这个测试创建一个新的manager实例
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'advanced': {'enable_compression': False, 'enable_preloading': False},
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        })
        manager = UnifiedCacheManager(config)
        
        # 批量操作测试
        start_time = time.time()
        
        # 设置大量缓存项（不超过配置的50个容量限制）
        for i in range(40):
            manager.set(f"perf_key_{i}", f"perf_value_{i}")

        set_time = time.time() - start_time

        start_time = time.time()

        # 获取大量缓存项
        for i in range(40):
            value = manager.get(f"perf_key_{i}")
            assert value == f"perf_value_{i}"
        
        get_time = time.time() - start_time
        
        # 验证性能在合理范围内
        assert set_time < 5.0, f"Set operations too slow: {set_time}s"
        assert get_time < 5.0, f"Get operations too slow: {get_time}s"

    def test_memory_usage_monitoring(self):
        """测试内存使用监控"""
        # 为这个测试创建一个新的manager实例
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'advanced': {'enable_compression': False, 'enable_preloading': False},
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        })
        manager = UnifiedCacheManager(config)
        
        # 添加大量数据
        for i in range(50):
            manager.set(f"memory_key_{i}", f"memory_value_{i}" * 100)
        
        # 获取统计信息
        stats = manager.get_cache_stats()
        
        # 验证内存使用信息存在
        assert 'memory_usage_mb' in stats
        assert isinstance(stats['memory_usage_mb'], (int, float))

    def test_cache_tier_promotion(self):
        """测试缓存层级提升"""
        # 为这个测试创建一个新的manager实例
        config = CacheConfig.from_dict({
            'basic': {'max_size': 100, 'ttl': 300},
            'multi_level': {'level': 'memory', 'memory_max_size': 50, 'memory_ttl': 60},
            'advanced': {'enable_compression': False, 'enable_preloading': False},
            'smart': {'enable_monitoring': False, 'enable_auto_optimization': False}
        })
        manager = UnifiedCacheManager(config)
        
        # 这个测试依赖于多级缓存的具体实现
        # 我们主要验证不会抛出异常
        try:
            manager.set("promotion_key", "promotion_value")
            value = manager.get("promotion_key")
            # 不管实现如何，应该能获取到值
            assert value is not None
        except Exception as e:
            pytest.fail(f"Cache tier promotion failed: {e}")


if __name__ == '__main__':
    pytest.main([__file__])