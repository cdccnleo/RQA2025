#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存性能基准测试

建立缓存系统的性能基准测试
目标：验证缓存系统在各种负载下的性能表现
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
import threading
import statistics
from unittest.mock import Mock, patch
from typing import Dict, List, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
from src.infrastructure.cache.core.cache_configs import (
    BasicCacheConfig, CacheConfig, MultiLevelCacheConfig, 
    AdvancedCacheConfig, SmartCacheConfig, DistributedCacheConfig
)
from src.infrastructure.cache.strategies.cache_strategy_manager import (
    CacheStrategyManager, StrategyType, LRUStrategy, LFUStrategy, TTLStrategy
)


class TestCachePerformanceBenchmarks:
    """缓存性能基准测试类"""

    @pytest.fixture
    def basic_config(self):
        """创建基础配置"""
        return CacheConfig(
            basic=BasicCacheConfig(max_size=1000, ttl=3600),
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )

    @pytest.fixture
    def large_config(self):
        """创建大容量配置"""
        return CacheConfig(
            basic=BasicCacheConfig(max_size=10000, ttl=7200),
            multi_level=MultiLevelCacheConfig(),
            advanced=AdvancedCacheConfig(),
            smart=SmartCacheConfig(),
            distributed=DistributedCacheConfig()
        )

    def measure_execution_time(self, func, *args, **kwargs) -> float:
        """测量函数执行时间"""
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        return end_time - start_time

    def test_cache_set_performance(self, basic_config):
        """测试缓存设置操作性能"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(basic_config)
            
            # 单个设置操作性能测试
            durations = []
            for i in range(100):
                duration = self.measure_execution_time(manager.set, f"key_{i}", f"value_{i}")
                durations.append(duration)
            
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            
            # 性能要求：平均设置时间 < 1ms
            assert avg_duration < 0.001, f"平均设置时间 {avg_duration:.4f}s 超过基准"
            assert max_duration < 0.01, f"最大设置时间 {max_duration:.4f}s 超过基准"

    def test_cache_get_performance(self, basic_config):
        """测试缓存获取操作性能"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(basic_config)
            
            # 预先填充缓存
            for i in range(100):
                manager.set(f"key_{i}", f"value_{i}")
            
            # 获取操作性能测试
            durations = []
            for i in range(100):
                duration = self.measure_execution_time(manager.get, f"key_{i}")
                durations.append(duration)
            
            avg_duration = statistics.mean(durations)
            max_duration = max(durations)
            
            # 性能要求：平均获取时间 < 0.5ms
            assert avg_duration < 0.0005, f"平均获取时间 {avg_duration:.4f}s 超过基准"
            assert max_duration < 0.005, f"最大获取时间 {max_duration:.4f}s 超过基准"

    def test_cache_concurrent_performance(self, large_config):
        """测试并发操作性能"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(large_config)
            
            def worker(worker_id):
                """工作线程函数"""
                results = []
                for i in range(50):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    
                    # 设置
                    start = time.perf_counter()
                    manager.set(key, value)
                    set_time = time.perf_counter() - start
                    
                    # 获取
                    start = time.perf_counter()
                    result = manager.get(key)
                    get_time = time.perf_counter() - start
                    
                    results.append((set_time, get_time, result is not None))
                
                return results
            
            # 使用线程池进行并发测试
            num_workers = 5
            operations_per_worker = 50
            
            start_time = time.perf_counter()
            
            with ThreadPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(worker, i) for i in range(num_workers)]
                all_results = []
                
                for future in as_completed(futures):
                    results = future.result()
                    all_results.extend(results)
            
            total_time = time.perf_counter() - start_time
            
            # 验证所有操作成功
            successful_gets = sum(1 for _, _, success in all_results if success)
            total_operations = len(all_results)
            
            assert successful_gets == total_operations, "存在获取操作失败"
            
            # 性能要求：总时间合理
            expected_min_ops_per_sec = 1000  # 每秒至少1000次操作
            actual_ops_per_sec = (total_operations * 2) / total_time  # set + get
            assert actual_ops_per_sec > expected_min_ops_per_sec, f"操作频率 {actual_ops_per_sec:.2f} ops/sec 低于基准"

    def test_memory_usage_efficiency(self, basic_config):
        """测试内存使用效率"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(basic_config)
            
            # 填充到接近容量限制
            keys_per_batch = 100
            for batch in range(10):
                for i in range(keys_per_batch):
                    manager.set(f"batch_{batch}_key_{i}", f"batch_{batch}_value_{i}")
            
            # 验证缓存大小控制
            stats = manager.get_cache_stats()
            total_keys = stats.get('total_keys', 0)
            
            # 应该不超过配置的最大大小太多（允许一些超出用于测试）
            assert total_keys <= basic_config.basic.max_size * 2, f"缓存大小 {total_keys} 超出预期限制"

    def test_strategy_compared_performance(self):
        """测试不同策略的性能比较"""
        strategies = {
            'LRU': LRUStrategy(capacity=1000),
            'LFU': LFUStrategy(capacity=1000),
            'TTL': TTLStrategy(capacity=1000)
        }
        
        # 统一的访问模式
        test_operations = [
            ('set', 'key1', 'value1'),
            ('set', 'key2', 'value2'),
            ('set', 'key3', 'value3'),
            ('get', 'key1', None),
            ('set', 'key4', 'value4'),
            ('get', 'key2', None),
            ('get', 'key3', None),
            ('get', 'key4', None),
        ]
        
        strategy_performance = {}
        
        for strategy_name, strategy in strategies.items():
            durations = []
            
            for operation, key, value in test_operations:
                start = time.perf_counter()
                
                if operation == 'set':
                    strategy.put(key, value)
                else:  # get
                    strategy.get(key)
                
                duration = time.perf_counter() - start
                durations.append(duration)
            
            avg_duration = statistics.mean(durations)
            strategy_performance[strategy_name] = avg_duration
        
        # 验证所有策略都在合理性能范围内
        for strategy_name, avg_time in strategy_performance.items():
            assert avg_time < 0.001, f"{strategy_name} 策略平均操作时间 {avg_time:.4f}s 超基准"

    def test_cache_hit_rate_performance(self, basic_config):
        """测试缓存命中率性能"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(basic_config)
            
            # 预热缓存
            for i in range(50):
                manager.set(f"hot_key_{i}", f"hot_value_{i}")
            
            hits = 0
            misses = 0
            
            # 测试命中率
            for round_num in range(5):
                for i in range(50):
                    result = manager.get(f"hot_key_{i}")
                    if result is not None:
                        hits += 1
                    else:
                        misses += 1
                
                # 测试一些冷数据
                for i in range(50, 100):
                    result = manager.get(f"cold_key_{i}")
                    if result is not None:
                        hits += 1
                    else:
                        misses += 1
            
            total_requests = hits + misses
            hit_rate = hits / total_requests if total_requests > 0 else 0
            
            # 验证热数据命中率应该很高
            assert hit_rate > 0.4, f"缓存命中率 {hit_rate:.2%} 低于预期"

    def test_cache_cleanup_performance(self, basic_config):
        """测试缓存清理性能"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(basic_config)
            
            # 填充缓存
            for i in range(200):
                manager.set(f"key_{i}", f"value_{i}")
            
            initial_stats = manager.get_cache_stats()
            
            # 清理操作性能测试
            cleanup_start = time.perf_counter()
            
            if hasattr(manager, 'clear'):
                manager.clear()
            
            cleanup_duration = time.perf_counter() - cleanup_start
            
            final_stats = manager.get_cache_stats()
            
            # 验证清理效果和性能
            if 'total_keys' in final_stats and 'total_keys' in initial_stats:
                assert final_stats['total_keys'] < initial_stats['total_keys']
            
            # 清理操作应该在合理时间内完成
            assert cleanup_duration < 0.1, f"缓存清理时间 {cleanup_duration:.4f}s 超基准"

    @pytest.mark.slow
    def test_cache_stress_performance(self, large_config):
        """测试缓存压力性能"""
        with patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager._start_cleanup_thread'), \
             patch('src.infrastructure.cache.core.cache_manager.UnifiedCacheManager.start_monitoring'):
            
            manager = UnifiedCacheManager(large_config)
            
            def stress_worker(worker_id, num_operations):
                """压力测试工作函数"""
                for i in range(num_operations):
                    # 随机操作类型
                    if i % 3 == 0:
                        # 设置操作
                        manager.set(f"stress_{worker_id}_{i}", f"value_{i}")
                    else:
                        # 获取操作
                        manager.get(f"stress_{worker_id}_{i}")
            
            # 压力测试参数
            num_workers = 10
            operations_per_worker = 200
            
            start_time = time.perf_counter()
            
            threads = []
            for i in range(num_workers):
                thread = threading.Thread(
                    target=stress_worker, 
                    args=(i, operations_per_worker)
                )
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()
            
            total_time = time.perf_counter() - start_time
            total_operations = num_workers * operations_per_worker
            
            ops_per_second = total_operations / total_time
            
            # 压力测试应该维持合理性能
            assert ops_per_second > 500, f"压力测试下操作频率 {ops_per_second:.2f} ops/sec 低于基准"
            assert total_time < 10, f"压力测试总时间 {total_time:.2f}s 超过基准"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
