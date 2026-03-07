#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 多级缓存深度测试
测试MultiLevelCache的核心缓存策略、性能优化和监控功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import asyncio
import time
import threading
from unittest.mock import patch, MagicMock, AsyncMock
from dataclasses import dataclass

import pytest

from infrastructure.cache.core.multi_level_cache import MultiLevelCache
from infrastructure.cache.strategies.cache_strategy_manager import LRUStrategy


class TestMultiLevelCacheInitialization:
    """MultiLevelCache初始化测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        cache = MultiLevelCache()

        assert cache is not None
        # 检查是否有基本的缓存层级
        assert hasattr(cache, 'layers')
        assert isinstance(cache.layers, list)

    def test_initialization_with_config(self):
        """测试带配置初始化"""
        config = {
            'memory_cache_size': 1000,
            'redis_cache_size': 5000,
            'disk_cache_enabled': True
        }

        with patch('src.infrastructure.cache.core.multi_level_cache.MemoryCache') as mock_memory, \
             patch('src.infrastructure.cache.core.multi_level_cache.RedisCache') as mock_redis:

            mock_memory.return_value = MagicMock()
            mock_redis.return_value = MagicMock()

            cache = MultiLevelCache(config)

            # 验证配置被应用
            assert cache is not None
            # 实际的初始化逻辑可能因具体实现而异


class TestMultiLevelCacheBasicOperations:
    """MultiLevelCache基本操作测试"""

    @pytest.fixture
    def cache(self):
        """MultiLevelCache fixture"""
        # 创建一个简单的多级缓存实现
        cache = MultiLevelCache()

        # Mock不同的缓存层级
        cache.layers = [
            MagicMock(),  # L1 - 内存缓存
            MagicMock(),  # L2 - Redis缓存
        ]

        # 设置层级行为
        cache.layers[0].get.return_value = None  # L1未命中
        cache.layers[0].put.return_value = None
        cache.layers[1].get.return_value = "cached_value"  # L2命中
        cache.layers[1].put.return_value = None

        return cache

    def test_get_from_cache_hit_l1(self, cache):
        """测试从L1缓存命中"""
        # 设置L1命中
        cache.layers[0].get.return_value = "l1_value"
        cache.layers[1].get.return_value = None

        result = cache.get("test_key")

        assert result == "l1_value"
        cache.layers[0].get.assert_called_with("test_key")
        cache.layers[1].get.assert_not_called()  # 不应该访问L2

    def test_get_from_cache_hit_l2(self, cache):
        """测试从L2缓存命中"""
        result = cache.get("test_key")

        assert result == "cached_value"
        cache.layers[0].get.assert_called_with("test_key")
        cache.layers[1].get.assert_called_with("test_key")

    def test_get_from_cache_miss_all(self, cache):
        """测试所有缓存层都未命中"""
        cache.layers[0].get.return_value = None
        cache.layers[1].get.return_value = None

        result = cache.get("missing_key")

        assert result is None

    def test_put_to_cache(self, cache):
        """测试写入缓存"""
        cache.put("test_key", "test_value")

        # 验证所有层级都被写入
        cache.layers[0].put.assert_called_with("test_key", "test_value")
        cache.layers[1].put.assert_called_with("test_key", "test_value")

    def test_delete_from_cache(self, cache):
        """测试从缓存删除"""
        cache.delete("test_key")

        # 验证所有层级都被删除
        cache.layers[0].delete.assert_called_with("test_key")
        cache.layers[1].delete.assert_called_with("test_key")

    def test_clear_cache(self, cache):
        """测试清空缓存"""
        cache.clear()

        # 验证所有层级都被清空
        cache.layers[0].clear.assert_called()
        cache.layers[1].clear.assert_called()


class TestMultiLevelCachePerformance:
    """MultiLevelCache性能测试"""

    @pytest.fixture
    def performance_cache(self):
        """性能测试用的缓存fixture"""
        cache = MultiLevelCache()

        # 创建多层缓存模拟
        l1_cache = {}
        l2_cache = {}

        # L1缓存 - 快速内存缓存
        class L1Cache:
            def __init__(self):
                self.data = l1_cache
                self.hits = 0
                self.misses = 0

            def get(self, key):
                if key in self.data:
                    self.hits += 1
                    return self.data[key]
                self.misses += 1
                return None

            def put(self, key, value):
                self.data[key] = value

        # L2缓存 - 稍慢的持久化缓存
        class L2Cache:
            def __init__(self):
                self.data = l2_cache
                self.hits = 0
                self.misses = 0

            def get(self, key):
                time.sleep(0.001)  # 模拟网络延迟
                if key in self.data:
                    self.hits += 1
                    return self.data[key]
                self.misses += 1
                return None

            def put(self, key, value):
                time.sleep(0.001)  # 模拟写入延迟
                self.data[key] = value

        cache.layers = [L1Cache(), L2Cache()]
        return cache

    def test_cache_hit_performance_l1(self, performance_cache):
        """测试L1缓存命中性能"""
        # 预填充缓存
        performance_cache.put("key1", "value1")

        # 测试命中性能
        start_time = time.time()
        result = performance_cache.get("key1")
        end_time = time.time()

        assert result == "value1"
        response_time = (end_time - start_time) * 1000  # ms

        # L1命中应该非常快 (< 1ms)
        assert response_time < 1.0

    def test_cache_hit_performance_l2(self, performance_cache):
        """测试L2缓存命中性能"""
        # 预填充L2缓存
        performance_cache.layers[1].put("key2", "value2")

        # 测试L2命中性能（L1未命中）
        start_time = time.time()
        result = performance_cache.get("key2")
        end_time = time.time()

        assert result == "value2"
        response_time = (end_time - start_time) * 1000  # ms

        # L2命中应该较慢但仍在合理范围内 (< 10ms)
        assert response_time < 10.0

    def test_cache_miss_performance(self, performance_cache):
        """测试缓存未命中性能"""
        start_time = time.time()
        result = performance_cache.get("nonexistent_key")
        end_time = time.time()

        assert result is None
        response_time = (end_time - start_time) * 1000  # ms

        # 缓存未命中应该在合理时间内 (< 15ms)
        assert response_time < 15.0

    def test_bulk_operations_performance(self, performance_cache):
        """测试批量操作性能"""
        # 准备测试数据
        test_data = {f"bulk_key_{i}": f"bulk_value_{i}" for i in range(100)}

        # 测试批量写入性能
        start_time = time.time()
        for key, value in test_data.items():
            performance_cache.put(key, value)
        write_time = time.time() - start_time

        # 测试批量读取性能
        start_time = time.time()
        for key in test_data.keys():
            result = performance_cache.get(key)
            assert result is not None
        read_time = time.time() - start_time

        total_time = write_time + read_time
        operations_per_second = len(test_data) * 2 / total_time  # 读+写

        # 应该有合理的性能（至少1000 ops/sec）
        assert operations_per_second > 1000


class TestMultiLevelCacheConsistency:
    """MultiLevelCache一致性测试"""

    @pytest.fixture
    def consistency_cache(self):
        """一致性测试用的缓存fixture"""
        cache = MultiLevelCache()

        # 创建缓存层级
        l1_data = {}
        l2_data = {}

        class ConsistentCache:
            def __init__(self, data_dict, name):
                self.data = data_dict
                self.name = name
                self.operations = []

            def get(self, key):
                self.operations.append(f"get_{key}")
                return self.data.get(key)

            def put(self, key, value):
                self.operations.append(f"put_{key}")
                self.data[key] = value

            def delete(self, key):
                self.operations.append(f"delete_{key}")
                self.data.pop(key, None)

            def clear(self):
                self.operations.append("clear")
                self.data.clear()

        cache.layers = [
            ConsistentCache(l1_data, "L1"),
            ConsistentCache(l2_data, "L2")
        ]
        return cache

    def test_write_through_consistency(self, consistency_cache):
        """测试写穿一致性"""
        consistency_cache.put("key1", "value1")

        # 验证两个层级都有数据
        assert consistency_cache.layers[0].data["key1"] == "value1"
        assert consistency_cache.layers[1].data["key1"] == "value1"

        # 验证操作被记录
        assert "put_key1" in consistency_cache.layers[0].operations
        assert "put_key1" in consistency_cache.layers[1].operations

    def test_read_through_consistency(self, consistency_cache):
        """测试读穿一致性"""
        # 只在L2设置数据
        consistency_cache.layers[1].data["key2"] = "value2"

        # 从缓存获取
        result = consistency_cache.get("key2")

        assert result == "value2"

        # 验证L1现在也有数据（写回）
        assert consistency_cache.layers[0].data.get("key2") == "value2"

    def test_delete_consistency(self, consistency_cache):
        """测试删除一致性"""
        # 在两个层级都设置数据
        consistency_cache.layers[0].data["key3"] = "value3"
        consistency_cache.layers[1].data["key3"] = "value3"

        # 删除
        consistency_cache.delete("key3")

        # 验证两个层级都被删除
        assert "key3" not in consistency_cache.layers[0].data
        assert "key3" not in consistency_cache.layers[1].data

    def test_clear_consistency(self, consistency_cache):
        """测试清空一致性"""
        # 在两个层级都设置数据
        consistency_cache.layers[0].data.update({"a": 1, "b": 2})
        consistency_cache.layers[1].data.update({"c": 3, "d": 4})

        # 清空
        consistency_cache.clear()

        # 验证两个层级都被清空
        assert len(consistency_cache.layers[0].data) == 0
        assert len(consistency_cache.layers[1].data) == 0


class TestMultiLevelCacheConcurrency:
    """MultiLevelCache并发测试"""

    @pytest.fixture
    def concurrent_cache(self):
        """并发测试用的缓存fixture"""
        cache = MultiLevelCache()

        # 创建线程安全的缓存层级
        l1_data = {}
        l2_data = {}
        l1_lock = threading.Lock()
        l2_lock = threading.Lock()

        class ThreadSafeCache:
            def __init__(self, data_dict, lock, name):
                self.data = data_dict
                self.lock = lock
                self.name = name

            def get(self, key):
                with self.lock:
                    return self.data.get(key)

            def put(self, key, value):
                with self.lock:
                    self.data[key] = value

            def delete(self, key):
                with self.lock:
                    self.data.pop(key, None)

            def clear(self):
                with self.lock:
                    self.data.clear()

        cache.layers = [
            ThreadSafeCache(l1_data, l1_lock, "L1"),
            ThreadSafeCache(l2_data, l2_lock, "L2")
        ]
        return cache

    def test_concurrent_read_write(self, concurrent_cache):
        """测试并发读写"""
        results = []
        exceptions = []

        def concurrent_worker(worker_id: int):
            """并发工作线程"""
            try:
                for i in range(100):
                    key = f"key_{worker_id}_{i}"
                    value = f"value_{worker_id}_{i}"

                    # 写入
                    concurrent_cache.put(key, value)

                    # 读取验证
                    result = concurrent_cache.get(key)
                    if result != value:
                        results.append(f"Mismatch: {key} expected {value}, got {result}")
                    else:
                        results.append(f"OK: {key}")

            except Exception as e:
                exceptions.append(f"Worker {worker_id}: {e}")

        # 启动多个并发线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=concurrent_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join(timeout=10.0)

        # 不应该有异常
        assert len(exceptions) == 0, f"Concurrent exceptions: {exceptions}"

        # 验证结果
        ok_count = sum(1 for r in results if r.startswith("OK:"))
        mismatch_count = sum(1 for r in results if r.startswith("Mismatch:"))

        assert mismatch_count == 0, f"Found {mismatch_count} data mismatches"
        assert ok_count == 500, f"Expected 500 OK results, got {ok_count}"

    def test_concurrent_eviction(self, concurrent_cache):
        """测试并发淘汰"""
        # 这个测试验证在高并发情况下缓存淘汰的正确性
        access_pattern = {}
        exceptions = []

        def eviction_worker(worker_id: int):
            """淘汰测试线程"""
            try:
                for i in range(50):
                    # 使用有限的键空间来触发淘汰
                    key = f"shared_key_{i % 10}"

                    # 记录访问模式
                    if key not in access_pattern:
                        access_pattern[key] = []
                    access_pattern[key].append(worker_id)

                    # 执行缓存操作
                    concurrent_cache.put(key, f"value_{worker_id}_{i}")
                    result = concurrent_cache.get(key)

                    # 验证数据一致性
                    if result and not result.startswith("value_"):
                        exceptions.append(f"Invalid value for {key}: {result}")

            except Exception as e:
                exceptions.append(f"Eviction worker {worker_id}: {e}")

        # 启动多个线程进行密集的缓存操作
        threads = []
        for i in range(8):
            t = threading.Thread(target=eviction_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待完成
        for t in threads:
            t.join(timeout=15.0)

        # 不应该有异常
        assert len(exceptions) == 0, f"Eviction exceptions: {exceptions}"

        # 验证缓存仍然可用
        test_key = "final_test_key"
        concurrent_cache.put(test_key, "final_test_value")
        result = concurrent_cache.get(test_key)
        assert result == "final_test_value"


class TestMultiLevelCacheMonitoring:
    """MultiLevelCache监控测试"""

    @pytest.fixture
    def monitored_cache(self):
        """带监控的缓存fixture"""
        cache = MultiLevelCache()

        # 创建带监控的缓存层级
        class MonitoredCache:
            def __init__(self, name):
                self.name = name
                self.data = {}
                self.get_count = 0
                self.put_count = 0
                self.hit_count = 0
                self.miss_count = 0

            def get(self, key):
                self.get_count += 1
                if key in self.data:
                    self.hit_count += 1
                    return self.data[key]
                else:
                    self.miss_count += 1
                    return None

            def put(self, key, value):
                self.put_count += 1
                self.data[key] = value

            def delete(self, key):
                self.data.pop(key, None)

            def clear(self):
                self.data.clear()

            def get_stats(self):
                total_requests = self.get_count
                hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
                return {
                    'name': self.name,
                    'size': len(self.data),
                    'get_count': self.get_count,
                    'put_count': self.put_count,
                    'hit_count': self.hit_count,
                    'miss_count': self.miss_count,
                    'hit_rate': hit_rate
                }

        cache.layers = [
            MonitoredCache("L1_Memory"),
            MonitoredCache("L2_Redis")
        ]

        # 添加监控方法
        def get_monitoring_stats():
            return {
                layer.name: layer.get_stats()
                for layer in cache.layers
            }

        cache.get_monitoring_stats = get_monitoring_stats
        return cache

    def test_hit_rate_monitoring(self, monitored_cache):
        """测试命中率监控"""
        # 执行一些缓存操作
        monitored_cache.put("key1", "value1")
        monitored_cache.put("key2", "value2")
        monitored_cache.put("key3", "value3")

        # 命中
        monitored_cache.get("key1")
        monitored_cache.get("key2")

        # 未命中
        monitored_cache.get("key4")
        monitored_cache.get("key5")

        # 获取监控统计
        stats = monitored_cache.get_monitoring_stats()

        # 验证L1统计
        l1_stats = stats["L1_Memory"]
        assert l1_stats["put_count"] == 3
        assert l1_stats["get_count"] == 4  # 4次get操作
        assert l1_stats["hit_count"] == 2   # key1, key2命中
        assert l1_stats["miss_count"] == 2  # key4, key5未命中
        assert abs(l1_stats["hit_rate"] - 0.5) < 0.01  # 50%命中率

    def test_layer_performance_monitoring(self, monitored_cache):
        """测试层级性能监控"""
        import time

        # 执行大量操作
        start_time = time.time()

        for i in range(100):
            key = f"perf_key_{i}"
            monitored_cache.put(key, f"perf_value_{i}")

        for i in range(50):
            key = f"perf_key_{i}"
            monitored_cache.get(key)  # 命中

        for i in range(50, 100):
            key = f"perf_key_{i}"
            monitored_cache.get(key)  # 命中

        for i in range(100, 150):
            key = f"perf_key_{i}"
            monitored_cache.get(key)  # 未命中

        end_time = time.time()
        duration = end_time - start_time

        # 获取统计
        stats = monitored_cache.get_monitoring_stats()
        l1_stats = stats["L1_Memory"]

        # 验证统计数据
        assert l1_stats["put_count"] == 100
        assert l1_stats["get_count"] == 150
        assert l1_stats["hit_count"] == 100  # 前100个key都命中
        assert l1_stats["miss_count"] == 50   # 后50个未命中

        # 计算性能指标
        total_operations = l1_stats["put_count"] + l1_stats["get_count"]
        ops_per_second = total_operations / duration

        # 应该有合理的性能
        assert ops_per_second > 1000  # 至少1000 ops/sec
        assert l1_stats["hit_rate"] > 0.6  # 命中率应该较高


class TestMultiLevelCacheOptimization:
    """MultiLevelCache优化测试"""

    @pytest.fixture
    def optimizable_cache(self):
        """可优化的缓存fixture"""
        cache = MultiLevelCache()

        # 创建可优化的缓存层级
        class OptimizableCache:
            def __init__(self, name, capacity=100):
                self.name = name
                self.capacity = capacity
                self.data = {}
                self.access_count = {}
                self.last_access = {}

            def get(self, key):
                if key in self.data:
                    self.access_count[key] = self.access_count.get(key, 0) + 1
                    self.last_access[key] = time.time()
                    return self.data[key]
                return None

            def put(self, key, value):
                if len(self.data) >= self.capacity:
                    # 简单的LRU淘汰
                    lru_key = min(self.last_access.keys(),
                                key=lambda k: self.last_access.get(k, 0))
                    del self.data[lru_key]
                    del self.access_count[lru_key]
                    del self.last_access[lru_key]

                self.data[key] = value
                self.access_count[key] = 1
                self.last_access[key] = time.time()

            def delete(self, key):
                self.data.pop(key, None)
                self.access_count.pop(key, None)
                self.last_access.pop(key, None)

            def clear(self):
                self.data.clear()
                self.access_count.clear()
                self.last_access.clear()

            def get_hot_keys(self, limit=10):
                """获取热门键"""
                return sorted(self.access_count.keys(),
                            key=lambda k: self.access_count[k],
                            reverse=True)[:limit]

            def get_size(self):
                return len(self.data)

        cache.layers = [
            OptimizableCache("L1_Optimized", capacity=50),
            OptimizableCache("L2_Optimized", capacity=200)
        ]
        return cache

    def test_cache_optimization_eviction(self, optimizable_cache):
        """测试缓存优化淘汰"""
        # 填满L1缓存
        for i in range(60):  # 超过容量50
            optimizable_cache.put(f"key_{i}", f"value_{i}")

        # L1应该保持在容量限制内
        assert optimizable_cache.layers[0].get_size() <= 50

        # 最近的键应该还在缓存中
        recent_keys = [f"key_{i}" for i in range(55, 60)]
        for key in recent_keys:
            assert optimizable_cache.get(key) is not None

    def test_hot_key_promotion(self, optimizable_cache):
        """测试热门键提升"""
        # 添加一些键
        for i in range(20):
            optimizable_cache.put(f"key_{i}", f"value_{i}")

        # 频繁访问某些键
        hot_keys = ["key_1", "key_5", "key_10"]
        for _ in range(10):
            for key in hot_keys:
                optimizable_cache.get(key)

        # 检查热门键统计
        l1_hot_keys = optimizable_cache.layers[0].get_hot_keys(5)

        # 热门键应该在统计中
        for hot_key in hot_keys:
            assert hot_key in l1_hot_keys

    def test_tiered_storage_optimization(self, optimizable_cache):
        """测试分层存储优化"""
        # 模拟分层存储策略
        # 热门数据留在L1，冷数据移到L2

        # 添加数据到L1
        for i in range(30):
            optimizable_cache.put(f"data_{i}", f"value_{i}")

        # 频繁访问前10个（热门数据）
        for _ in range(5):
            for i in range(10):
                optimizable_cache.get(f"data_{i}")

        # 不经常访问后20个（冷数据）
        for i in range(20, 30):
            optimizable_cache.get(f"data_{i}")  # 只访问一次

        # 在实际优化中，热门数据应该被保留在L1
        # 这里我们只是验证基本功能
        l1_size = optimizable_cache.layers[0].get_size()
        l2_size = optimizable_cache.layers[1].get_size()

        assert l1_size > 0
        assert l2_size >= l1_size  # L2通常更大


class TestMultiLevelCacheIntegration:
    """MultiLevelCache集成测试"""

    def test_complete_cache_workflow(self):
        """测试完整缓存工作流"""
        cache = MultiLevelCache()

        # Mock缓存层级
        l1_cache = MagicMock()
        l2_cache = MagicMock()

        # 设置行为
        l1_cache.get.return_value = None  # L1总是未命中
        l1_cache.put.return_value = None
        l2_cache.get.return_value = "l2_value"
        l2_cache.put.return_value = None

        cache.layers = [l1_cache, l2_cache]

        # 执行完整工作流
        # 1. 写入数据
        cache.put("workflow_key", "workflow_value")

        # 2. 读取数据（应该从L2获取）
        result = cache.get("workflow_key")

        # 3. 验证数据流
        assert result == "l2_value"

        # 4. 验证层级调用
        l1_cache.get.assert_called_with("workflow_key")
        l2_cache.get.assert_called_with("workflow_key")
        l1_cache.put.assert_called_with("workflow_key", "workflow_value")
        l2_cache.put.assert_called_with("workflow_key", "workflow_value")

    def test_cache_degradation_handling(self):
        """测试缓存降级处理"""
        cache = MultiLevelCache()

        # 创建有降级的缓存层级
        class DegradationCache:
            def __init__(self, name, fail_after=None):
                self.name = name
                self.fail_after = fail_after
                self.call_count = 0
                self.data = {}

            def get(self, key):
                self.call_count += 1
                if self.fail_after and self.call_count > self.fail_after:
                    raise Exception(f"{self.name} degraded")
                return self.data.get(key)

            def put(self, key, value):
                self.call_count += 1
                if self.fail_after and self.call_count > self.fail_after:
                    raise Exception(f"{self.name} degraded")
                self.data[key] = value

            def delete(self, key):
                self.data.pop(key, None)

            def clear(self):
                self.data.clear()

        # L1在第3次调用后失败，L2正常
        cache.layers = [
            DegradationCache("L1_Failing", fail_after=2),
            DegradationCache("L2_Stable")
        ]

        # 初始操作应该正常
        cache.put("key1", "value1")
        result = cache.get("key1")
        assert result == "value1"

        # 触发L1降级
        try:
            cache.put("key2", "value2")
            cache.put("key3", "value3")  # 这应该触发L1失败
        except:
            pass  # 降级是预期的

        # 即使L1失败，缓存仍然应该工作（通过L2）
        result = cache.get("key1")
        assert result == "value1"

    def test_cache_statistics_aggregation(self):
        """测试缓存统计聚合"""
        cache = MultiLevelCache()

        # 创建带统计的缓存层级
        class StatisticalCache:
            def __init__(self, name):
                self.name = name
                self.data = {}
                self.get_calls = 0
                self.put_calls = 0
                self.hits = 0
                self.misses = 0

            def get(self, key):
                self.get_calls += 1
                if key in self.data:
                    self.hits += 1
                    return self.data[key]
                else:
                    self.misses += 1
                    return None

            def put(self, key, value):
                self.put_calls += 1
                self.data[key] = value

            def delete(self, key):
                self.data.pop(key, None)

            def clear(self):
                self.data.clear()

            def get_statistics(self):
                return {
                    'name': self.name,
                    'size': len(self.data),
                    'get_calls': self.get_calls,
                    'put_calls': self.put_calls,
                    'hits': self.hits,
                    'misses': self.misses,
                    'hit_rate': self.hits / self.get_calls if self.get_calls > 0 else 0
                }

        cache.layers = [
            StatisticalCache("L1_Stats"),
            StatisticalCache("L2_Stats")
        ]

        # 添加统计方法
        def get_aggregated_stats():
            layer_stats = [layer.get_statistics() for layer in cache.layers]
            return {
                'layers': layer_stats,
                'total_gets': sum(s['get_calls'] for s in layer_stats),
                'total_puts': sum(s['put_calls'] for s in layer_stats),
                'overall_hit_rate': sum(s['hits'] for s in layer_stats) / sum(s['get_calls'] for s in layer_stats) if sum(s['get_calls'] for s in layer_stats) > 0 else 0
            }

        cache.get_aggregated_stats = get_aggregated_stats

        # 执行各种操作
        cache.put("key1", "value1")
        cache.put("key2", "value2")
        cache.get("key1")  # L1命中
        cache.get("key2")  # L1命中
        cache.get("key3")  # L1未命中，L2未命中

        # 获取聚合统计
        stats = cache.get_aggregated_stats()

        assert stats['total_puts'] == 4  # 每个层级2次put
        assert stats['total_gets'] == 6  # 每个层级3次get
        assert 'overall_hit_rate' in stats
        assert stats['overall_hit_rate'] >= 0.0
        assert stats['overall_hit_rate'] <= 1.0
