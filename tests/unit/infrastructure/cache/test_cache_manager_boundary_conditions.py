#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存管理器边界条件和异常场景测试

测试目标：提升cache_manager.py的边界条件和异常场景覆盖率
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock, call
from typing import Dict, Any

from src.infrastructure.cache.core.cache_manager import (
    UnifiedCacheManager,
    CacheConfig,
    create_unified_cache,
    create_memory_cache,
    create_redis_cache,
    create_hybrid_cache
)


class TestCacheConfigBoundaryConditions:
    """测试CacheConfig的边界条件"""

    def test_cache_config_zero_values(self):
        """测试配置零值"""
        config = CacheConfig(enabled=True, max_size=0, ttl=0)

        assert config.enabled == True
        assert config.max_size == 0
        assert config.ttl == 0

    def test_cache_config_negative_values(self):
        """测试配置负值"""
        config = CacheConfig(enabled=True, max_size=-100, ttl=-300)

        assert config.max_size == -100
        assert config.ttl == -300

    def test_cache_config_extreme_values(self):
        """测试配置极端值"""
        config = CacheConfig(
            enabled=True,
            max_size=999999999,
            ttl=2147483647  # 32-bit int max
        )

        assert config.max_size == 999999999
        assert config.ttl == 2147483647


class TestUnifiedCacheManagerBoundaryConditions:
    """测试UnifiedCacheManager的边界条件"""

    @pytest.fixture
    def cache_manager(self):
        """创建缓存管理器fixture"""
        config = CacheConfig(enabled=True, max_size=10, ttl=60)
        manager = UnifiedCacheManager(config)
        yield manager
        manager.shutdown()

    def test_get_with_none_key(self, cache_manager):
        """测试获取None键"""
        result = cache_manager.get(None)
        assert result is None
        assert cache_manager._stats["misses"] == 1

    def test_get_with_empty_string_key(self, cache_manager):
        """测试获取空字符串键"""
        result = cache_manager.get("")
        assert result is None
        assert cache_manager._stats["misses"] == 1

    def test_get_with_unicode_key(self, cache_manager):
        """测试获取Unicode键"""
        key = "测试键🚀"
        value = "测试值"

        cache_manager.set(key, value)
        result = cache_manager.get(key)

        assert result == value
        assert cache_manager._stats["hits"] == 1

    def test_get_with_very_long_key(self, cache_manager):
        """测试获取超长键"""
        long_key = "x" * 10000
        value = "test_value"

        cache_manager.set(long_key, value)
        result = cache_manager.get(long_key)

        assert result == value
        assert cache_manager._stats["hits"] == 1

    def test_set_with_none_value(self, cache_manager):
        """测试设置None值"""
        cache_manager.set("key", None)
        result = cache_manager.get("key")

        assert result is None
        assert cache_manager._stats["hits"] == 1

    def test_set_with_complex_object(self, cache_manager):
        """测试设置复杂对象"""
        complex_obj = {
            "nested": {
                "list": [1, 2, {"deep": "value"}],
                "tuple": (1, 2, 3),
                "set": {1, 2, 3}
            },
            "function": lambda x: x * 2
        }

        cache_manager.set("complex", complex_obj)
        result = cache_manager.get("complex")

        assert result == complex_obj
        assert result["nested"]["list"][2]["deep"] == "value"

    def test_set_with_circular_reference(self, cache_manager):
        """测试设置循环引用对象"""
        obj1 = {"name": "obj1"}
        obj2 = {"name": "obj2", "ref": obj1}
        obj1["ref"] = obj2  # 创建循环引用

        # 这应该不会抛出异常
        cache_manager.set("circular", obj1)
        result = cache_manager.get("circular")

        assert result is not None
        assert result["name"] == "obj1"
        assert result["ref"]["name"] == "obj2"

    def test_stats_overflow_protection(self, cache_manager):
        """测试统计信息溢出保护"""
        # 模拟大量操作
        for i in range(10000):
            cache_manager.get(f"key_{i}")

        stats = cache_manager.get_stats()
        assert isinstance(stats["misses"], int)
        assert stats["misses"] == 10000

    def test_concurrent_access_boundary(self, cache_manager):
        """测试并发访问边界条件"""
        results = []
        errors = []

        def worker(worker_id):
            try:
                for i in range(100):
                    key = f"worker_{worker_id}_key_{i}"
                    cache_manager.set(key, f"value_{i}")
                    result = cache_manager.get(key)
                    if result != f"value_{i}":
                        errors.append(f"Worker {worker_id}: expected {f'value_{i}'}, got {result}")
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # 启动多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Concurrent access errors: {errors}"

    def test_memory_pressure_simulation(self, cache_manager):
        """测试内存压力模拟"""
        # 添加大量数据
        large_data = "x" * 10000  # 10KB字符串

        for i in range(100):
            cache_manager.set(f"large_key_{i}", large_data)

        # 验证所有数据都能正确存储和检索
        for i in range(100):
            result = cache_manager.get(f"large_key_{i}")
            assert result == large_data

        assert cache_manager.size() == 100

    def test_shutdown_and_restart(self, cache_manager):
        """测试关闭和重启"""
        # 添加一些数据
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")

        assert cache_manager.size() == 2

        # 关闭
        cache_manager.shutdown()

        assert cache_manager.size() == 0
        assert cache_manager.get_stats() == {"hits": 0, "misses": 0}

        # 重启后应该可以正常使用
        cache_manager.set("key3", "value3")
        assert cache_manager.get("key3") == "value3"


class TestFactoryFunctionsBoundaryConditions:
    """测试工厂函数的边界条件"""

    def test_create_unified_cache_with_none_config(self):
        """测试使用None配置创建统一缓存"""
        manager = create_unified_cache(None)

        assert isinstance(manager, UnifiedCacheManager)
        assert manager.config.enabled == True
        assert manager.config.max_size == 1000
        assert manager.config.ttl == 300

    def test_create_memory_cache_boundary_values(self):
        """测试创建内存缓存的边界值"""
        # 零大小
        manager = create_memory_cache(max_size=0, ttl=0)
        assert manager.config.max_size == 0
        assert manager.config.ttl == 0

        # 负值
        manager = create_memory_cache(max_size=-100, ttl=-50)
        assert manager.config.max_size == -100
        assert manager.config.ttl == -50

        # 超大值
        manager = create_memory_cache(max_size=1000000, ttl=31536000)  # 1年
        assert manager.config.max_size == 1000000
        assert manager.config.ttl == 31536000

    def test_create_redis_cache_boundary_values(self):
        """测试创建Redis缓存的边界值"""
        # 边界端口号
        manager = create_redis_cache(host="localhost", port=0, max_size=0, ttl=0)
        assert manager.config.max_size == 0
        assert manager.config.ttl == 0

        # 超大端口号
        manager = create_redis_cache(host="localhost", port=65535, max_size=1000000, ttl=86400)
        assert manager.config.max_size == 1000000
        assert manager.config.ttl == 86400

    def test_create_hybrid_cache_boundary_values(self):
        """测试创建混合缓存的边界值"""
        manager = create_hybrid_cache(
            memory_size=0,
            redis_host="",
            redis_port=-1,
            ttl=-100
        )

        assert manager.config.max_size == 0
        assert manager.config.ttl == -100

    def test_factory_functions_return_types(self):
        """测试工厂函数返回类型"""
        manager1 = create_unified_cache()
        manager2 = create_memory_cache()
        manager3 = create_redis_cache()
        manager4 = create_hybrid_cache()

        assert all(isinstance(m, UnifiedCacheManager) for m in [manager1, manager2, manager3, manager4])


class TestUnifiedCacheManagerExceptionScenarios:
    """测试UnifiedCacheManager的异常场景"""

    @pytest.fixture
    def cache_manager(self):
        """创建缓存管理器fixture"""
        config = CacheConfig(enabled=True, max_size=10, ttl=60)
        manager = UnifiedCacheManager(config)
        yield manager
        manager.shutdown()

    def test_get_with_invalid_key_type(self, cache_manager):
        """测试使用无效键类型"""
        # 数字键
        cache_manager.set(123, "value")
        result = cache_manager.get(123)
        assert result == "value"

        # 元组键（可哈希）
        key_tuple = ("type", "complex")
        cache_manager.set(key_tuple, "tuple_value")
        result = cache_manager.get(key_tuple)
        assert result == "tuple_value"

        # 列表键应该抛出异常
        with pytest.raises(TypeError, match="unhashable type"):
            cache_manager.set([1, 2, 3], "list_value")

        # 字典键也应该抛出异常
        with pytest.raises(TypeError, match="unhashable type"):
            cache_manager.set({"type": "complex"}, "dict_value")

    def test_set_with_unhashable_key(self, cache_manager):
        """测试使用不可哈希的键"""
        # 列表作为键（在Python中列表不可哈希）
        with pytest.raises(TypeError):
            cache_manager._cache[[1, 2, 3]] = "value"

    def test_monitoring_status_with_corrupted_stats(self, cache_manager):
        """测试监控状态在统计信息损坏时的表现"""
        # 手动损坏统计信息
        cache_manager._stats = None

        # 这应该不会崩溃
        with patch.object(cache_manager, 'get_stats', return_value={"hits": 0, "misses": 0}):
            status = cache_manager.get_monitoring_status()
            assert "size" in status
            assert "stats" in status

    def test_size_calculation_accuracy(self, cache_manager):
        """测试大小计算准确性"""
        # 添加不同数量的项目
        for i in range(5):
            cache_manager.set(f"key_{i}", f"value_{i}")

        assert cache_manager.size() == 5

        # 删除一些项目
        del cache_manager._cache["key_2"]
        del cache_manager._cache["key_4"]

        assert cache_manager.size() == 3

        # 清空
        cache_manager.clear()
        assert cache_manager.size() == 0

    def test_stats_isolation(self, cache_manager):
        """测试统计信息隔离"""
        # 获取统计信息的副本
        stats1 = cache_manager.get_stats()

        # 修改缓存操作
        cache_manager.get("nonexistent")
        cache_manager.set("key", "value")
        cache_manager.get("key")

        # 再次获取统计信息
        stats2 = cache_manager.get_stats()

        # 验证副本没有被修改
        assert stats1 != stats2
        assert stats1["misses"] == 0
        assert stats2["misses"] == 1
        assert stats2["hits"] == 1

    def test_thread_safety_under_load(self, cache_manager):
        """测试高负载下的线程安全性"""
        import concurrent.futures

        def stress_worker(worker_id):
            results = []
            for i in range(1000):
                key = f"worker_{worker_id}_{i}"
                cache_manager.set(key, f"value_{i}")
                result = cache_manager.get(key)
                results.append(result == f"value_{i}")
            return all(results)

        # 使用线程池执行器
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(stress_worker, i) for i in range(4)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        assert all(results), "Some threads failed the stress test"

    def test_memory_cleanup_on_shutdown(self, cache_manager):
        """测试关闭时的内存清理"""
        # 添加大量数据
        for i in range(1000):
            cache_manager.set(f"key_{i}", f"value_{i}" * 100)  # 创建大对象

        initial_size = cache_manager.size()
        assert initial_size == 1000

        # 关闭应该清理所有数据
        cache_manager.shutdown()

        assert cache_manager.size() == 0
        assert len(cache_manager._cache) == 0

        # 统计信息也应该重置
        stats = cache_manager.get_stats()
        assert stats["hits"] == 0
        assert stats["misses"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
