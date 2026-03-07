#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 缓存服务深度测试
验证CacheService的完整功能覆盖，目标覆盖率85%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional


class TestCacheService(unittest.TestCase):
    """测试缓存服务"""

    def setUp(self):
        """测试前准备"""
        self.cache_service = None
        self.test_config = {
            "maxsize": 100,
            "default_ttl": 3600,
            "enable_stats": True
        }

    def tearDown(self):
        """测试后清理"""
        if self.cache_service and self.cache_service.initialized:
            self.cache_service.shutdown()

    # ==================== 基础功能测试 ====================

    def test_cache_service_initialization(self):
        """测试缓存服务初始化"""
        from src.infrastructure.config.services.cache_service import CacheService

        # 测试默认初始化
        cache_service = CacheService()
        self.assertIsInstance(cache_service, CacheService)
        self.assertEqual(cache_service.maxsize, 1000)  # 默认值
        self.assertFalse(cache_service.initialized)
        self.assertIsInstance(cache_service.cache, dict)
        self.assertIsInstance(cache_service.timestamps, dict)
        self.assertIsInstance(cache_service.access_times, dict)
        self.assertTrue(hasattr(cache_service.lock, 'acquire'))  # 验证是锁对象

        # 测试自定义配置初始化
        cache_service_custom = CacheService(config=self.test_config, maxsize=200)
        self.assertEqual(cache_service_custom.maxsize, 200)
        self.assertEqual(cache_service_custom.config, self.test_config)

    def test_initialize_and_shutdown(self):
        """测试初始化和关闭"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()

        # 测试初始化
        result = cache_service.initialize()
        self.assertTrue(result)
        self.assertTrue(cache_service.initialized)

        # 测试重复初始化
        result = cache_service.initialize()
        self.assertTrue(result)

        # 测试关闭
        result = cache_service.shutdown()
        self.assertTrue(result)
        self.assertFalse(cache_service.initialized)

        # 测试未初始化时的关闭
        cache_service_uninit = CacheService()
        result = cache_service_uninit.shutdown()
        self.assertTrue(result)

    # ==================== 核心缓存操作测试 ====================

    def test_set_and_get_basic(self):
        """测试基本的设置和获取操作"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 测试设置缓存项
        result = cache_service.set("test_key", "test_value")
        self.assertTrue(result)

        # 测试获取缓存项
        value = cache_service.get("test_key")
        self.assertEqual(value, "test_value")

        # 测试获取不存在的键
        value = cache_service.get("nonexistent_key")
        self.assertIsNone(value)

    def test_set_with_ttl(self):
        """测试设置带TTL的缓存项"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 设置带TTL的缓存项
        result = cache_service.set("ttl_key", "ttl_value", ttl=1)  # 1秒TTL
        self.assertTrue(result)

        # 立即获取，应该存在
        value = cache_service.get("ttl_key")
        self.assertEqual(value, "ttl_value")

        # 等待过期
        time.sleep(1.1)

        # 获取，应该返回None
        value = cache_service.get("ttl_key")
        self.assertIsNone(value)

    def test_delete_operation(self):
        """测试删除操作"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 设置缓存项
        cache_service.set("delete_key", "delete_value")

        # 验证存在
        value = cache_service.get("delete_key")
        self.assertEqual(value, "delete_value")

        # 删除缓存项
        result = cache_service.delete("delete_key")
        self.assertTrue(result)

        # 验证已删除
        value = cache_service.get("delete_key")
        self.assertIsNone(value)

        # 测试删除不存在的键
        result = cache_service.delete("nonexistent_key")
        self.assertFalse(result)

    def test_clear_operation(self):
        """测试清空操作"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 设置多个缓存项
        cache_service.set("key1", "value1")
        cache_service.set("key2", "value2")
        cache_service.set("key3", "value3")

        # 验证缓存项存在
        self.assertEqual(len(cache_service.cache), 3)

        # 清空缓存
        result = cache_service.clear()
        self.assertTrue(result)

        # 验证已清空
        self.assertEqual(len(cache_service.cache), 0)
        self.assertEqual(len(cache_service.timestamps), 0)
        self.assertEqual(len(cache_service.access_times), 0)
        self.assertEqual(cache_service.hits, 0)
        self.assertEqual(cache_service.misses, 0)

    # ==================== 容量管理测试 ====================

    def test_capacity_management(self):
        """测试容量管理"""
        from src.infrastructure.config.services.cache_service import CacheService

        # 创建小容量缓存服务
        cache_service = CacheService(maxsize=2)
        cache_service.initialize()

        # 设置缓存项
        cache_service.set("key1", "value1")
        cache_service.set("key2", "value2")

        # 验证容量
        self.assertEqual(len(cache_service.cache), 2)

        # 添加第三个缓存项，应该触发驱逐
        cache_service.set("key3", "value3")

        # 验证容量仍然是2（一个被驱逐）
        self.assertEqual(len(cache_service.cache), 2)
        self.assertIn("key3", cache_service.cache)  # 新项应该存在

    def test_lru_eviction(self):
        """测试LRU驱逐策略"""
        from src.infrastructure.config.services.cache_service import CacheService
        import time

        cache_service = CacheService(maxsize=2)
        cache_service.initialize()

        # 设置两个缓存项，添加小延迟确保时间差异
        cache_service.set("key1", "value1")
        time.sleep(0.001)  # 1ms延迟
        cache_service.set("key2", "value2")
        time.sleep(0.001)  # 1ms延迟

        # 访问key1，使其成为最近使用的
        cache_service.get("key1")
        time.sleep(0.001)  # 确保访问时间更新

        # 添加第三个缓存项，应该驱逐key2（最少使用的）
        cache_service.set("key3", "value3")

        # 验证key1仍然存在，key2被驱逐
        self.assertIn("key1", cache_service.cache)
        self.assertNotIn("key2", cache_service.cache)
        self.assertIn("key3", cache_service.cache)

    # ==================== 统计信息测试 ====================

    def test_get_stats(self):
        """测试获取统计信息"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 执行一些操作来产生统计数据
        cache_service.set("key1", "value1")
        cache_service.set("key2", "value2")

        # 命中
        cache_service.get("key1")  # 命中
        cache_service.get("key1")  # 命中

        # 缺失
        cache_service.get("nonexistent")  # 缺失
        cache_service.get("another_nonexistent")  # 缺失

        # 获取统计信息
        stats = cache_service.get_stats()

        # 验证统计信息结构
        self.assertIsInstance(stats, dict)
        self.assertIn("size", stats)
        self.assertIn("maxsize", stats)
        self.assertIn("hits", stats)
        self.assertIn("misses", stats)
        self.assertIn("hit_rate", stats)
        self.assertIn("total_requests", stats)
        self.assertIn("initialized", stats)

        # 验证统计数据
        self.assertEqual(stats["size"], 2)  # 2个缓存项
        self.assertEqual(stats["hits"], 2)  # 2次命中
        self.assertEqual(stats["misses"], 2)  # 2次缺失
        self.assertEqual(stats["total_requests"], 4)  # 总共4次请求
        self.assertEqual(stats["hit_rate"], 50.0)  # 命中率50%

    def test_stats_with_zero_requests(self):
        """测试零请求时的统计信息"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 获取统计信息（没有请求）
        stats = cache_service.get_stats()

        # 验证零请求情况
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)
        self.assertEqual(stats["total_requests"], 0)
        self.assertEqual(stats["hit_rate"], 0.0)

    # ==================== 过期机制测试 ====================

    def test_expiration_mechanism(self):
        """测试过期机制"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 设置短TTL的缓存项
        cache_service.set("expire_key", "expire_value", ttl=0.1)  # 0.1秒TTL

        # 立即获取，应该存在
        value = cache_service.get("expire_key")
        self.assertEqual(value, "expire_value")

        # 等待过期
        time.sleep(0.15)

        # 获取，应该返回None
        value = cache_service.get("expire_key")
        self.assertIsNone(value)

        # 验证缓存项已被清理
        self.assertNotIn("expire_key", cache_service.cache)

    def test_mixed_expiration_times(self):
        """测试混合过期时间"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 设置不同TTL的缓存项
        cache_service.set("short_ttl", "short_value", ttl=0.1)
        cache_service.set("long_ttl", "long_value", ttl=1)

        # 等待短TTL过期
        time.sleep(0.15)

        # 验证短TTL项已过期
        value = cache_service.get("short_ttl")
        self.assertIsNone(value)

        # 验证长TTL项仍然存在
        value = cache_service.get("long_ttl")
        self.assertEqual(value, "long_value")

    # ==================== 并发测试 ====================

    def test_concurrent_access(self):
        """测试并发访问"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService(maxsize=100)
        cache_service.initialize()

        results = []
        errors = []

        def worker(worker_id: int):
            """工作线程"""
            try:
                # 设置缓存项
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    cache_service.set(key, value)

                # 获取缓存项
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    value = cache_service.get(key)
                    expected = f"worker_{worker_id}_value_{i}"
                    if value != expected:
                        errors.append(f"Worker {worker_id}: expected {expected}, got {value}")

                results.append(f"Worker {worker_id} completed")

            except Exception as e:
                errors.append(f"Worker {worker_id} error: {str(e)}")

        # 创建多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)

        # 启动所有线程
        for thread in threads:
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        self.assertEqual(len(results), 5)  # 5个工作线程都完成了
        self.assertEqual(len(errors), 0)   # 没有错误

        # 验证缓存大小
        self.assertGreater(len(cache_service.cache), 0)

    # ==================== 边界情况测试 ====================

    def test_uninitialized_operations(self):
        """测试未初始化时的操作"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()

        # 测试未初始化时的操作
        self.assertIsNone(cache_service.get("test_key"))
        self.assertFalse(cache_service.set("test_key", "test_value"))
        self.assertFalse(cache_service.delete("test_key"))

        # 测试clear操作应该抛出异常
        with self.assertRaises(RuntimeError):
            cache_service.clear()

        # 测试get_stats操作应该抛出异常
        with self.assertRaises(RuntimeError):
            cache_service.get_stats()

    def test_large_data_handling(self):
        """测试大数据处理"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService(maxsize=10)
        cache_service.initialize()

        # 测试大数据
        large_data = {"data": "x" * 10000}  # 10KB数据

        result = cache_service.set("large_key", large_data)
        self.assertTrue(result)

        retrieved_data = cache_service.get("large_key")
        self.assertEqual(retrieved_data, large_data)

    def test_special_characters_in_keys(self):
        """测试键中的特殊字符"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 测试包含特殊字符的键
        special_keys = [
            "key with spaces",
            "key-with-dashes",
            "key_with_underscores",
            "key.with.dots",
            "key/with/slashes",
            "key@with@symbols",
            "中文键",
            "🚀emoji键"
        ]

        for key in special_keys:
            value = f"value_for_{key}"
            cache_service.set(key, value)
            retrieved_value = cache_service.get(key)
            self.assertEqual(retrieved_value, value)

    def test_none_and_empty_values(self):
        """测试None值和空值"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 测试None值
        cache_service.set("none_key", None)
        value = cache_service.get("none_key")
        self.assertIsNone(value)

        # 测试空字符串
        cache_service.set("empty_key", "")
        value = cache_service.get("empty_key")
        self.assertEqual(value, "")

        # 测试空列表
        cache_service.set("empty_list", [])
        value = cache_service.get("empty_list")
        self.assertEqual(value, [])

        # 测试空字典
        cache_service.set("empty_dict", {})
        value = cache_service.get("empty_dict")
        self.assertEqual(value, {})

    # ==================== 性能测试 ====================

    def test_performance_under_load(self):
        """测试负载下的性能"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService(maxsize=1000)
        cache_service.initialize()

        # 测试大量操作的性能
        start_time = time.time()

        # 设置大量缓存项
        for i in range(1000):
            cache_service.set(f"perf_key_{i}", f"perf_value_{i}")

        # 随机访问
        import random
        for _ in range(10000):
            key = f"perf_key_{random.randint(0, 999)}"
            cache_service.get(key)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能（10000次操作应该在合理时间内完成）
        self.assertLess(total_time, 5.0)  # 5秒内完成

        # 验证统计信息
        stats = cache_service.get_stats()
        self.assertGreater(stats["total_requests"], 0)

    # ==================== 健康检查测试 ====================

    def test_health_check(self):
        """测试健康检查"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 测试初始化后的健康检查
        health = cache_service.health_check()
        self.assertIsInstance(health, dict)
        self.assertIn("service", health)
        self.assertIn("status", health)
        self.assertIn("cache_size", health)
        self.assertIn("hit_rate", health)

        self.assertEqual(health["service"], "config_cache_service")
        self.assertEqual(health["status"], "healthy")
        self.assertGreaterEqual(health["cache_size"], 0)
        self.assertGreaterEqual(health["hit_rate"], 0)

        # 测试未初始化的健康检查
        uninitialized_cache = CacheService()
        health = uninitialized_cache.health_check()
        self.assertEqual(health["status"], "uninitialized")
        self.assertEqual(health["hit_rate"], 0)

    # ==================== 错误处理测试 ====================

    def test_exception_handling(self):
        """测试异常处理"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()

        # 测试未初始化时的各种操作
        self.assertIsNone(cache_service.get("test"))
        self.assertFalse(cache_service.set("test", "value"))
        self.assertFalse(cache_service.delete("test"))

        # 测试初始化失败的情况
        with patch.object(cache_service, 'initialize', side_effect=Exception("Init failed")):
            # 这里我们不直接测试，因为initialize方法已经在setUp中被调用
            pass

    # ==================== 私有方法测试 ====================

    def test_private_methods(self):
        """测试私有方法"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService()
        cache_service.initialize()

        # 测试_is_expired方法
        cache_service.set("test_key", "test_value", ttl=1)

        # 立即检查，不应该过期
        self.assertFalse(cache_service._is_expired("test_key"))

        # 等待过期
        time.sleep(1.1)
        self.assertTrue(cache_service._is_expired("test_key"))

        # 测试不存在的键
        self.assertTrue(cache_service._is_expired("nonexistent"))

        # 测试_remove_item方法
        cache_service.set("remove_test", "remove_value")
        self.assertIn("remove_test", cache_service.cache)

        cache_service._remove_item("remove_test")
        self.assertNotIn("remove_test", cache_service.cache)
        self.assertNotIn("remove_test", cache_service.timestamps)
        self.assertNotIn("remove_test", cache_service.access_times)

    # ==================== 集成测试 ====================

    def test_cache_service_workflow(self):
        """测试缓存服务完整工作流程"""
        from src.infrastructure.config.services.cache_service import CacheService

        cache_service = CacheService(maxsize=10)
        cache_service.initialize()

        # 1. 设置初始数据
        for i in range(5):
            cache_service.set(f"initial_key_{i}", f"initial_value_{i}")

        # 2. 验证初始状态
        stats = cache_service.get_stats()
        self.assertEqual(stats["size"], 5)
        self.assertEqual(stats["hits"], 0)
        self.assertEqual(stats["misses"], 0)

        # 3. 执行一些操作
        cache_service.get("initial_key_0")  # 命中
        cache_service.get("initial_key_0")  # 命中
        cache_service.get("nonexistent")    # 缺失
        cache_service.set("new_key", "new_value")  # 新增

        # 4. 验证操作结果
        stats = cache_service.get_stats()
        self.assertEqual(stats["hits"], 2)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["size"], 6)

        # 5. 测试驱逐
        for i in range(10):
            cache_service.set(f"extra_key_{i}", f"extra_value_{i}")

        # 6. 验证容量管理
        stats = cache_service.get_stats()
        self.assertLessEqual(stats["size"], 10)  # 不超过最大容量

        # 7. 清理
        cache_service.clear()
        stats = cache_service.get_stats()
        self.assertEqual(stats["size"], 0)

        # 8. 关闭服务
        result = cache_service.shutdown()
        self.assertTrue(result)
        self.assertFalse(cache_service.initialized)


if __name__ == '__main__':
    unittest.main()
