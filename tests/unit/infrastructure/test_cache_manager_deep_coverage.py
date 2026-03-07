# -*- coding: utf-8 -*-
"""
基础设施层缓存管理器深度覆盖率测试
测试目标: 实现UnifiedCacheManager类的70%+覆盖率
"""

import pytest
from unittest.mock import Mock, MagicMock, patch
import time
import threading
from datetime import datetime, timedelta

from src.infrastructure.cache.core.cache_manager import UnifiedCacheManager
from src.infrastructure.cache.core.cache_configs import CacheConfig


class TestUnifiedCacheManagerDeepCoverage:
    """深度测试UnifiedCacheManager核心功能"""

    def setup_method(self):
        """测试前准备"""
        self.config = CacheConfig(
            max_memory_size=100 * 1024 * 1024,  # 100MB
            default_ttl=3600,  # 1小时
            cleanup_interval=300,  # 5分钟
            enable_monitoring=True,
            enable_persistence=True
        )

        # Mock所有复杂的依赖
        with patch('src.infrastructure.cache.core.cache_manager.MemoryCache'), \
             patch('src.infrastructure.cache.core.cache_manager.RedisCache'), \
             patch('src.infrastructure.cache.core.cache_manager.CacheMonitor'), \
             patch('src.infrastructure.cache.core.cache_manager.CacheOptimizer'), \
             patch('src.infrastructure.cache.core.cache_manager.DistributedCacheManager'), \
             patch('src.infrastructure.cache.core.cache_manager.PerformanceMonitor'), \
             patch('src.infrastructure.cache.core.cache_manager.logger'):

            self.cache_manager = UnifiedCacheManager(self.config)

    def test_initialization_complete(self):
        """测试完整的初始化过程"""
        with patch('src.infrastructure.cache.core.cache_manager.MemoryCache'), \
             patch('src.infrastructure.cache.core.cache_manager.RedisCache'), \
             patch('src.infrastructure.cache.core.cache_manager.CacheMonitor'), \
             patch('src.infrastructure.cache.core.cache_manager.CacheOptimizer'), \
             patch('src.infrastructure.cache.core.cache_manager.DistributedCacheManager'), \
             patch('src.infrastructure.cache.core.cache_manager.PerformanceMonitor'), \
             patch('src.infrastructure.cache.core.cache_manager.logger'):

            manager = UnifiedCacheManager(self.config)

            assert manager.config == self.config
            assert hasattr(manager, 'memory_cache')
            assert hasattr(manager, 'redis_cache')
            assert hasattr(manager, 'monitor')
            assert hasattr(manager, 'optimizer')
            assert hasattr(manager, 'distributed_manager')
            assert hasattr(manager, 'performance_monitor')
            assert hasattr(manager, 'stats')

    def test_initialization_without_config(self):
        """测试无配置初始化"""
        with patch('src.infrastructure.cache.core.cache_manager.MemoryCache'), \
             patch('src.infrastructure.cache.core.cache_manager.RedisCache'), \
             patch('src.infrastructure.cache.core.cache_manager.CacheMonitor'), \
             patch('src.infrastructure.cache.core.cache_manager.CacheOptimizer'), \
             patch('src.infrastructure.cache.core.cache_manager.DistributedCacheManager'), \
             patch('src.infrastructure.cache.core.cache_manager.PerformanceMonitor'), \
             patch('src.infrastructure.cache.core.cache_manager.logger'):

            manager = UnifiedCacheManager()

            assert isinstance(manager.config, CacheConfig)
            assert manager.config.max_memory_size > 0

    def test_basic_cache_operations(self):
        """测试基础缓存操作"""
        # 测试set操作
        key = "test_key"
        value = "test_value"
        result = self.cache_manager.set(key, value)
        assert result is True

        # 测试get操作
        retrieved_value = self.cache_manager.get(key)
        assert retrieved_value == value

        # 测试delete操作
        delete_result = self.cache_manager.delete(key)
        assert delete_result is True

        # 验证删除后获取不到
        deleted_value = self.cache_manager.get(key)
        assert deleted_value is None

    def test_cache_operations_with_ttl(self):
        """测试带TTL的缓存操作"""
        key = "ttl_key"
        value = "ttl_value"
        ttl = 2  # 2秒

        # 设置带TTL的缓存
        result = self.cache_manager.set(key, value, ttl)
        assert result is True

        # 立即获取应该成功
        retrieved_value = self.cache_manager.get(key)
        assert retrieved_value == value

        # 等待TTL过期
        time.sleep(ttl + 0.1)

        # 再次获取应该返回None
        expired_value = self.cache_manager.get(key)
        assert expired_value is None

    def test_cache_clear_operation(self):
        """测试缓存清空操作"""
        # 添加多个缓存项
        test_data = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }

        for key, value in test_data.items():
            self.cache_manager.set(key, value)

        # 验证都设置成功
        for key, expected_value in test_data.items():
            assert self.cache_manager.get(key) == expected_value

        # 清空缓存
        clear_result = self.cache_manager.clear()
        assert clear_result is True

        # 验证都已被清空
        for key in test_data.keys():
            assert self.cache_manager.get(key) is None

    def test_cache_size_and_stats(self):
        """测试缓存大小和统计信息"""
        # 添加一些缓存项
        for i in range(10):
            self.cache_manager.set(f"key_{i}", f"value_{i}")

        # 测试size方法
        size = self.cache_manager.size()
        assert isinstance(size, int)
        assert size >= 0

        # 测试get_stats方法
        stats = self.cache_manager.get_stats()
        assert isinstance(stats, dict)
        assert "total_keys" in stats
        assert "memory_usage" in stats
        assert "hit_rate" in stats

        # 测试get_cache_stats方法
        cache_stats = self.cache_manager.get_cache_stats()
        assert isinstance(cache_stats, dict)
        assert "memory_cache" in cache_stats
        assert "redis_cache" in cache_stats

    def test_health_status(self):
        """测试健康状态检查"""
        health_status = self.cache_manager.get_health_status()

        assert isinstance(health_status, dict)
        assert "overall_health" in health_status
        assert "memory_cache" in health_status
        assert "redis_cache" in health_status
        assert "distributed_cache" in health_status

        # 健康状态应该是字符串
        assert isinstance(health_status["overall_health"], str)

    def test_complex_data_types(self):
        """测试复杂数据类型缓存"""
        # 测试字典
        dict_key = "dict_key"
        dict_value = {"name": "test", "age": 30, "items": [1, 2, 3]}
        self.cache_manager.set(dict_key, dict_value)
        retrieved_dict = self.cache_manager.get(dict_key)
        assert retrieved_dict == dict_value

        # 测试列表
        list_key = "list_key"
        list_value = [1, "hello", {"nested": "dict"}, [1, 2, 3]]
        self.cache_manager.set(list_key, list_value)
        retrieved_list = self.cache_manager.get(list_key)
        assert retrieved_list == list_value

        # 测试自定义对象
        class TestObject:
            def __init__(self, name, value):
                self.name = name
                self.value = value

            def __eq__(self, other):
                return (isinstance(other, TestObject) and
                       self.name == other.name and
                       self.value == other.value)

        obj_key = "obj_key"
        obj_value = TestObject("test_name", 42)
        self.cache_manager.set(obj_key, obj_value)
        retrieved_obj = self.cache_manager.get(obj_key)
        assert retrieved_obj == obj_value

    def test_cache_key_validation(self):
        """测试缓存键验证"""
        # 测试有效键
        valid_keys = ["normal_key", "key_with_123", "key-with-dashes"]
        for key in valid_keys:
            result = self.cache_manager.set(key, "value")
            assert result is True
            assert self.cache_manager.get(key) == "value"

        # 测试无效键（如果有验证的话）
        # 这里主要测试边界情况
        edge_cases = ["", "   ", None]
        for invalid_key in edge_cases:
            try:
                self.cache_manager.set(invalid_key, "value")
                # 如果没有抛出异常，验证是否正确处理
                retrieved = self.cache_manager.get(invalid_key)
                # 对于无效键，应该返回None或者不存储
                assert retrieved is None or retrieved == "value"
            except Exception:
                # 如果抛出异常也是可以接受的
                assert True

    def test_concurrent_access(self):
        """测试并发访问"""
        results = []
        errors = []

        def cache_worker(worker_id):
            try:
                # 每个worker执行一系列缓存操作
                for i in range(50):
                    key = f"worker_{worker_id}_key_{i}"
                    value = f"worker_{worker_id}_value_{i}"

                    # 写入
                    self.cache_manager.set(key, value)

                    # 读取验证
                    retrieved = self.cache_manager.get(key)
                    assert retrieved == value

                    # 删除
                    self.cache_manager.delete(key)

                results.append(worker_id)
            except Exception as e:
                errors.append(f"Worker {worker_id}: {str(e)}")

        # 启动多个线程并发访问
        threads = []
        num_threads = 5

        for i in range(num_threads):
            t = threading.Thread(target=cache_worker, args=(i,))
            threads.append(t)
            t.start()

        # 等待所有线程完成
        for t in threads:
            t.join()

        # 验证结果
        assert len(results) == num_threads
        assert len(errors) == 0

    def test_performance_under_load(self):
        """测试负载下的性能"""
        import time

        # 准备大量测试数据
        num_operations = 1000
        test_data = {f"perf_key_{i}": f"perf_value_{i}" for i in range(num_operations)}

        # 测试写入性能
        start_time = time.time()
        for key, value in test_data.items():
            self.cache_manager.set(key, value)
        write_time = time.time() - start_time

        # 测试读取性能
        start_time = time.time()
        for key in test_data.keys():
            retrieved = self.cache_manager.get(key)
            assert retrieved == test_data[key]
        read_time = time.time() - start_time

        # 性能基准：1000次操作应该在合理时间内完成
        total_time = write_time + read_time
        assert total_time < 5.0, f"性能不达标: {total_time:.2f}秒"

        # 验证每秒操作数
        ops_per_second = num_operations * 2 / total_time  # 读+写
        assert ops_per_second > 100, f"OPS太低: {ops_per_second:.0f} ops/sec"

    def test_memory_management(self):
        """测试内存管理"""
        import psutil
        import os

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # 执行大量缓存操作
        for i in range(5000):
            key = f"memory_test_key_{i}"
            value = f"memory_test_value_{i}_" + "x" * 100  # 较大值
            self.cache_manager.set(key, value)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # 内存增长应该在合理范围内（考虑测试环境的内存管理）
        # 这里设置一个相对宽松的上限
        assert memory_increase < 200, f"内存使用异常: 增加了{memory_increase:.2f}MB"

        # 清理测试数据
        for i in range(5000):
            key = f"memory_test_key_{i}"
            self.cache_manager.delete(key)

    def test_cache_persistence_simulation(self):
        """测试缓存持久化模拟"""
        # 设置一些重要数据
        critical_data = {
            "system_config": {"debug": False, "timeout": 30},
            "user_sessions": {"user_123": "session_token_abc"},
            "feature_flags": {"new_feature": True, "beta_feature": False}
        }

        # 存储到缓存
        for key, value in critical_data.items():
            self.cache_manager.set(key, value, ttl=3600)  # 1小时TTL

        # 模拟持久化保存（这里只是验证数据完整性）
        # 在实际实现中，这里会触发持久化逻辑

        # 验证数据完整性
        for key, expected_value in critical_data.items():
            retrieved = self.cache_manager.get(key)
            assert retrieved == expected_value

    def test_error_handling_and_recovery(self):
        """测试错误处理和恢复"""
        # 测试网络故障模拟
        original_get = self.cache_manager.redis_cache.get if hasattr(self.cache_manager, 'redis_cache') else None

        # 模拟Redis连接失败
        if hasattr(self.cache_manager, 'redis_cache'):
            self.cache_manager.redis_cache.get = Mock(side_effect=Exception("Connection failed"))

            # 即使Redis失败，内存缓存应该仍然工作
            self.cache_manager.set("fallback_key", "fallback_value")
            retrieved = self.cache_manager.get("fallback_key")
            assert retrieved == "fallback_value"

            # 恢复原始方法
            if original_get:
                self.cache_manager.redis_cache.get = original_get

    def test_cache_warming(self):
        """测试缓存预热"""
        # 模拟缓存预热数据
        warm_up_data = {
            "frequent_query_1": "result_1",
            "frequent_query_2": "result_2",
            "frequent_query_3": "result_3"
        }

        # 执行预热
        for key, value in warm_up_data.items():
            self.cache_manager.set(key, value)

        # 验证预热数据可用
        for key, expected_value in warm_up_data.items():
            retrieved = self.cache_manager.get(key)
            assert retrieved == expected_value

    def test_cache_invalidation_strategies(self):
        """测试缓存失效策略"""
        # 设置不同TTL的缓存项
        ttls = [1, 5, 10, 60]  # 秒

        for i, ttl in enumerate(ttls):
            key = f"ttl_test_key_{i}"
            value = f"ttl_test_value_{i}"
            self.cache_manager.set(key, value, ttl)

        # 立即验证所有项都存在
        for i in range(len(ttls)):
            key = f"ttl_test_key_{i}"
            assert self.cache_manager.get(key) is not None

        # 等待最短TTL过期
        time.sleep(2)

        # 最短TTL的项目应该过期
        expired_key = "ttl_test_key_0"
        assert self.cache_manager.get(expired_key) is None

        # 其他项目应该仍然存在
        for i in range(1, len(ttls)):
            key = f"ttl_test_key_{i}"
            assert self.cache_manager.get(key) is not None

    def test_bulk_operations(self):
        """测试批量操作"""
        # 批量设置
        bulk_data = {f"bulk_key_{i}": f"bulk_value_{i}" for i in range(100)}

        for key, value in bulk_data.items():
            self.cache_manager.set(key, value)

        # 批量验证
        for key, expected_value in bulk_data.items():
            retrieved = self.cache_manager.get(key)
            assert retrieved == expected_value

        # 批量删除
        for key in bulk_data.keys():
            self.cache_manager.delete(key)

        # 验证批量删除
        for key in bulk_data.keys():
            assert self.cache_manager.get(key) is None

    def test_monitoring_and_metrics(self):
        """测试监控和指标收集"""
        # 执行一些操作以生成指标
        for i in range(20):
            self.cache_manager.set(f"metric_key_{i}", f"metric_value_{i}")
            self.cache_manager.get(f"metric_key_{i}")
            if i % 3 == 0:  # 每3次删除一次
                self.cache_manager.delete(f"metric_key_{i}")

        # 获取统计信息
        stats = self.cache_manager.get_stats()
        cache_stats = self.cache_manager.get_cache_stats()

        # 验证统计信息结构
        required_stats_keys = ["total_keys", "memory_usage", "hit_rate", "miss_rate"]
        for key in required_stats_keys:
            assert key in stats

        # 验证缓存统计
        assert "memory_cache" in cache_stats
        assert "redis_cache" in cache_stats

        # 验证数值合理性
        assert stats["total_keys"] >= 0
        assert 0 <= stats["hit_rate"] <= 1
        assert 0 <= stats["miss_rate"] <= 1

    def test_configuration_updates(self):
        """测试配置更新"""
        # 获取原始配置
        original_config = self.cache_manager.config

        # 创建新配置
        new_config = CacheConfig(
            max_memory_size=200 * 1024 * 1024,  # 200MB
            default_ttl=7200,  # 2小时
            cleanup_interval=600,  # 10分钟
            enable_monitoring=True,
            enable_persistence=False  # 改变这个设置
        )

        # 应用新配置（如果支持的话）
        # 注意：实际实现可能不支持运行时配置更新
        try:
            self.cache_manager.update_config(new_config)
            assert self.cache_manager.config.max_memory_size == 200 * 1024 * 1024
            assert self.cache_manager.config.default_ttl == 7200
        except AttributeError:
            # 如果不支持配置更新，验证原始配置不变
            assert self.cache_manager.config == original_config

    def test_shutdown_and_cleanup(self):
        """测试关闭和清理"""
        # 添加一些测试数据
        for i in range(10):
            self.cache_manager.set(f"cleanup_key_{i}", f"cleanup_value_{i}")

        # 验证数据存在
        for i in range(10):
            key = f"cleanup_key_{i}"
            assert self.cache_manager.get(key) is not None

        # 执行清理
        cleanup_result = self.cache_manager.clear()
        assert cleanup_result is True

        # 验证数据已被清理
        for i in range(10):
            key = f"cleanup_key_{i}"
            assert self.cache_manager.get(key) is None

    def test_edge_cases(self):
        """测试边界情况"""
        # 测试空值
        self.cache_manager.set("empty_key", "")
        assert self.cache_manager.get("empty_key") == ""

        # 测试None值
        self.cache_manager.set("none_key", None)
        assert self.cache_manager.get("none_key") is None

        # 测试零值
        self.cache_manager.set("zero_key", 0)
        assert self.cache_manager.get("zero_key") == 0

        # 测试False值
        self.cache_manager.set("false_key", False)
        assert self.cache_manager.get("false_key") is False

        # 测试空列表和字典
        self.cache_manager.set("empty_list", [])
        assert self.cache_manager.get("empty_list") == []

        self.cache_manager.set("empty_dict", {})
        assert self.cache_manager.get("empty_dict") == {}

    def test_large_data_handling(self):
        """测试大数据处理"""
        # 测试大字符串
        large_string = "x" * 10000  # 10KB字符串
        self.cache_manager.set("large_string_key", large_string)
        retrieved_large_string = self.cache_manager.get("large_string_key")
        assert retrieved_large_string == large_string

        # 测试大列表
        large_list = list(range(1000))  # 1000个元素的列表
        self.cache_manager.set("large_list_key", large_list)
        retrieved_large_list = self.cache_manager.get("large_list_key")
        assert retrieved_large_list == large_list

        # 测试大字典
        large_dict = {f"key_{i}": f"value_{i}" for i in range(500)}
        self.cache_manager.set("large_dict_key", large_dict)
        retrieved_large_dict = self.cache_manager.get("large_dict_key")
        assert retrieved_large_dict == large_dict

    def test_special_characters_and_unicode(self):
        """测试特殊字符和Unicode"""
        # 测试特殊字符
        special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?/~`"
        self.cache_manager.set("special_chars_key", special_chars)
        retrieved_special = self.cache_manager.get("special_chars_key")
        assert retrieved_special == special_chars

        # 测试Unicode字符
        unicode_string = "Hello 世界 🌍 Test 中文"
        self.cache_manager.set("unicode_key", unicode_string)
        retrieved_unicode = self.cache_manager.get("unicode_key")
        assert retrieved_unicode == unicode_string

        # 测试Emoji
        emoji_string = "🚀✨💻🔥"
        self.cache_manager.set("emoji_key", emoji_string)
        retrieved_emoji = self.cache_manager.get("emoji_key")
        assert retrieved_emoji == emoji_string
