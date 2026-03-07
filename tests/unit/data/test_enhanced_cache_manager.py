#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强缓存管理器测试
测试数据层增强缓存管理器组件
"""

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
import time
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Optional, Dict

from src.data.cache.enhanced_cache_manager import EnhancedCacheManager



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class TestEnhancedCacheManager:
    """增强缓存管理器测试"""

    def setup_method(self):
        """测试前准备"""
        self.temp_dir = tempfile.mkdtemp()
        self.cache = EnhancedCacheManager(
            cache_dir=self.temp_dir,
            max_memory_size=10 * 1024 * 1024,  # 10MB
            max_disk_size=100 * 1024 * 1024    # 100MB
        )

    def teardown_method(self):
        """测试后清理"""
        # 清理临时文件
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    def test_enhanced_cache_manager_initialization(self):
        """测试增强缓存管理器初始化"""
        assert self.cache.disk_cache_dir == self.temp_dir
        assert self.cache.max_memory_size == 10 * 1024 * 1024
        assert self.cache.max_disk_size == 100 * 1024 * 1024
        assert isinstance(self.cache.memory_cache, dict)
        assert hasattr(self.cache, '_lock')
        assert os.path.exists(self.temp_dir)

    def test_enhanced_cache_manager_basic_operations(self):
        """测试增强缓存管理器基本操作"""
        # 测试设置和获取
        assert self.cache.set("test_key", "test_value")
        assert self.cache.get("test_key") == "test_value"

        # 测试不存在的键
        assert self.cache.get("nonexistent") is None

        # 测试过期时间
        self.cache.set("expire_key", "expire_value", expire=1)
        assert self.cache.get("expire_key") == "expire_value"

        # 等待过期
        time.sleep(1.1)
        assert self.cache.get("expire_key") is None

    def test_enhanced_cache_manager_prefix_operations(self):
        """测试增强缓存管理器前缀操作"""
        # 设置带前缀的缓存
        self.cache.set("user_123", "user_data", prefix="users")
        self.cache.set("config_app", "config_data", prefix="configs")

        # 获取带前缀的缓存
        assert self.cache.get("user_123", prefix="users") == "user_data"
        assert self.cache.get("config_app", prefix="configs") == "config_data"

        # 测试不同前缀的同名键
        assert self.cache.get("user_123", prefix="configs") is None

    def test_enhanced_cache_manager_memory_management(self):
        """测试增强缓存管理器内存管理"""
        # 设置小内存限制
        small_cache = EnhancedCacheManager(
            cache_dir=tempfile.mkdtemp(),
            max_memory_size=1000,  # 1KB
            max_disk_size=10 * 1024 * 1024
        )

        try:
            # 添加大数据
            large_data = "x" * 2000  # 2KB数据

            # 设置大数据（应该触发内存清理）
            small_cache.set("large_key", large_data)

            # 验证数据被正确处理
            assert small_cache.get("large_key") == large_data
        finally:
            # 清理临时目录
            if os.path.exists(small_cache.disk_cache_dir):
                import shutil
                shutil.rmtree(small_cache.disk_cache_dir)

    def test_enhanced_cache_manager_disk_cache(self):
        """测试增强缓存管理器磁盘缓存"""
        # 创建大数据
        large_data = {"data": list(range(1000))}

        # 设置到缓存
        self.cache.set("disk_test", large_data)

        # 清除内存缓存
        self.cache.memory_cache.clear()
        self.cache.memory_size = 0

        # 从磁盘获取
        retrieved = self.cache.get("disk_test")
        assert retrieved == large_data

    def test_enhanced_cache_manager_cache_promotion(self):
        """测试增强缓存管理器缓存提升"""
        # 设置数据到磁盘缓存
        disk_data = "disk_only_data"
        self.cache.set("promote_test", disk_data)

        # 清除内存缓存
        self.cache.memory_cache.clear()
        self.cache.memory_size = 0

        # 第一次获取应该从磁盘加载并提升到内存
        result1 = self.cache.get("promote_test")
        assert result1 == disk_data

        # 验证数据现在在内存中
        cache_key = self.cache._generate_cache_key("promote_test", "")
        assert cache_key in self.cache.memory_cache

        # 第二次获取应该从内存获取
        result2 = self.cache.get("promote_test")
        assert result2 == disk_data

    def test_enhanced_cache_manager_data_types(self):
        """测试增强缓存管理器不同数据类型"""
        # 测试字典
        dict_data = {"key": "value", "number": 42}
        self.cache.set("dict_key", dict_data)
        assert self.cache.get("dict_key") == dict_data

        # 测试列表
        list_data = [1, 2, 3, {"nested": "data"}]
        self.cache.set("list_key", list_data)
        assert self.cache.get("list_key") == list_data

        # 测试DataFrame
        df_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        self.cache.set("df_key", df_data)
        retrieved_df = self.cache.get("df_key")
        pd.testing.assert_frame_equal(retrieved_df, df_data)

        # 测试numpy数组
        np_data = np.array([1, 2, 3, 4, 5])
        self.cache.set("np_key", np_data)
        retrieved_np = self.cache.get("np_key")
        np.testing.assert_array_equal(retrieved_np, np_data)

    def test_enhanced_cache_manager_clear_operations(self):
        """测试增强缓存管理器清空操作"""
        # 设置一些数据
        self.cache.set("key1", "value1")
        self.cache.set("key2", "value2", prefix="group1")
        self.cache.set("key3", "value3", prefix="group1")
        self.cache.set("key4", "value4", prefix="group2")

        # 验证数据存在
        assert self.cache.get("key1") == "value1"
        assert self.cache.get("key2", prefix="group1") == "value2"
        assert self.cache.get("key3", prefix="group1") == "value3"
        assert self.cache.get("key4", prefix="group2") == "value4"

        # 清空特定前缀
        self.cache.clear(prefix="group1")

        # 验证特定前缀被清空
        assert self.cache.get("key1") == "value1"  # 不带前缀的保留
        assert self.cache.get("key2", prefix="group1") is None
        assert self.cache.get("key3", prefix="group1") is None
        assert self.cache.get("key4", prefix="group2") == "value4"  # 其他前缀保留

        # 清空所有
        self.cache.clear()
        assert self.cache.get("key1") is None
        assert self.cache.get("key4", prefix="group2") is None

    def test_enhanced_cache_manager_statistics(self):
        """测试增强缓存管理器统计功能"""
        # 执行一些操作
        self.cache.set("stat_key1", "value1")
        self.cache.get("stat_key1")  # 命中
        self.cache.get("nonexistent")  # 未命中

        # 获取统计信息
        stats = self.cache.get_stats()
        assert isinstance(stats, dict)

        # 验证统计字段
        assert "total_operations" in stats
        assert "memory_hits" in stats
        assert "disk_hits" in stats
        assert "misses" in stats
        assert "hit_rate" in stats
        assert "memory_size" in stats
        assert "disk_size" in stats
        assert "memory_cache_count" in stats

        # 验证统计值
        assert stats["total_operations"] >= 3
        assert stats["memory_hits"] >= 1
        assert stats["misses"] >= 1
        assert 0.0 <= stats["hit_rate"] <= 1.0

    def test_enhanced_cache_manager_concurrent_access(self):
        """测试增强缓存管理器并发访问"""
        import threading

        results = []
        errors = []

        def concurrent_worker(worker_id):
            """并发工作线程"""
            try:
                for i in range(20):
                    key = f"concurrent_key_{worker_id}_{i}"
                    value = f"concurrent_value_{worker_id}_{i}"

                    self.cache.set(key, value, expire=60)
                    retrieved = self.cache.get(key)

                    if retrieved != value:
                        errors.append(f"Worker {worker_id}: expected {value}, got {retrieved}")
                    else:
                        results.append(True)

                    time.sleep(0.001)  # 小延迟
            except Exception as e:
                errors.append(f"Worker {worker_id}: {e}")

        # 创建多个线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=concurrent_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证结果
        assert len(errors) == 0, f"Concurrent access errors: {errors}"
        assert len(results) == 60  # 3 workers * 20 operations

    def test_enhanced_cache_manager_cache_key_generation(self):
        """测试增强缓存管理器缓存键生成"""
        # 测试基本键生成
        key1 = self.cache._generate_cache_key("test_key", "")
        assert isinstance(key1, str)
        assert len(key1) > 0

        # 测试带前缀的键生成
        key2 = self.cache._generate_cache_key("test_key", "prefix")
        key3 = self.cache._generate_cache_key("test_key", "prefix")
        assert key2 == key3  # 相同输入应该生成相同键

        # 测试不同输入生成不同键
        key4 = self.cache._generate_cache_key("different_key", "prefix")
        assert key2 != key4

    def test_enhanced_cache_manager_memory_size_calculation(self):
        """测试增强缓存管理器内存大小计算"""
        # 测试简单对象
        size1 = self.cache._get_memory_size("test_string")
        assert size1 > 0

        # 测试字典
        dict_obj = {"key": "value", "number": 42}
        size2 = self.cache._get_memory_size(dict_obj)
        assert size2 > size1

        # 测试列表
        list_obj = [1, 2, 3, 4, 5]
        size3 = self.cache._get_memory_size(list_obj)
        assert size3 > 0

        # 测试复杂对象
        complex_obj = {
            "data": list(range(100)),
            "nested": {"deep": "value" * 10}
        }
        size4 = self.cache._get_memory_size(complex_obj)
        assert size4 > size2

    def test_enhanced_cache_manager_cleanup_operations(self):
        """测试增强缓存管理器清理操作"""
        # 设置一些过期数据
        self.cache.set("expired_key", "expired_value", expire=1)
        self.cache.set("valid_key", "valid_value", expire=60)

        # 等待过期
        time.sleep(1.1)

        # 手动触发清理
        self.cache._cleanup_memory_cache()

        # 验证过期数据被清理
        assert self.cache.get("expired_key") is None
        assert self.cache.get("valid_key") == "valid_value"

    def test_enhanced_cache_manager_disk_operations(self):
        """测试增强缓存管理器磁盘操作"""
        # 测试磁盘缓存大小计算
        initial_size = self.cache._get_disk_cache_size()
        assert isinstance(initial_size, int)
        assert initial_size >= 0

        # 设置数据到磁盘
        self.cache.set("disk_size_test", "x" * 1000)
        time.sleep(0.1)  # 等待写入

        # 验证磁盘大小增加
        new_size = self.cache._get_disk_cache_size()
        assert new_size >= initial_size

    def test_enhanced_cache_manager_error_handling(self):
        """测试增强缓存管理器错误处理"""
        # 测试无效缓存目录
        invalid_cache = EnhancedCacheManager(
            cache_dir="/invalid/path/that/does/not/exist",
            max_memory_size=1024,
            max_disk_size=1024
        )

        # 应该仍然能工作（使用内存缓存）
        invalid_cache.set("error_test", "error_value")
        result = invalid_cache.get("error_test")
        assert result == "error_value"

        # 清理
        if os.path.exists(invalid_cache.disk_cache_dir):
            import shutil
            shutil.rmtree(invalid_cache.disk_cache_dir)

    def test_enhanced_cache_manager_large_data_handling(self):
        """测试增强缓存管理器大数据处理"""
        # 创建大数据
        large_data = {
            "array": list(range(10000)),
            "nested": {
                "deep": "value" * 100,
                "list": [i * 2 for i in range(1000)]
            }
        }

        # 设置大数据
        self.cache.set("large_data", large_data, expire=300)

        # 验证可以正确存储和检索
        retrieved = self.cache.get("large_data")
        assert retrieved == large_data

        # 验证统计信息
        stats = self.cache.get_stats()
        assert stats["memory_size"] > 0


class TestEnhancedCacheManagerIntegration:
    """增强缓存管理器集成测试"""

    def test_enhanced_cache_manager_full_workflow(self):
        """测试增强缓存管理器完整工作流程"""
        temp_dir = tempfile.mkdtemp()
        cache = EnhancedCacheManager(
            cache_dir=temp_dir,
            max_memory_size=5 * 1024 * 1024,  # 5MB
            max_disk_size=50 * 1024 * 1024     # 50MB
        )

        try:
            # 1. 设置不同类型的数据
            cache.set("string_data", "Hello World")
            cache.set("dict_data", {"user": "test", "age": 25})
            cache.set("list_data", [1, 2, 3, 4, 5])

            # 2. 创建DataFrame数据
            df = pd.DataFrame({
                'timestamp': pd.date_range('2023-01-01', periods=100),
                'value': np.random.randn(100),
                'category': np.random.choice(['A', 'B', 'C'], 100)
            })
            cache.set("dataframe_data", df)

            # 3. 设置带前缀的数据
            cache.set("user_001", {"name": "Alice", "role": "admin"}, prefix="users")
            cache.set("user_002", {"name": "Bob", "role": "user"}, prefix="users")
            cache.set("config_db", {"host": "localhost", "port": 5432}, prefix="configs")

            # 4. 验证所有数据
            assert cache.get("string_data") == "Hello World"
            assert cache.get("dict_data") == {"user": "test", "age": 25}
            assert cache.get("list_data") == [1, 2, 3, 4, 5]

            retrieved_df = cache.get("dataframe_data")
            pd.testing.assert_frame_equal(retrieved_df, df)

            assert cache.get("user_001", prefix="users") == {"name": "Alice", "role": "admin"}
            assert cache.get("user_002", prefix="users") == {"name": "Bob", "role": "user"}
            assert cache.get("config_db", prefix="configs") == {"host": "localhost", "port": 5432}

            # 5. 测试缓存统计
            stats = cache.get_stats()
            assert stats["total_operations"] >= 9  # 6 sets + 3 gets
            assert stats["memory_cache_count"] > 0

            # 6. 测试清理操作
            cache.clear(prefix="users")
            assert cache.get("user_001", prefix="users") is None
            assert cache.get("user_002", prefix="users") is None
            assert cache.get("config_db", prefix="configs") is not None  # 其他前缀保留

        finally:
            # 清理临时目录
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)

    def test_enhanced_cache_manager_performance_simulation(self):
        """测试增强缓存管理器性能模拟"""
        temp_dir = tempfile.mkdtemp()
        cache = EnhancedCacheManager(
            cache_dir=temp_dir,
            max_memory_size=20 * 1024 * 1024,  # 20MB
            max_disk_size=200 * 1024 * 1024     # 200MB
        )

        try:
            # 模拟高频缓存操作
            start_time = time.time()

            for i in range(100):
                key = f"perf_key_{i}"
                data = {"index": i, "data": "x" * 100}  # 100字节数据
                cache.set(key, data)

                # 立即读取验证
                retrieved = cache.get(key)
                assert retrieved == data

            end_time = time.time()
            duration = end_time - start_time

            # 验证性能在合理范围内（每操作不应超过100ms）
            assert duration < 10.0, f"Performance test took {duration:.2f}s"

            # 验证统计信息
            stats = cache.get_stats()
            assert stats["total_operations"] >= 200  # 100 sets + 100 gets
            assert stats["hit_rate"] > 0.8  # 命中率应该很高

        finally:
            if os.path.exists(temp_dir):
                import shutil
                shutil.rmtree(temp_dir)
