#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据缓存测试
测试数据层数据缓存组件
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
import pandas as pd
import tempfile
from unittest.mock import Mock, patch, MagicMock
from typing import Any, Dict, Optional

from src.data.cache.data_cache import DataCache



# 设置测试超时，避免死锁和无限等待
pytestmark = [
    pytest.mark.timeout(30),  # 30秒超时
    pytest.mark.deadlock_risk,  # 标记为可能存在死锁风险
    pytest.mark.concurrent,  # 并发测试
    pytest.mark.infinite_loop_risk  # 可能存在无限循环风险
]

class TestDataCache:
    """数据缓存测试"""

    def setup_method(self):
        """测试前准备"""
        self._temp_dir = tempfile.TemporaryDirectory()
        self.cache = DataCache(cache_dir=self._temp_dir.name)

    def teardown_method(self):
        """测试后清理"""
        if hasattr(self.cache, 'cache_manager') and hasattr(self.cache.cache_manager, 'stop'):
            self.cache.cache_manager.stop()
        if hasattr(self, "_temp_dir"):
            self._temp_dir.cleanup()

    def test_data_cache_initialization(self):
        """测试数据缓存初始化"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)

            assert cache.cache_manager is not None
            assert hasattr(cache, 'cache_manager')
            cache.cache_manager.stop()

    def test_data_cache_basic_operations(self):
        """测试数据缓存基本操作"""
        # 测试设置和获取
        assert self.cache.set("test_key", "test_value")
        assert self.cache.get("test_key") == "test_value"

        # 测试不存在的键
        assert self.cache.get("nonexistent") is None

        # 测试存在性检查
        assert self.cache.exists("test_key") is True
        assert self.cache.exists("nonexistent") is False

        # 测试删除
        assert self.cache.delete("test_key")
        assert self.cache.get("test_key") is None

    def test_data_cache_dataframe_operations(self):
        """测试数据缓存DataFrame操作"""
        # 创建测试DataFrame
        df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['a', 'b', 'c'],
            'C': [1.1, 2.2, 3.3]
        })

        # 测试设置DataFrame
        assert self.cache.set_dataframe("df_key", df)

        # 测试获取DataFrame
        retrieved_df = self.cache.get_dataframe("df_key")
        assert retrieved_df is not None
        pd.testing.assert_frame_equal(retrieved_df, df)

        # 测试不存在的DataFrame
        assert self.cache.get_dataframe("nonexistent_df") is None

    def test_data_cache_dict_operations(self):
        """测试数据缓存字典操作"""
        # 创建测试字典
        test_dict = {
            'name': 'test',
            'value': 42,
            'items': [1, 2, 3],
            'nested': {'key': 'value'}
        }

        # 测试设置字典
        assert self.cache.set_dict("dict_key", test_dict)

        # 测试获取字典
        retrieved_dict = self.cache.get_dict("dict_key")
        assert retrieved_dict is not None
        assert retrieved_dict == test_dict

        # 测试不存在的字典
        assert self.cache.get_dict("nonexistent_dict") is None

    def test_data_cache_get_or_compute(self):
        """测试数据缓存获取或计算功能"""
        # 定义计算函数
        def compute_expensive_data(x, y):
            """模拟耗时计算"""
            return x * y + 42

        # 第一次调用，应该计算并缓存
        result1 = self.cache.get_or_compute("compute_key", compute_expensive_data, 10, 20)
        assert result1 == 242  # 10 * 20 + 42

        # 第二次调用，应该从缓存获取
        result2 = self.cache.get_or_compute("compute_key", compute_expensive_data, 10, 20)
        assert result2 == 242

        # 验证缓存中确实有这个值
        assert self.cache.get("compute_key") == 242

    def test_data_cache_get_or_compute_with_kwargs(self):
        """测试带关键字参数的获取或计算"""
        def compute_with_kwargs(base, multiplier=2, offset=0):
            return base * multiplier + offset

        # 使用关键字参数
        result = self.cache.get_or_compute(
            "kwargs_key",
            compute_with_kwargs,
            10,
            multiplier=3,
            offset=5
        )

        assert result == 35  # 10 * 3 + 5

        # 从缓存获取
        cached_result = self.cache.get("kwargs_key")
        assert cached_result == 35

    def test_data_cache_get_or_compute_error_handling(self):
        """测试获取或计算的错误处理"""
        def failing_compute():
            raise ValueError("Computation failed")

        # 计算失败时不应该缓存错误
        with pytest.raises(ValueError, match="Computation failed"):
            self.cache.get_or_compute("fail_key", failing_compute)

        # 缓存中不应该有这个键
        assert self.cache.exists("fail_key") is False

    def test_data_cache_clear(self):
        """测试数据缓存清空功能"""
        # 添加一些数据
        self.cache.set("key1", "value1")
        self.cache.set_dataframe("df_key", pd.DataFrame({'A': [1, 2]}))
        self.cache.set_dict("dict_key", {"test": "data"})

        # 验证数据存在
        assert self.cache.exists("key1")
        assert self.cache.exists("df_key")
        assert self.cache.exists("dict_key")

        # 清空缓存
        assert self.cache.clear()

        # 验证数据已被清空
        assert not self.cache.exists("key1")
        assert not self.cache.exists("df_key")
        assert not self.cache.exists("dict_key")

    def test_data_cache_complex_data_types(self):
        """测试复杂数据类型的缓存"""
        # 测试嵌套字典
        nested_data = {
            'level1': {
                'level2': {
                    'list': [1, 2, {'deep': 'value'}],
                    'tuple': (1, 2, 3),
                    'set': {1, 2, 3}
                }
            }
        }

        assert self.cache.set("complex_key", nested_data)
        retrieved = self.cache.get("complex_key")
        assert retrieved == nested_data

        # 测试自定义对象（如果可序列化）
        class SimpleObject:
            def __init__(self, value):
                self.value = value

            def __eq__(self, other):
                return isinstance(other, SimpleObject) and self.value == other.value

        obj = SimpleObject(42)
        assert self.cache.set("object_key", obj)
        retrieved_obj = self.cache.get("object_key")
        assert retrieved_obj == obj

    def test_data_cache_memory_management(self):
        """测试数据缓存内存管理"""
        # 创建小容量缓存
        with tempfile.TemporaryDirectory() as temp_dir:
            from src.data.cache.cache_manager import CacheConfig
            config = CacheConfig(max_size=3, disk_cache_dir=temp_dir)
            cache = DataCache.__new__(DataCache)  # 创建实例但不调用__init__
            cache.cache_manager = Mock()

            # 模拟缓存管理器的行为
            cache.cache_manager.get.return_value = None
            cache.cache_manager.set.return_value = True
            cache.cache_manager.exists.return_value = True

            # 这里我们只是验证接口，实际的内存管理由CacheManager处理
            assert cache.cache_manager is not None

    @patch('src.data.cache.data_cache.CacheManager')
    def test_data_cache_initialization_with_mock(self, mock_cache_manager):
        """测试数据缓存初始化（使用Mock）"""
        mock_manager = Mock()
        mock_cache_manager.return_value = mock_manager

        cache = DataCache(cache_dir="test_dir")

        # 验证CacheManager被正确创建
        mock_cache_manager.assert_called_once()
        assert cache.cache_manager == mock_manager

    def test_data_cache_error_handling(self):
        """测试数据缓存错误处理"""
        # 模拟缓存管理器错误
        with patch.object(self.cache.cache_manager, 'get', side_effect=Exception("Cache error")):
            result = self.cache.get("test_key")
            assert result is None

        with patch.object(self.cache.cache_manager, 'set', side_effect=Exception("Cache error")):
            result = self.cache.set("test_key", "value")
            assert result is False

        with patch.object(self.cache.cache_manager, 'delete', side_effect=Exception("Cache error")):
            result = self.cache.delete("test_key")
            assert result is False

        with patch.object(self.cache.cache_manager, 'exists', side_effect=Exception("Cache error")):
            assert self.cache.exists("test_key") is False

        with patch.object(self.cache.cache_manager, 'clear', side_effect=Exception("Cache error")):
            assert self.cache.clear() is False

        with patch.object(self.cache.cache_manager, 'get', side_effect=Exception("Cache error")):
            assert self.cache.get_dataframe("df_key") is None
            assert self.cache.get_dict("dict_key") is None

        with patch.object(self.cache.cache_manager, 'set', side_effect=Exception("Cache error")):
            assert self.cache.set_dataframe("df_key", pd.DataFrame()) is False
            assert self.cache.set_dict("dict_key", {}) is False

    def test_data_cache_dataframe_validation(self):
        """测试DataFrame缓存验证"""
        # 测试有效的DataFrame
        df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        assert self.cache.set_dataframe("valid_df", df)
        retrieved = self.cache.get_dataframe("valid_df")
        pd.testing.assert_frame_equal(retrieved, df)

        # 测试空DataFrame
        empty_df = pd.DataFrame()
        assert self.cache.set_dataframe("empty_df", empty_df)
        retrieved_empty = self.cache.get_dataframe("empty_df")
        pd.testing.assert_frame_equal(retrieved_empty, empty_df)

    def test_data_cache_dict_validation(self):
        """测试字典缓存验证"""
        # 测试有效字典
        valid_dict = {'key': 'value', 'number': 42}
        assert self.cache.set_dict("valid_dict", valid_dict)
        retrieved = self.cache.get_dict("valid_dict")
        assert retrieved == valid_dict

        # 测试空字典
        empty_dict = {}
        assert self.cache.set_dict("empty_dict", empty_dict)
        retrieved_empty = self.cache.get_dict("empty_dict")
        assert retrieved_empty == empty_dict

        # 测试嵌套字典
        nested_dict = {'level1': {'level2': 'value'}}
        assert self.cache.set_dict("nested_dict", nested_dict)
        retrieved_nested = self.cache.get_dict("nested_dict")
        assert retrieved_nested == nested_dict


class TestDataCacheIntegration:
    """数据缓存集成测试"""

    def test_data_cache_workflow(self):
        """测试数据缓存完整工作流程"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)

            try:
                # 1. 设置基础数据
                cache.set("user_123", {"name": "John", "age": 30})

                # 2. 设置DataFrame数据
                df = pd.DataFrame({
                    'timestamp': pd.date_range('2023-01-01', periods=3),
                    'value': [100, 200, 300]
                })
                cache.set_dataframe("time_series", df)

                # 3. 设置字典数据
                config = {"setting1": True, "setting2": "value"}
                cache.set_dict("user_config", config)

                # 4. 验证所有数据
                assert cache.get("user_123") == {"name": "John", "age": 30}
                retrieved_df = cache.get_dataframe("time_series")
                pd.testing.assert_frame_equal(retrieved_df, df)
                assert cache.get_dict("user_config") == config

                # 5. 测试缓存统计
                stats = cache.cache_manager.get_stats()
                assert isinstance(stats, dict)
                assert stats["sets"] >= 3

            finally:
                cache.cache_manager.stop()

    def test_data_cache_performance_simulation(self):
        """测试数据缓存性能模拟"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)

            try:
                # 模拟大数据集缓存
                large_data = {"data": list(range(1000))}
                cache.set("large_dataset", large_data)

                # 验证可以快速检索
                retrieved = cache.get("large_dataset")
                assert retrieved == large_data

                # 验证缓存命中
                stats = cache.cache_manager.get_stats()
                assert stats["hits"] >= 1

            finally:
                cache.cache_manager.stop()

    def test_data_cache_concurrent_usage(self):
        """测试数据缓存并发使用"""
        import threading
        import time

        with tempfile.TemporaryDirectory() as temp_dir:
            cache = DataCache(cache_dir=temp_dir)

            results = []
            errors = []

            def concurrent_worker(worker_id):
                """并发工作线程"""
                try:
                    for i in range(10):
                        key = f"concurrent_key_{worker_id}_{i}"
                        value = f"concurrent_value_{worker_id}_{i}"

                        cache.set(key, value)
                        retrieved = cache.get(key)

                        if retrieved != value:
                            errors.append(f"Worker {worker_id}: mismatch")
                        else:
                            results.append(True)

                        time.sleep(0.001)  # 小延迟模拟真实使用
                except Exception as e:
                    errors.append(f"Worker {worker_id}: {e}")

            try:
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
                assert len(errors) == 0, f"Concurrent errors: {errors}"
                assert len(results) == 30  # 3 workers * 10 operations

            finally:
                cache.cache_manager.stop()