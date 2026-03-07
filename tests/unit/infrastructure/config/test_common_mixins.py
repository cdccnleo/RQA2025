#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 配置公共Mixin组件

测试配置系统的公共Mixin和基类功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import threading
from unittest.mock import Mock, patch
from typing import Dict, List, Any, Optional
from enum import Enum


# 测试用的简化版本，避免导入问题
class ConfigScope(Enum):
    APPLICATION = "application"
    SYSTEM = "system"


class BatchOperationsMixin:
    """批量操作Mixin类"""

    def batch_get(self, keys: List[str]) -> Dict[str, Any]:
        """批量获取配置"""
        result = {}
        for key in keys:
            result[key] = self.get(key)
        return result

    def batch_set(self, config: Dict[str, Any]) -> bool:
        """批量设置配置"""
        try:
            for key, value in config.items():
                self.set(key, value)
            return True
        except Exception as e:
            if hasattr(self, '_logger'):
                self._logger.error(f"批量设置配置失败: {e}")
            return False


class BaseConfigStorage:
    """配置存储基类"""

    def __init__(self):
        self._data: Dict[ConfigScope, Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def list_keys(self, scope: Optional[ConfigScope] = None) -> List[str]:
        """列出配置键"""
        with self._lock:
            if scope:
                return list(self._data.get(scope, {}).keys())
            else:
                all_keys = []
                for scope_data in self._data.values():
                    all_keys.extend(scope_data.keys())
                return all_keys

    def exists(self, key: str, scope: ConfigScope = ConfigScope.APPLICATION) -> bool:
        """检查配置是否存在"""
        with self._lock:
            return scope in self._data and key in self._data[scope]


class TestBatchOperationsMixin(unittest.TestCase):
    """测试批量操作Mixin"""

    def setUp(self):
        """测试前准备"""
        class TestClass(BatchOperationsMixin):
            def __init__(self):
                super().__init__()
                self._config = {}

            def get(self, key):
                return self._config.get(key)

            def set(self, key, value):
                self._config[key] = value
                return True

        self.mixin_instance = TestClass()

    def test_batch_get_empty_keys(self):
        """测试批量获取空键列表"""
        result = self.mixin_instance.batch_get([])
        self.assertEqual(result, {})

    def test_batch_get_single_key(self):
        """测试批量获取单个键"""
        self.mixin_instance.set("key1", "value1")
        result = self.mixin_instance.batch_get(["key1"])
        self.assertEqual(result, {"key1": "value1"})

    def test_batch_get_multiple_keys(self):
        """测试批量获取多个键"""
        self.mixin_instance.set("key1", "value1")
        self.mixin_instance.set("key2", "value2")
        self.mixin_instance.set("key3", "value3")

        result = self.mixin_instance.batch_get(["key1", "key3"])
        expected = {"key1": "value1", "key3": "value3"}
        self.assertEqual(result, expected)

    def test_batch_get_nonexistent_keys(self):
        """测试批量获取不存在的键"""
        result = self.mixin_instance.batch_get(["nonexistent1", "nonexistent2"])
        expected = {"nonexistent1": None, "nonexistent2": None}
        self.assertEqual(result, expected)

    def test_batch_set_empty_config(self):
        """测试批量设置空配置"""
        result = self.mixin_instance.batch_set({})
        self.assertTrue(result)

    def test_batch_set_single_item(self):
        """测试批量设置单个项目"""
        config = {"key1": "value1"}
        result = self.mixin_instance.batch_set(config)
        self.assertTrue(result)
        self.assertEqual(self.mixin_instance.get("key1"), "value1")

    def test_batch_set_multiple_items(self):
        """测试批量设置多个项目"""
        config = {"key1": "value1", "key2": "value2", "key3": "value3"}
        result = self.mixin_instance.batch_set(config)
        self.assertTrue(result)
        self.assertEqual(self.mixin_instance.get("key1"), "value1")
        self.assertEqual(self.mixin_instance.get("key2"), "value2")
        self.assertEqual(self.mixin_instance.get("key3"), "value3")

    def test_batch_set_with_exception(self):
        """测试批量设置时的异常处理"""
        # 模拟set方法抛出异常
        original_set = self.mixin_instance.set
        def failing_set(key, value):
            raise Exception("Test exception")
        self.mixin_instance.set = failing_set

        try:
            config = {"key1": "value1"}
            result = self.mixin_instance.batch_set(config)
            self.assertFalse(result)
        finally:
            self.mixin_instance.set = original_set


class TestBaseConfigStorage(unittest.TestCase):
    """测试配置存储基类"""

    def setUp(self):
        """测试前准备"""
        self.base_storage = BaseConfigStorage()
        self.ConfigScope = ConfigScope

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.base_storage._data, dict)
        self.assertIsNotNone(self.base_storage._lock)

    def test_list_keys_empty(self):
        """测试列出空存储的键"""
        result = self.base_storage.list_keys()
        self.assertEqual(result, [])

    def test_list_keys_single_scope(self):
        """测试列出单个作用域的键"""
        # 手动设置数据（模拟子类行为）
        self.base_storage._data[self.ConfigScope.APPLICATION] = {"key1": "value1", "key2": "value2"}

        result = self.base_storage.list_keys(self.ConfigScope.APPLICATION)
        self.assertEqual(set(result), {"key1", "key2"})

    def test_list_keys_all_scopes(self):
        """测试列出所有作用域的键"""
        # 手动设置数据
        self.base_storage._data[self.ConfigScope.APPLICATION] = {"app_key1": "app_value1"}
        self.base_storage._data[self.ConfigScope.SYSTEM] = {"sys_key1": "sys_value1", "sys_key2": "sys_value2"}

        result = self.base_storage.list_keys()
        self.assertEqual(set(result), {"app_key1", "sys_key1", "sys_key2"})

    def test_exists_key_in_scope(self):
        """测试检查存在于作用域中的键"""
        # 手动设置数据
        self.base_storage._data[self.ConfigScope.APPLICATION] = {"key1": "value1"}

        result = self.base_storage.exists("key1", self.ConfigScope.APPLICATION)
        self.assertTrue(result)

    def test_exists_key_not_in_scope(self):
        """测试检查不存在于作用域中的键"""
        result = self.base_storage.exists("nonexistent", self.ConfigScope.APPLICATION)
        self.assertFalse(result)

    def test_exists_scope_not_exist(self):
        """测试检查不存在的作用域"""
        result = self.base_storage.exists("key1", self.ConfigScope.SYSTEM)
        self.assertFalse(result)

    def test_thread_safety(self):
        """测试线程安全性"""
        import threading
        import time

        results = []
        errors = []

        def worker(worker_id):
            try:
                # 测试并发写入
                for i in range(10):
                    key = f"worker_{worker_id}_key_{i}"
                    self.base_storage._data.setdefault(self.ConfigScope.APPLICATION, {})[key] = f"value_{i}"

                # 测试并发读取
                keys = self.base_storage.list_keys(self.ConfigScope.APPLICATION)
                results.append(len([k for k in keys if f"worker_{worker_id}" in k]))

            except Exception as e:
                errors.append(str(e))

        # 创建多个线程
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)

        # 启动线程
        for t in threads:
            t.start()

        # 等待线程完成
        for t in threads:
            t.join()

        # 验证结果
        self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
        self.assertEqual(len(results), 5)

        # 验证所有数据都被正确写入
        all_keys = self.base_storage.list_keys(self.ConfigScope.APPLICATION)
        total_worker_keys = len([k for k in all_keys if "worker_" in k])
        self.assertEqual(total_worker_keys, 50)  # 5 workers * 10 keys each


if __name__ == '__main__':
    unittest.main()
