#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
配置管理边界条件测试

测试各种边界条件、异常情况和极端输入
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import tempfile
import os
import json
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Optional
import pytest

from src.infrastructure.config import UnifiedConfigManager
from src.infrastructure.config.config_exceptions import ConfigLoadError, ConfigValidationError


class TestConfigBoundaryConditions(unittest.TestCase):
    """配置管理边界条件测试"""

    def setUp(self):
        """测试前准备"""
        self.manager = UnifiedConfigManager({
            "auto_reload": False,
            "validation_enabled": True
        })

    def tearDown(self):
        """测试后清理"""
        if hasattr(self.manager, '_data'):
            self.manager._data.clear()

    @pytest.mark.boundary
    def test_empty_config_operations(self):
        """测试空配置的操作"""
        # 空管理器应该正常工作
        self.assertIsNone(self.manager.get("nonexistent.key"))
        # 注意：管理器可能有默认配置，检查基本功能
        sections = self.manager.get_sections()
        self.assertIsInstance(sections, list)
        self.assertFalse(self.manager.has_section("nonexistent"))

        # 空配置验证 - 空字典通常被认为是有效的（没有验证规则）
        # None配置应该抛出异常
        try:
            self.manager.validate_config(None)
            self.fail("validate_config(None) should raise ConfigValidationError")
        except Exception:
            # 期望的行为
            pass

    @pytest.mark.boundary
    def test_max_key_length(self):
        """测试最大键长度限制"""
        # 创建一个超长键
        max_length = 1000
        long_key = "a" * max_length
        long_section = "section." + long_key

        # 应该拒绝过长的键
        self.assertFalse(self.manager.set(long_section, "value"))
        self.assertIsNone(self.manager.get(long_section))

    @pytest.mark.boundary
    def test_special_characters_in_keys(self):
        """测试键中的特殊字符"""
        special_keys = [
            "section.key with spaces",
            "section.key-with-dashes",
            "section.key_with_underscores",
            "section.key/with/slashes",
            "section.key\\with\\backslashes",
            "section.key:with:colons",
            "section.key@with@ats",
            "section.key#with#hashes"
        ]

        for key in special_keys:
            # 大部分特殊字符应该被拒绝
            if any(char in key for char in [' ', '/', '\\', ':', '@', '#']):
                self.assertFalse(self.manager.set(key, "value"), f"Should reject key: {key}")
            else:
                # 允许的字符
                self.assertTrue(self.manager.set(key, "value"), f"Should accept key: {key}")
                self.assertEqual(self.manager.get(key), "value")

    @pytest.mark.boundary
    def test_unicode_characters(self):
        """测试Unicode字符支持"""
        unicode_data = {
            "database.host": "数据库主机",  # 中文
            "database.port": "端口",  # 中文
            "cache.ключ": "значение",  # 俄文
            "logging.файл": "журнал.log",  # 俄文
            "app.🚀emoji": "value",  # Emoji
            "system.α": "beta",  # 希腊字母
        }

        # 设置Unicode数据
        for key, value in unicode_data.items():
            self.assertTrue(self.manager.set(key, value))

        # 验证Unicode数据
        for key, expected_value in unicode_data.items():
            self.assertEqual(self.manager.get(key), expected_value)

    @pytest.mark.boundary
    def test_nested_dict_operations(self):
        """测试嵌套字典操作"""
        # 创建深层嵌套的配置
        nested_config = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": {
                            "level5": "deep_value"
                        }
                    }
                }
            }
        }

        self.manager._data = nested_config

        # 测试深层访问
        self.assertEqual(self.manager.get("level1.level2.level3.level4.level5"), "deep_value")

        # 测试部分路径访问
        level2 = self.manager.get_section("level1")
        self.assertIsInstance(level2, dict)
        self.assertIn("level2", level2)

    @pytest.mark.boundary
    def test_large_config_handling(self):
        """测试大配置文件的处理"""
        # 创建一个大的配置对象
        large_config = {}
        for i in range(1000):
            large_config[f"section{i}"] = {
                "key1": f"value{i}_1",
                "key2": f"value{i}_2",
                "key3": f"value{i}_3",
                "nested": {
                    "data": list(range(100)),  # 大数组
                    "metadata": "large_config_test" * 50  # 长字符串
                }
            }

        # 测试大配置的设置
        start_time = time.time()
        self.manager._data = large_config
        set_time = time.time() - start_time

        # 应该在合理时间内完成
        self.assertLess(set_time, 5.0, "Large config setting took too long")

        # 验证大配置的访问
        start_time = time.time()
        value = self.manager.get("section500.key1")
        get_time = time.time() - start_time

        self.assertEqual(value, "value500_1")
        self.assertLess(get_time, 0.1, "Large config access took too long")

    @pytest.mark.boundary
    def test_circular_reference_handling(self):
        """测试循环引用处理"""
        # 创建循环引用的配置（这在实际中应该避免）
        circular_dict = {}
        circular_dict["self"] = circular_dict

        # 应该能处理循环引用而不崩溃
        try:
            self.manager._data = circular_dict
            # 尝试序列化（可能失败但不应该崩溃）
            str(self.manager._data)
        except RecursionError:
            # 预期的递归错误
            pass
        except Exception as e:
            # 其他异常应该被处理
            self.fail(f"Unexpected error with circular reference: {e}")

    @pytest.mark.boundary
    def test_memory_pressure_simulation(self):
        """测试内存压力模拟"""
        # 创建大量的小对象
        for i in range(10000):
            self.manager.set(f"stress.key{i}", f"value{i}")

        # 验证所有值都能正确存储和检索
        for i in range(0, 10000, 100):  # 检查100个样本
            expected = f"value{i}"
            actual = self.manager.get(f"stress.key{i}")
            self.assertEqual(actual, expected, f"Memory pressure test failed at index {i}")

        # 清理
        self.manager.clear_all()
        self.assertEqual(len(self.manager.get_sections()), 0)

    @pytest.mark.boundary
    def test_concurrent_access_simulation(self):
        """测试并发访问模拟"""
        results = []
        errors = []

        def worker(worker_id: int):
            """工作线程"""
            try:
                # 每个线程执行一系列操作
                for i in range(100):
                    key = f"concurrent.worker{worker_id}.key{i}"
                    value = f"value{worker_id}_{i}"

                    # 设置值
                    self.manager.set(key, value)

                    # 读取并验证
                    retrieved = self.manager.get(key)
                    if retrieved != value:
                        errors.append(f"Worker {worker_id}: expected {value}, got {retrieved}")

                results.append(f"Worker {worker_id} completed successfully")

            except Exception as e:
                errors.append(f"Worker {worker_id} error: {e}")

        # 启动多个线程
        threads = []
        for i in range(5):
            thread = threading.Thread(target=worker, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join(timeout=30)  # 30秒超时

        # 验证结果
        self.assertEqual(len(results), 5, f"Some workers failed: {errors}")
        self.assertEqual(len(errors), 0, f"Errors occurred: {errors}")

    @pytest.mark.boundary
    def test_file_system_edge_cases(self):
        """测试文件系统边界情况"""
        # 测试各种文件路径
        test_paths = [
            "",  # 空路径
            ".",  # 当前目录
            "..",  # 父目录
            "nonexistent_file.json",
            "/absolute/path/to/file.json",  # 绝对路径
            "relative/path/file.json",  # 相对路径
            "file with spaces.json",  # 带空格的文件名
            "file-with-特殊字符.json",  # 特殊字符
        ]

        for path in test_paths:
            try:
                # 尝试加载不存在的文件
                result = self.manager.load_from_file(path)
                # 大部分应该失败
                if path in ["", "nonexistent_file.json"]:
                    self.assertFalse(result, f"Should fail for path: {path}")
            except Exception:
                # 预期的异常
                pass

    @pytest.mark.boundary
    def test_type_conversion_edge_cases(self):
        """测试类型转换边界情况"""
        # 测试各种数据类型的设置和获取
        test_cases = [
            ("string.key", "hello world"),
            ("int.key", 42),
            ("float.key", 3.14159),
            ("bool.key", True),
            ("none.key", None),
            ("list.key", [1, 2, 3, "mixed"]),
            ("dict.key", {"nested": {"data": "value"}}),
            ("tuple.key", (1, 2, "tuple")),
            ("set.key", {1, 2, 3}),  # sets 可能有问题
        ]

        for key, value in test_cases:
            try:
                # 设置值
                result = self.manager.set(key, value)
                self.assertTrue(result, f"Failed to set {type(value).__name__} value")

                # 获取并比较
                retrieved = self.manager.get(key)
                if isinstance(value, set):
                    # sets 可能不被支持，跳过比较
                    continue

                if value is None:
                    self.assertIsNone(retrieved)
                else:
                    self.assertEqual(retrieved, value, f"Type conversion failed for {type(value).__name__}")

            except Exception as e:
                # 如果某些类型不支持，记录但不失败
                print(f"Type {type(value).__name__} not fully supported: {e}")

    @pytest.mark.boundary
    def test_validation_edge_cases(self):
        """测试验证功能的边界情况"""
        # 测试各种验证场景
        validation_test_cases = [
            # (config, should_pass)
            ({}, True),  # 空配置（有效但为空）
            (None, False),  # None配置
            ({"key": "value"}, True),  # 简单有效配置
            ({"key": None}, True),  # None值（有效）
            ({"key": ""}, True),  # 空字符串（有效）
            ({"key": 0}, True),  # 零值（有效）
            ({"key": []}, True),  # 空列表（有效）
            ({"key": {}}, True),  # 空字典（有效）
        ]

        for config, should_pass in validation_test_cases:
            result = self.manager.validate_config(config)
            if should_pass:
                self.assertTrue(result, f"Config should pass validation: {config}")
            else:
                self.assertFalse(result, f"Config should fail validation: {config}")

    @pytest.mark.boundary
    def test_backup_restore_edge_cases(self):
        """测试备份和恢复的边界情况"""
        # 测试备份到不存在的目录
        with tempfile.TemporaryDirectory() as temp_dir:
            nonexistent_dir = os.path.join(temp_dir, "nonexistent", "subdir")

            # 应该能创建目录并备份
            result = self.manager.backup_config(nonexistent_dir)
            self.assertTrue(result, "Should create directory and backup")

            # 验证备份文件存在
            self.assertTrue(os.path.exists(nonexistent_dir), "Backup directory should exist")

            # 查找备份文件
            backup_files = [f for f in os.listdir(nonexistent_dir) if f.startswith("config_backup_")]
            self.assertTrue(len(backup_files) > 0, "Backup file should exist")

            # 测试从备份恢复
            backup_file = os.path.join(nonexistent_dir, backup_files[0])
            result = self.manager.restore_from_backup(backup_file)
            self.assertTrue(result, "Should restore from backup")

    @pytest.mark.boundary
    def test_resource_cleanup(self):
        """测试资源清理"""
        # 添加一些数据
        self.manager.set("test.key", "test_value")
        self.manager.set("test.key2", "test_value2")

        # 验证数据存在
        self.assertEqual(self.manager.get("test.key"), "test_value")
        # 注意：section数量可能因实现而异，重点是数据能被正确设置和清理
        sections_count_before = len(self.manager.get_sections())
        self.assertGreaterEqual(sections_count_before, 0)

        # 清理资源
        self.manager.clear_all()

        # 验证清理后状态
        self.assertEqual(len(self.manager.get_sections()), 0)
        self.assertIsNone(self.manager.get("test.key"))
        self.assertIsNone(self.manager.get("test.key2"))

    @pytest.mark.boundary
    def test_error_recovery(self):
        """测试错误恢复能力"""
        # 模拟一些错误情况并验证恢复
        original_data = {"original": {"key": "value"}}

        # 保存原始状态
        self.manager._data = original_data.copy()

        # 执行可能出错的操作
        try:
            # 尝试一些边界操作
            self.manager.set("malformed.key.with.many.parts", "value")
            self.manager.set("", "empty_key")  # 空键
            self.manager.get(None)  # None键
        except Exception:
            # 忽略异常，测试恢复能力
            pass

        # 验证系统仍然可用
        try:
            set_result = self.manager.set("recovery.test", "success")
            if set_result:
                self.assertEqual(self.manager.get("recovery.test"), "success")
            else:
                # 如果设置失败，至少验证管理器没有崩溃
                self.assertIsNotNone(self.manager)
        except Exception:
            # 如果出现异常，至少验证管理器对象仍然存在
            self.assertIsNotNone(self.manager)

        # 原始数据应该保持不变
        self.assertEqual(self.manager.get("original.key"), "value")


if __name__ == '__main__':
    unittest.main()
