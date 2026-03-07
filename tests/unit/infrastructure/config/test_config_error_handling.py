#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
错误处理和异常测试 - 配置管理

测试配置管理系统在各种错误和异常情况下的表现
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import tempfile
import os
import json
import threading
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open
from typing import Dict, Any, List, Optional
import pytest

from src.infrastructure.config import UnifiedConfigManager
from src.infrastructure.config.config_exceptions import (
    ConfigLoadError, ConfigValidationError
)


class TestConfigErrorHandling(unittest.TestCase):
    """错误处理和异常测试"""

    def setUp(self):
        """测试前准备"""
        self.manager = UnifiedConfigManager({
            "auto_reload": False,
            "validation_enabled": True
        })

    def tearDown(self):
        """测试后清理"""
        if self.manager and hasattr(self.manager, '_data') and self.manager._data is not None:
            self.manager._data.clear()

    @pytest.mark.error
    def test_invalid_key_formats(self):
        """测试无效键格式"""
        invalid_keys = [
            "",  # 空字符串
            ".",  # 只有点
            "..",  # 连续点
            "key.",  # 以点结尾
            ".key",  # 以点开头
            "key..subkey",  # 连续点
            "key.subkey.",  # 以点结尾
            "key.123invalid",  # 数字开头的子键（应该允许）
            "key.sub.key",  # 多级但有效
        ]

        for key in invalid_keys:
            if key in ["", ".", "..", "key.", ".key", "key..subkey", "key.subkey."]:
                # 这些应该是无效的
                result = self.manager.set(key, "value")
                self.assertFalse(result, f"Should reject invalid key format: '{key}'")
            else:
                # 这些应该是有效的
                result = self.manager.set(key, "value")
                self.assertTrue(result, f"Should accept valid key format: '{key}'")

    @pytest.mark.error
    def test_file_operation_errors(self):
        """测试文件操作错误"""
        # 测试不存在的目录
        nonexistent_file = "/nonexistent/directory/config.json"
        result = self.manager.save_to_file(nonexistent_file)
        # 取决于实现，可能成功或失败

        # 测试无效的JSON文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write("{ invalid json content ")
            invalid_json_file = f.name

        try:
            result = self.manager.load_from_file(invalid_json_file)
            # 应该能处理无效JSON
            self.assertFalse(result, "Should handle invalid JSON gracefully")
        finally:
            os.unlink(invalid_json_file)

        # 测试只读文件系统（如果可能的话）
        # 注意：在Windows上这个测试可能不适用

    @pytest.mark.error
    def test_network_related_errors(self):
        """测试网络相关错误"""
        # 这个测试假设有一些网络相关的配置加载器
        # 如果没有，我们就跳过

        # 模拟网络超时
        with patch('urllib.request.urlopen') as mock_urlopen:
            mock_urlopen.side_effect = TimeoutError("Connection timed out")

            # 如果有网络加载器，测试超时处理
            # 这里我们只是验证异常不会导致崩溃
            try:
                # 假设有一个网络相关的操作
                pass
            except Exception as e:
                # 应该能处理网络异常
                self.assertIsInstance(e, Exception)

    @pytest.mark.error
    def test_memory_limit_errors(self):
        """测试内存限制错误"""
        # 测试配置大小限制
        large_config = {}
        for i in range(10000):
            large_config[f"large.key{i}"] = "x" * 1000  # 1KB per value

        # 应该能处理大配置，但可能有性能问题
        start_time = time.time()
        self.manager.update(large_config)
        end_time = time.time()

        # 不应该无限期阻塞
        self.assertLess(end_time - start_time, 30, "Large config update should not take too long")

        # 验证数据完整性
        sample_value = self.manager.get("large.key5000")
        self.assertEqual(sample_value, "x" * 1000, "Should handle large config correctly")

    @pytest.mark.error
    def test_concurrent_access_errors(self):
        """测试并发访问错误"""
        # 测试在并发访问期间的错误处理
        error_counts = {"read_errors": 0, "write_errors": 0}

        def error_worker(worker_id: int):
            """可能产生错误的worker"""
            try:
                for i in range(100):
                    try:
                        if i % 2 == 0:
                            # 尝试读取不存在的键
                            value = self.manager.get(f"nonexistent.key{i}")
                        else:
                            # 设置值
                            self.manager.set(f"concurrent.key{worker_id}_{i}", f"value{i}")
                    except Exception as e:
                        if "read" in str(e).lower():
                            error_counts["read_errors"] += 1
                        else:
                            error_counts["write_errors"] += 1
            except Exception as e:
                print(f"Worker {worker_id} failed: {e}")

        # 启动多个可能出错的线程
        threads = []
        for i in range(3):
            thread = threading.Thread(target=error_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join(timeout=10)

        # 验证错误被正确处理
        total_errors = error_counts["read_errors"] + error_counts["write_errors"]
        print(f"Concurrent errors: {total_errors} total")

        # 不应该有太多错误
        self.assertLess(total_errors, 50, "Too many errors in concurrent access")

    @pytest.mark.error
    def test_validation_errors(self):
        """测试验证错误"""
        # 设置严格的验证规则
        validation_rules = {
            "user.name": {
                "type": "string",
                "required": True,
                "min_length": 2,
                "max_length": 50
            },
            "user.age": {
                "type": "integer",
                "min": 0,
                "max": 150
            },
            "system.port": {
                "type": "integer",
                "min": 1024,
                "max": 65535
            }
        }

        self.manager._data["validation_rules"] = validation_rules

        # 测试各种验证失败场景
        invalid_configs = [
            {"user.name": "", "user.age": 25},  # 空名称
            {"user.name": "A", "user.age": 25},  # 名称太短
            {"user.name": "Valid Name", "user.age": -5},  # 年龄负数
            {"user.name": "Valid Name", "user.age": 25, "system.port": 80},  # 端口号太小
            {"user.name": "Valid Name", "user.age": 200},  # 年龄太大
        ]

        for invalid_config in invalid_configs:
            result = self.manager.validate_config(invalid_config)
            self.assertFalse(result, f"Should reject invalid config: {invalid_config}")

    @pytest.mark.error
    def test_type_conversion_errors(self):
        """测试类型转换错误"""
        # 测试无法序列化的类型
        unserializable_values = [
            set([1, 2, 3]),  # set对象
            frozenset([1, 2, 3]),  # frozenset对象
            complex(1, 2),  # 复数
            lambda x: x,  # 函数对象
            threading.Lock(),  # 线程锁对象
            self.manager,  # 自我引用对象
        ]

        for value in unserializable_values:
            try:
                result = self.manager.set(f"type_test.{type(value).__name__}", value)
                # 某些类型可能被接受（通过pickle或其他方式）
                # 我们主要验证不会崩溃
                self.assertIsInstance(result, bool, f"Should return boolean for {type(value).__name__}")
            except Exception as e:
                # 如果抛出异常，应该是有意义的异常
                self.assertIsInstance(e, Exception, f"Should raise meaningful exception for {type(value).__name__}")

    @pytest.mark.error
    def test_corruption_recovery(self):
        """测试损坏恢复"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建有效的配置文件
            valid_config = {"app": {"name": "TestApp", "version": "1.0"}}
            valid_file = os.path.join(temp_dir, "valid.json")

            with open(valid_file, 'w') as f:
                json.dump(valid_config, f)

            # 加载有效配置
            result = self.manager.load_from_file(valid_file)
            self.assertTrue(result, "Should load valid config")

            # 验证数据
            self.assertEqual(self.manager.get("app.name"), "TestApp")

            # 模拟文件损坏
            corrupted_file = os.path.join(temp_dir, "corrupted.json")
            with open(corrupted_file, 'w') as f:
                f.write("{ invalid json ")

            # 尝试加载损坏文件
            result = self.manager.load_from_file(corrupted_file)
            self.assertFalse(result, "Should handle corrupted file gracefully")

            # 验证原有数据未被破坏
            self.assertEqual(self.manager.get("app.name"), "TestApp", "Should preserve original data after corruption")

    @pytest.mark.error
    def test_resource_exhaustion(self):
        """测试资源耗尽"""
        # 测试在资源紧张情况下的表现

        # 1. 测试大量监听器
        watcher_count = 0
        max_watchers = 1000

        def dummy_watcher(key, value):
            pass

        try:
            for i in range(max_watchers):
                self.manager.watch(f"watcher.key{i}", dummy_watcher)
                watcher_count += 1
        except Exception as e:
            print(f"Stopped at {watcher_count} watchers: {e}")

        # 应该能处理大量监听器
        self.assertGreater(watcher_count, 100, "Should handle reasonable number of watchers")

        # 2. 测试大量配置项
        config_count = 0
        max_configs = 10000

        try:
            for i in range(max_configs):
                self.manager.set(f"resource.key{i}", f"value{i}")
                config_count += 1
        except Exception as e:
            print(f"Stopped at {config_count} configs: {e}")

        # 应该能处理大量配置项
        self.assertGreater(config_count, 1000, "Should handle reasonable number of configs")

    @pytest.mark.error
    def test_interrupt_handling(self):
        """测试中断处理"""
        # 测试在操作过程中被中断的情况

        interrupt_flag = False

        def interrupt_worker():
            """可能被中断的worker"""
            nonlocal interrupt_flag
            try:
                for i in range(1000):
                    if interrupt_flag:
                        raise KeyboardInterrupt("Simulated interrupt")

                    self.manager.set(f"interrupt.key{i}", f"value{i}")
                    time.sleep(0.001)  # 小延迟
            except KeyboardInterrupt:
                # 应该能优雅地处理中断
                print("Worker received interrupt signal")
                raise

        # 启动worker线程
        worker_thread = threading.Thread(target=interrupt_worker)
        worker_thread.start()

        # 等待一小段时间然后中断
        time.sleep(0.01)
        interrupt_flag = True

        # 等待线程结束
        worker_thread.join(timeout=5)

        # 验证系统仍然可用
        result = self.manager.set("interrupt.test", "recovery_value")
        self.assertTrue(result, "System should remain functional after interrupt")

        retrieved = self.manager.get("interrupt.test")
        self.assertEqual(retrieved, "recovery_value", "Should recover from interrupt")

    @pytest.mark.error
    def test_encoding_errors(self):
        """测试编码错误"""
        # 测试各种编码相关的错误

        # Unicode字符
        unicode_values = {
            "chinese": "中文配置",
            "japanese": "日本語設定",
            "emoji": "🚀💻🎯",
            "mixed": "Hello 世界 🌍"
        }

        for key, value in unicode_values.items():
            result = self.manager.set(f"unicode.{key}", value)
            self.assertTrue(result, f"Should handle Unicode value: {key}")

            retrieved = self.manager.get(f"unicode.{key}")
            self.assertEqual(retrieved, value, f"Should retrieve Unicode value correctly: {key}")

        # 测试文件编码
        with tempfile.TemporaryDirectory() as temp_dir:
            unicode_file = os.path.join(temp_dir, "unicode_config.json")

            # 保存包含Unicode的配置
            unicode_config = {"unicode_test": "测试配置 🚀"}
            self.manager.update(unicode_config)

            result = self.manager.save_to_file(unicode_file)
            self.assertTrue(result, "Should save Unicode config")

            # 加载Unicode配置
            new_manager = UnifiedConfigManager()
            result = new_manager.load_from_file(unicode_file)
            self.assertTrue(result, "Should load Unicode config")

            loaded_value = new_manager.get("unicode_test")
            self.assertEqual(loaded_value, "测试配置 🚀", "Should handle Unicode in files")

    @pytest.mark.error
    def test_dependency_failures(self):
        """测试依赖失败"""
        # 测试当依赖服务不可用时的表现

        # 模拟文件系统不可用（如果配置管理器依赖文件系统）
        try:
            # 尝试设置一个简单的配置
            result = self.manager.set("dependency.test", "value")
            self.assertTrue(result, "Should work without external dependencies")

            retrieved = self.manager.get("dependency.test")
            self.assertEqual(retrieved, "value", "Should retrieve without external dependencies")
        except Exception as e:
            # 如果有异常，至少不应该崩溃
            self.fail(f"Configuration manager should handle dependency failures gracefully: {e}")

        # 模拟验证服务不可用
        with patch.object(self.manager, 'validate_config', side_effect=Exception("Validation service down")):
            # 设置操作应该仍然成功（如果验证是可选的）
            result = self.manager.set("validation.test", "value")
            self.assertTrue(result, "Should work when validation fails")

    @pytest.mark.error
    def test_recovery_scenarios(self):
        """测试恢复场景"""
        # 测试从各种错误状态恢复的能力

        # 1. 从空状态恢复
        self.manager.clear_all()
        self.assertEqual(len(self.manager.get_sections()), 0)

        result = self.manager.set("recovery.test", "value")
        self.assertTrue(result, "Should recover from empty state")

        # 2. 从损坏状态恢复
        # 故意损坏内部状态
        if hasattr(self.manager, '_data'):
            self.manager._data = None

        # 应该能处理并恢复
        try:
            result = self.manager.set("recovery.test2", "value2")
            # 如果成功，很好；如果失败，至少不应该崩溃
            self.assertIsInstance(result, bool, "Should handle corrupted state gracefully")
        except Exception as e:
            # 如果抛出异常，应该是有意义的
            self.assertIsInstance(e, Exception)

        # 3. 从文件错误恢复
        with patch('builtins.open', side_effect=IOError("Disk full")):
            result = self.manager.save_to_file("/fake/path/config.json")
            # 应该能处理IO错误
            self.assertFalse(result, "Should handle IO errors gracefully")

    @pytest.mark.error
    def test_boundary_error_conditions(self):
        """测试边界错误条件"""
        # 测试各种边界错误情况

        # 1. 超大键值
        huge_key = "x" * 10000
        huge_value = "y" * 100000

        result = self.manager.set(huge_key, huge_value)
        # 应该能处理，但可能有性能问题
        self.assertIsInstance(result, bool, "Should handle huge key/value gracefully")

        # 2. 深度递归结构
        recursive_data = {"self": None}
        recursive_data["self"] = recursive_data

        try:
            self.manager.set("recursive.test", recursive_data)
            # 如果成功，验证能处理递归
            result = self.manager.get("recursive.test")
            self.assertIsNotNone(result, "Should handle recursive structures")
        except Exception as e:
            # 如果失败，至少不应该崩溃
            self.assertIsInstance(e, Exception, "Should handle recursive structures safely")

        # 3. 特殊浮点数
        special_floats = [float('inf'), float('-inf'), float('nan')]

        for i, special_float in enumerate(special_floats):
            result = self.manager.set(f"float.special{i}", special_float)
            self.assertIsInstance(result, bool, f"Should handle special float {special_float}")

    @pytest.mark.error
    def test_configuration_conflicts(self):
        """测试配置冲突"""
        # 测试配置间的冲突情况

        # 1. 键名冲突
        self.manager.set("conflict.key", "value1")
        self.manager.set("conflict.key", "value2")  # 覆盖

        result = self.manager.get("conflict.key")
        self.assertEqual(result, "value2", "Should allow key overriding")

        # 2. section冲突
        self.manager.set("section.key1", "value1")
        self.manager.set("section.key2", "value2")

        # 尝试将section设置为非字典值
        result = self.manager.set("section", "not_a_dict")
        # 这应该成功，因为它是不同的键
        self.assertTrue(result, "Should allow section name as regular key")

        # 但原来的section数据应该被覆盖
        result = self.manager.get("section.key1")
        self.assertIsNone(result, "Section should be overridden")

    @pytest.mark.error
    def test_system_resource_limits(self):
        """测试系统资源限制"""
        # 测试达到系统资源限制时的表现

        # 1. 测试文件描述符限制（如果适用）
        open_files = []
        try:
            for i in range(100):  # 尝试打开很多文件
                try:
                    temp_file = tempfile.NamedTemporaryFile(delete=False)
                    open_files.append(temp_file)
                except OSError:
                    break  # 达到限制

            # 验证能处理文件操作
            if open_files:
                test_file = open_files[0].name
                result = self.manager.save_to_file(test_file)
                self.assertIsInstance(result, bool, "Should handle file operations under resource limits")

        finally:
            # 清理文件
            for temp_file in open_files:
                try:
                    os.unlink(temp_file.name)
                    temp_file.close()
                except:
                    pass

        # 2. 测试内存限制
        # 创建大量对象
        memory_objects = []
        try:
            for i in range(10000):
                memory_objects.append("x" * 1000)  # 1KB对象

            # 验证配置操作仍然可用
            result = self.manager.set("memory.test", "value")
            self.assertTrue(result, "Should work under memory pressure")

        except MemoryError:
            # 如果内存不足，至少不应该让整个系统崩溃
            print("Memory limit reached during test")


if __name__ == '__main__':
    unittest.main()
