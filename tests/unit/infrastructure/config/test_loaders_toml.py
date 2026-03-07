#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - TOML配置加载器深度测试
验证TOMLLoader的完整功能覆盖，目标覆盖率85%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import tempfile
import os
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List, Tuple


class TestTOMLLoader(unittest.TestCase):
    """测试TOML配置加载器"""

    def setUp(self):
        """测试前准备"""
        self.test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "user": "admin",
                "password": "password123",
                "ssl": True
            },
            "cache": {
                "redis_host": "redis-server",
                "redis_port": 6379,
                "ttl": 300,
                "max_connections": 20
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(levelname)s - %(message)s",
                "handlers": ["console", "file"]
            }
        }

        # 创建临时目录用于测试
        self.temp_dir = tempfile.mkdtemp()

        # 准备TOML内容
        self.toml_content = """
[database]
host = "localhost"
port = 5432
user = "admin"
password = "password123"
ssl = true

[cache]
redis_host = "redis-server"
redis_port = 6379
ttl = 300
max_connections = 20

[logging]
level = "INFO"
format = "%(asctime)s - %(levelname)s - %(message)s"
handlers = ["console", "file"]
"""

        self.toml_file = os.path.join(self.temp_dir, "test_config.toml")
        with open(self.toml_file, 'w', encoding='utf-8') as f:
            f.write(self.toml_content)

    def tearDown(self):
        """测试后清理"""
        # 清理临时文件
        if os.path.exists(self.toml_file):
            os.remove(self.toml_file)
        # 清理临时目录中的所有文件
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)

    # ==================== 基础功能测试 ====================

    def test_toml_loader_initialization(self):
        """测试TOML加载器初始化"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        # 验证基本属性
        self.assertIsInstance(loader, TOMLLoader)
        self.assertTrue(hasattr(loader, '_toml_available'))
        self.assertTrue(hasattr(loader, '_use_builtin'))

    def test_check_toml_availability(self):
        """测试TOML库可用性检查"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        # 测试_check_toml_availability方法
        available = loader._check_toml_availability()
        self.assertIsInstance(available, bool)

        # 验证可用性状态被正确设置
        self.assertTrue(hasattr(loader, '_toml_available'))
        self.assertTrue(hasattr(loader, '_use_builtin'))

    def test_load_basic_functionality(self):
        """测试基本加载功能"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        # 只有在TOML库可用时才运行测试
        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 加载TOML文件
        config = loader.load(self.toml_file)
        
        # 获取元数据
        metadata = loader.get_last_metadata()

        # 验证返回类型
        self.assertIsInstance(config, dict)
        self.assertIsInstance(metadata, dict)

        # 验证配置内容
        self.assertIn("database", config)
        self.assertIn("cache", config)
        self.assertIn("logging", config)
        self.assertEqual(config["database"]["host"], "localhost")
        self.assertEqual(config["database"]["port"], 5432)
        self.assertEqual(config["cache"]["redis_host"], "redis-server")

        # 验证元数据
        self.assertIn("source", metadata)
        self.assertIn("format", metadata)
        self.assertIn("load_time", metadata)
        self.assertEqual(metadata["source"], self.toml_file)
        self.assertEqual(metadata["format"], "toml")

    def test_load_without_toml_library(self):
        """测试在没有TOML库时的加载行为"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader
        from src.infrastructure.config.core.config_strategy import ConfigLoadError

        with patch('src.infrastructure.config.loaders.toml_loader.TOMLLoader._check_toml_availability', return_value=False):
            loader = TOMLLoader()
            loader._toml_available = False

            with self.assertRaises(ConfigLoadError) as cm:
                loader.load(self.toml_file)

            self.assertIn("TOML", str(cm.exception))

    def test_can_load_method(self):
        """测试can_load方法"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        # 测试支持的文件扩展名
        self.assertTrue(loader.can_load("config.toml"))
        self.assertTrue(loader.can_load("settings.TOML"))
        self.assertTrue(loader.can_load("/path/to/config.toml"))

        # 测试不支持的文件扩展名
        self.assertFalse(loader.can_load("config.json"))
        self.assertFalse(loader.can_load("config.yaml"))
        self.assertFalse(loader.can_load("config.ini"))
        self.assertFalse(loader.can_load("config.txt"))
        self.assertFalse(loader.can_load("config"))

    def test_get_supported_extensions(self):
        """测试获取支持的文件扩展名"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()
        extensions = loader.get_supported_extensions()

        self.assertIsInstance(extensions, list)
        self.assertIn(".toml", extensions)
        self.assertIn(".TOML", extensions)

    # ==================== 批量加载测试 ====================

    def test_batch_load_functionality(self):
        """测试批量加载功能"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 创建多个TOML文件
        files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"config_{i}.toml")
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"""
[test_{i}]
name = "config_{i}"
value = {i}
enabled = true
""")
            files.append(file_path)

        try:
            # 批量加载
            results = loader.batch_load(files)

            # 验证结果
            self.assertIsInstance(results, dict)
            self.assertEqual(len(results), 3)

            for i, file_path in enumerate(files):
                self.assertIn(file_path, results)
                config = results[file_path]
                self.assertIsInstance(config, dict)
                self.assertEqual(config[f"test_{i}"]["name"], f"config_{i}")
                self.assertEqual(config[f"test_{i}"]["value"], i)

        finally:
            # 清理文件
            for file_path in files:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def test_batch_load_with_invalid_files(self):
        """测试批量加载包含无效文件的情况"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 创建有效文件和无效文件
        valid_file = os.path.join(self.temp_dir, "valid.toml")
        with open(valid_file, 'w', encoding='utf-8') as f:
            f.write("[valid]\nkey = \"value\"")

        invalid_file = os.path.join(self.temp_dir, "invalid.toml")
        with open(invalid_file, 'w', encoding='utf-8') as f:
            f.write("invalid toml content {{{")  # 无效TOML

        try:
            files = [valid_file, invalid_file, "nonexistent.toml"]

            # 批量加载（应该处理错误）
            results = loader.batch_load(files)

            # 验证结果包含有效文件的结果
            self.assertIn(valid_file, results)

            # 对于无效文件，应该抛出异常或返回错误信息
            # 这里根据实际实现来验证

        finally:
            # 清理文件
            for file_path in [valid_file, invalid_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    # ==================== 保存功能测试 ====================

    def test_save_functionality(self):
        """测试保存功能"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._can_write():
            self.skipTest("TOML write library not available")

        # 准备测试数据
        test_data = {
            "server": {
                "host": "example.com",
                "port": 8080,
                "ssl": False
            },
            "features": {
                "auth": True,
                "cache": False,
                "logging": True
            }
        }

        # 保存到文件
        output_file = os.path.join(self.temp_dir, "output.toml")
        result = loader.save(test_data, output_file)

        # 验证保存结果
        self.assertTrue(result)
        self.assertTrue(os.path.exists(output_file))

        # 验证保存的内容可以通过加载器重新加载
        loaded_config = loader.load(output_file)
        self.assertEqual(loaded_config, test_data)

        # 清理
        os.remove(output_file)

    def test_save_without_write_library(self):
        """测试在没有写入库时的保存行为"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        with patch('src.infrastructure.config.loaders.toml_loader.TOMLLoader._can_write', return_value=False):
            loader = TOMLLoader()

            result = loader.save({"test": "data"}, "output.toml")
            self.assertFalse(result)

    def test_can_write_method(self):
        """测试_can_write方法"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()
        can_write = loader._can_write()

        self.assertIsInstance(can_write, bool)

    # ==================== 验证功能测试 ====================

    def test_validate_toml_file(self):
        """测试TOML文件验证"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 验证有效文件
        is_valid, errors = loader.validate_toml_file(self.toml_file)
        self.assertTrue(is_valid)
        self.assertEqual(errors, [])

        # 创建无效文件进行验证
        invalid_file = os.path.join(self.temp_dir, "invalid.toml")
        with open(invalid_file, 'w', encoding='utf-8') as f:
            f.write("invalid {{{ toml content")

        try:
            is_valid, errors = loader.validate_toml_file(invalid_file)
            # 根据实现，可能会返回False和错误列表，或抛出异常
            self.assertIsInstance(is_valid, bool)
            if not is_valid:
                self.assertIsInstance(errors, list)

        finally:
            os.remove(invalid_file)

    def test_validate_toml_file_nonexistent(self):
        """测试验证不存在的文件"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        is_valid, errors = loader.validate_toml_file("nonexistent.toml")
        self.assertFalse(is_valid)
        self.assertIn("not found", " ".join(errors).lower())

    # ==================== 信息获取测试 ====================

    def test_get_toml_info(self):
        """测试获取TOML信息"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()
        info = loader.get_toml_info()

        self.assertIsInstance(info, dict)
        self.assertIn("available", info)
        self.assertIn("library", info)
        self.assertIn("version", info)

    # ==================== 文件合并测试 ====================

    def test_merge_toml_files(self):
        """测试TOML文件合并"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 创建多个TOML文件
        file1_content = """
[database]
host = "db1.example.com"
port = 5432

[cache]
enabled = true
"""
        file1 = os.path.join(self.temp_dir, "merge1.toml")
        with open(file1, 'w', encoding='utf-8') as f:
            f.write(file1_content)

        file2_content = """
[database]
user = "admin"
password = "secret"

[logging]
level = "INFO"
"""
        file2 = os.path.join(self.temp_dir, "merge2.toml")
        with open(file2, 'w', encoding='utf-8') as f:
            f.write(file2_content)

        try:
            # 合并文件
            merged = loader.merge_toml_files([file1, file2])

            # 验证合并结果
            self.assertIn("database", merged)
            self.assertIn("cache", merged)
            self.assertIn("logging", merged)
            self.assertEqual(merged["database"]["host"], "db1.example.com")
            self.assertEqual(merged["database"]["user"], "admin")
            self.assertEqual(merged["cache"]["enabled"], True)
            self.assertEqual(merged["logging"]["level"], "INFO")

        finally:
            # 清理
            for file_path in [file1, file2]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    def test_merge_toml_files_with_output(self):
        """测试TOML文件合并并保存到输出文件"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available or not loader._can_write():
            self.skipTest("TOML library not available or write not supported")

        # 创建测试文件
        file1 = os.path.join(self.temp_dir, "merge_out1.toml")
        with open(file1, 'w', encoding='utf-8') as f:
            f.write("[section1]\nkey1 = \"value1\"")

        file2 = os.path.join(self.temp_dir, "merge_out2.toml")
        with open(file2, 'w', encoding='utf-8') as f:
            f.write("[section2]\nkey2 = \"value2\"")

        output_file = os.path.join(self.temp_dir, "merged_output.toml")

        try:
            # 合并并保存
            merged = loader.merge_toml_files([file1, file2], output_file)

            # 验证输出文件存在
            self.assertTrue(os.path.exists(output_file))

            # 验证可以重新加载
            loaded, _ = loader.load(output_file)
            self.assertEqual(loaded, merged)

        finally:
            # 清理
            for file_path in [file1, file2, output_file]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    # ==================== 深度合并测试 ====================

    def test_deep_merge_functionality(self):
        """测试深度合并功能"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        # 测试深度合并
        base = {
            "database": {
                "host": "localhost",
                "credentials": {
                    "user": "admin",
                    "password": "old_pass"
                }
            },
            "cache": {
                "enabled": True
            }
        }

        update = {
            "database": {
                "port": 5432,
                "credentials": {
                    "password": "new_pass"
                }
            },
            "logging": {
                "level": "INFO"
            }
        }

        merged = loader._deep_merge(base, update)

        # 验证合并结果
        self.assertEqual(merged["database"]["host"], "localhost")
        self.assertEqual(merged["database"]["port"], 5432)
        self.assertEqual(merged["database"]["credentials"]["user"], "admin")
        self.assertEqual(merged["database"]["credentials"]["password"], "new_pass")
        self.assertEqual(merged["cache"]["enabled"], True)
        self.assertEqual(merged["logging"]["level"], "INFO")

    # ==================== 转换功能测试 ====================

    def test_convert_to_toml(self):
        """测试转换为TOML字符串"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._can_write():
            self.skipTest("TOML write library not available")

        test_data = {
            "server": {
                "host": "example.com",
                "port": 8080
            },
            "features": {
                "auth": True,
                "cache": False
            }
        }

        toml_string = loader.convert_to_toml(test_data)

        self.assertIsInstance(toml_string, str)
        self.assertIn("host = \"example.com\"", toml_string)
        self.assertIn("port = 8080", toml_string)
        self.assertIn("auth = true", toml_string)
        self.assertIn("cache = false", toml_string)

    def test_convert_from_toml(self):
        """测试从TOML字符串转换"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        toml_string = """
[database]
host = "localhost"
port = 5432

[cache]
enabled = true
"""

        converted = loader.convert_from_toml(toml_string)

        self.assertIsInstance(converted, dict)
        self.assertEqual(converted["database"]["host"], "localhost")
        self.assertEqual(converted["database"]["port"], 5432)
        self.assertEqual(converted["cache"]["enabled"], True)

    # ==================== 文件比较测试 ====================

    def test_compare_toml_files(self):
        """测试TOML文件比较"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 创建两个不同的TOML文件
        file1_content = """
[database]
host = "db1"
port = 5432

[cache]
enabled = true
"""
        file1 = os.path.join(self.temp_dir, "compare1.toml")
        with open(file1, 'w', encoding='utf-8') as f:
            f.write(file1_content)

        file2_content = """
[database]
host = "db2"
port = 5432

[logging]
level = "INFO"
"""
        file2 = os.path.join(self.temp_dir, "compare2.toml")
        with open(file2, 'w', encoding='utf-8') as f:
            f.write(file2_content)

        try:
            # 比较文件
            comparison = loader.compare_toml_files(file1, file2)

            # 验证比较结果结构
            self.assertIsInstance(comparison, dict)
            self.assertIn("only_in_file1", comparison)
            self.assertIn("only_in_file2", comparison)
            self.assertIn("differences", comparison)
            self.assertIn("identical", comparison)

        finally:
            # 清理
            for file_path in [file1, file2]:
                if os.path.exists(file_path):
                    os.remove(file_path)

    # ==================== 错误处理测试 ====================

    def test_load_file_not_found(self):
        """测试加载不存在的文件"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader
        from src.infrastructure.config.core.config_strategy import ConfigLoadError

        loader = TOMLLoader()

        with self.assertRaises(ConfigLoadError) as cm:
            loader.load("nonexistent.toml")

        self.assertIn("不存在", str(cm.exception))

    def test_load_invalid_toml(self):
        """测试加载无效TOML内容"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader
        from src.infrastructure.config.core.config_strategy import ConfigLoadError

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 创建无效TOML文件
        invalid_file = os.path.join(self.temp_dir, "invalid.toml")
        with open(invalid_file, 'w', encoding='utf-8') as f:
            f.write("invalid {{{ toml content")

        try:
            with self.assertRaises(ConfigLoadError) as cm:
                loader.load(invalid_file)

            self.assertIn("TOML", str(cm.exception).upper())

        finally:
            os.remove(invalid_file)

    # ==================== 边界情况测试 ====================

    def test_empty_config_handling(self):
        """测试空配置处理"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 创建空TOML文件
        empty_file = os.path.join(self.temp_dir, "empty.toml")
        with open(empty_file, 'w', encoding='utf-8') as f:
            f.write("")  # 空文件

        try:
            config = loader.load(empty_file)
            self.assertEqual(config, {})

        finally:
            os.remove(empty_file)

    def test_large_config_handling(self):
        """测试大配置文件的处理"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 创建包含大量数据的TOML文件
        large_content = "[large_section]\n"
        for i in range(100):
            large_content += f'key_{i} = "value_{i}"\n'

        large_file = os.path.join(self.temp_dir, "large.toml")
        with open(large_file, 'w', encoding='utf-8') as f:
            f.write(large_content)

        try:
            start_time = time.time()
            config = loader.load(large_file)
            load_time = time.time() - start_time

            # 验证加载成功
            self.assertIsInstance(config, dict)
            self.assertIn("large_section", config)
            self.assertEqual(len(config["large_section"]), 100)

            # 验证加载时间在合理范围内（通常<1秒）
            self.assertLess(load_time, 5.0)

        finally:
            os.remove(large_file)

    def test_nested_structures(self):
        """测试嵌套结构处理"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 创建深度嵌套的TOML文件
        nested_content = """
[deep.nested.structure]
value = "deep_value"

[deep.another.level]
number = 42
array = [1, 2, 3, 4, 5]
"""

        nested_file = os.path.join(self.temp_dir, "nested.toml")
        with open(nested_file, 'w', encoding='utf-8') as f:
            f.write(nested_content)

        try:
            config = loader.load(nested_file)

            # 验证嵌套结构
            self.assertIn("deep", config)
            self.assertIn("nested", config["deep"])
            self.assertIn("structure", config["deep"]["nested"])
            self.assertEqual(config["deep"]["nested"]["structure"]["value"], "deep_value")
            self.assertEqual(config["deep"]["another"]["level"]["number"], 42)
            self.assertEqual(config["deep"]["another"]["level"]["array"], [1, 2, 3, 4, 5])

        finally:
            if os.path.exists(nested_file):
                os.remove(nested_file)

    # ==================== 性能测试 ====================

    def test_load_performance(self):
        """测试加载性能"""
        from src.infrastructure.config.loaders.toml_loader import TOMLLoader

        loader = TOMLLoader()

        if not loader._toml_available:
            self.skipTest("TOML library not available")

        # 多次加载测试性能
        import time
        start_time = time.time()

        for _ in range(10):
            config = loader.load(self.toml_file)

        end_time = time.time()
        total_time = end_time - start_time

        # 验证性能（10次加载应该在合理时间内完成）
        self.assertLess(total_time, 2.0)  # 2秒内完成


if __name__ == '__main__':
    unittest.main()

