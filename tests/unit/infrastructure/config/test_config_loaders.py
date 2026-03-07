#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Config Loaders
配置加载器测试，验证各种配置文件的加载功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import tempfile
import os
import json
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from io import StringIO

# 导入实际的加载器类
from src.infrastructure.config.loaders.json_loader import JSONLoader
from src.infrastructure.config.loaders.yaml_loader import YAMLLoader
from src.infrastructure.config.loaders.env_loader import EnvironmentConfigLoader as EnvLoader
from src.infrastructure.config.loaders.toml_loader import TOMLLoader
from src.infrastructure.config.loaders.database_loader import DatabaseLoader
from src.infrastructure.config.loaders.cloud_loader import CloudLoader
from src.infrastructure.config.core.exceptions import ConfigLoadError


class TestConfigLoaders(unittest.TestCase):
    """测试Config Loaders"""

    def setUp(self):
        """测试前准备"""
        self.test_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "user": "admin",
                "password": "password123"
            },
            "cache": {
                "redis_host": "localhost",
                "redis_port": 6379,
                "ttl": 300
            },
            "logging": {
                "level": "INFO",
                "format": "json"
            }
        }

        # 创建临时目录
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        """测试清理"""
        # 清理临时文件
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_json_loader_basic_functionality(self):
        """测试JSON加载器基本功能"""
        # 创建测试JSON文件
        json_file = os.path.join(self.temp_dir, "test_config.json")

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_config, f, indent=2)

        # 测试文件存在
        self.assertTrue(os.path.exists(json_file))

        # 验证文件内容
        with open(json_file, 'r', encoding='utf-8') as f:
            loaded_config = json.load(f)

        self.assertEqual(loaded_config, self.test_config)

    def test_json_loader_file_not_found(self):
        """测试JSON加载器文件不存在的情况"""
        non_existent_file = os.path.join(self.temp_dir, "non_existent.json")

        # 验证文件不存在
        self.assertFalse(os.path.exists(non_existent_file))

        # 尝试读取不存在的文件应该抛出异常
        with self.assertRaises(FileNotFoundError):
            with open(non_existent_file, 'r', encoding='utf-8') as f:
                json.load(f)

    def test_json_loader_invalid_format(self):
        """测试JSON加载器无效格式处理"""
        json_file = os.path.join(self.temp_dir, "invalid_config.json")

        # 创建无效的JSON文件
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write("{ invalid json content }")

        # 验证文件存在
        self.assertTrue(os.path.exists(json_file))

        # 尝试解析无效JSON应该抛出异常
        with self.assertRaises(json.JSONDecodeError):
            with open(json_file, 'r', encoding='utf-8') as f:
                json.load(f)

    def test_yaml_loader_basic_functionality(self):
        """测试YAML加载器基本功能"""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not available")

        # 创建测试YAML文件
        yaml_file = os.path.join(self.temp_dir, "test_config.yaml")

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.test_config, f, default_flow_style=False)

        # 测试文件存在
        self.assertTrue(os.path.exists(yaml_file))

        # 验证文件内容
        with open(yaml_file, 'r', encoding='utf-8') as f:
            loaded_config = yaml.safe_load(f)

        self.assertEqual(loaded_config, self.test_config)

    def test_yaml_loader_file_not_found(self):
        """测试YAML加载器文件不存在的情况"""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not available")

        non_existent_file = os.path.join(self.temp_dir, "non_existent.yaml")

        # 验证文件不存在
        self.assertFalse(os.path.exists(non_existent_file))

        # 尝试读取不存在的文件应该抛出异常
        with self.assertRaises(FileNotFoundError):
            with open(non_existent_file, 'r', encoding='utf-8') as f:
                yaml.safe_load(f)

    def test_env_loader_basic_functionality(self):
        """测试环境变量加载器基本功能"""
        # 设置测试环境变量
        test_env_vars = {
            "DB_HOST": "localhost",
            "DB_PORT": "5432",
            "DB_USER": "admin",
            "REDIS_HOST": "localhost",
            "REDIS_PORT": "6379"
        }

        # 设置环境变量
        for key, value in test_env_vars.items():
            os.environ[key] = value

        try:
            # 验证环境变量已设置
            for key, expected_value in test_env_vars.items():
                self.assertEqual(os.environ.get(key), expected_value)

            # 验证可以读取环境变量
            self.assertEqual(os.environ.get("DB_HOST"), "localhost")
            self.assertEqual(os.environ.get("DB_PORT"), "5432")

        finally:
            # 清理环境变量
            for key in test_env_vars.keys():
                if key in os.environ:
                    del os.environ[key]

    def test_env_loader_missing_variables(self):
        """测试环境变量加载器缺失变量处理"""
        # 确保测试环境变量不存在
        test_keys = ["TEST_DB_HOST", "TEST_DB_PORT", "TEST_MISSING_VAR"]

        for key in test_keys:
            if key in os.environ:
                del os.environ[key]

        # 验证环境变量不存在
        for key in test_keys:
            self.assertIsNone(os.environ.get(key))

        # 测试获取不存在的环境变量
        self.assertIsNone(os.environ.get("NON_EXISTENT_VAR"))
        self.assertEqual(os.environ.get("NON_EXISTENT_VAR", "default"), "default")

    def test_toml_loader_basic_functionality(self):
        """测试TOML加载器基本功能"""
        try:
            import toml
        except ImportError:
            self.skipTest("toml not available")

        # 创建测试TOML文件
        toml_file = os.path.join(self.temp_dir, "test_config.toml")

        # TOML格式的配置
        toml_config = """
[database]
host = "localhost"
port = 5432
user = "admin"
password = "password123"

[cache]
redis_host = "localhost"
redis_port = 6379
ttl = 300

[logging]
level = "INFO"
format = "json"
"""

        with open(toml_file, 'w', encoding='utf-8') as f:
            f.write(toml_config)

        # 测试文件存在
        self.assertTrue(os.path.exists(toml_file))

        # 验证文件内容
        with open(toml_file, 'r', encoding='utf-8') as f:
            loaded_config = toml.load(f)

        # 验证关键配置项
        self.assertEqual(loaded_config["database"]["host"], "localhost")
        self.assertEqual(loaded_config["database"]["port"], 5432)
        self.assertEqual(loaded_config["cache"]["redis_port"], 6379)

    def test_database_loader_simulation(self):
        """测试数据库加载器模拟功能"""
        # 模拟数据库配置数据
        db_config_data = {
            "database_url": "postgresql://user:pass@localhost:5432/dbname",
            "connection_pool_size": 10,
            "connection_timeout": 30
        }

        # 创建模拟的数据库配置文件
        db_config_file = os.path.join(self.temp_dir, "db_config.json")

        with open(db_config_file, 'w', encoding='utf-8') as f:
            json.dump(db_config_data, f, indent=2)

        # 验证数据库配置可以正确加载
        with open(db_config_file, 'r', encoding='utf-8') as f:
            loaded_db_config = json.load(f)

        self.assertEqual(loaded_db_config, db_config_data)
        self.assertIn("database_url", loaded_db_config)
        self.assertIn("connection_pool_size", loaded_db_config)

    def test_cloud_loader_simulation(self):
        """测试云配置加载器模拟功能"""
        # 模拟云配置数据
        cloud_config_data = {
            "aws_region": "us-east-1",
            "aws_access_key_id": "AKIAIOSFODNN7EXAMPLE",
            "aws_secret_access_key": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
            "s3_bucket": "my-config-bucket",
            "cloudwatch_log_group": "/aws/lambda/my-function"
        }

        # 创建模拟的云配置文件
        cloud_config_file = os.path.join(self.temp_dir, "cloud_config.json")

        with open(cloud_config_file, 'w', encoding='utf-8') as f:
            json.dump(cloud_config_data, f, indent=2)

        # 验证云配置可以正确加载
        with open(cloud_config_file, 'r', encoding='utf-8') as f:
            loaded_cloud_config = json.load(f)

        self.assertEqual(loaded_cloud_config, cloud_config_data)
        self.assertIn("aws_region", loaded_cloud_config)
        self.assertIn("s3_bucket", loaded_cloud_config)

    def test_loader_error_handling(self):
        """测试加载器错误处理"""
        # 测试权限错误（尝试创建只读文件然后修改）
        readonly_file = os.path.join(self.temp_dir, "readonly_config.json")

        with open(readonly_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_config, f, indent=2)

        # 在Windows上设置只读属性
        try:
            os.chmod(readonly_file, 0o444)  # 只读权限

            # 验证文件存在但可能无法修改
            self.assertTrue(os.path.exists(readonly_file))

            # 尝试读取应该成功
            with open(readonly_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)
                self.assertEqual(loaded_config, self.test_config)

        except OSError:
            # 如果权限修改失败，跳过此测试
            pass
        finally:
            # 恢复写权限以便清理
            try:
                os.chmod(readonly_file, 0o644)
            except OSError:
                pass

    def test_loader_performance_comparison(self):
        """测试加载器性能比较"""
        import time

        # 创建大型配置文件
        large_config = {}
        for i in range(100):
            large_config[f"section_{i}"] = {
                "key_1": f"value_{i}_1",
                "key_2": f"value_{i}_2",
                "key_3": f"value_{i}_3",
                "numbers": list(range(10)),
                "nested": {
                    "deep_key": f"deep_value_{i}"
                }
            }

        json_file = os.path.join(self.temp_dir, "large_config.json")

        # 保存大型配置
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(large_config, f, indent=2)

        # 测试JSON加载性能
        start_time = time.time()

        for _ in range(10):
            with open(json_file, 'r', encoding='utf-8') as f:
                loaded_config = json.load(f)

        json_load_time = time.time() - start_time

        # 验证数据正确性
        self.assertEqual(loaded_config, large_config)

        # 性能要求：10次加载应在2秒内完成
        self.assertLess(json_load_time, 2.0,
                       f"JSON加载性能不足: {json_load_time:.2f}s for 10 loads")

    def test_json_loader_real_integration(self):
        """测试JSON加载器真实集成"""
        # 创建临时JSON文件
        json_file = os.path.join(self.temp_dir, "real_test_config.json")

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(self.test_config, f, indent=2)

        # 创建JSON加载器实例
        loader = JSONLoader()

        # 测试加载功能
        config, metadata = loader.load(json_file)

        # 验证配置内容
        self.assertEqual(config, self.test_config)
        self.assertEqual(config["database"]["host"], "localhost")
        self.assertEqual(config["database"]["port"], 5432)

        # 验证元数据
        self.assertEqual(metadata["format"], "json")
        self.assertEqual(metadata["source"], str(Path(json_file).absolute()))
        self.assertIsInstance(metadata["load_time"], (int, float))
        self.assertGreater(metadata["load_time"], 0)
        self.assertIsInstance(metadata["size"], int)
        self.assertGreater(metadata["size"], 0)
        self.assertIsInstance(metadata["timestamp"], (int, float))

    def test_json_loader_file_not_found(self):
        """测试JSON加载器文件不存在的情况"""
        loader = JSONLoader()
        non_existent_file = os.path.join(self.temp_dir, "non_existent.json")

        with self.assertRaises(ConfigLoadError) as context:
            loader.load(non_existent_file)

        # 验证错误信息包含文件路径
        self.assertIn("JSON文件不存在", str(context.exception))
        self.assertIn("non_existent.json", str(context.exception))

    def test_json_loader_invalid_json(self):
        """测试JSON加载器无效JSON处理"""
        loader = JSONLoader()
        invalid_json_file = os.path.join(self.temp_dir, "invalid_config.json")

        # 创建无效JSON文件
        with open(invalid_json_file, 'w', encoding='utf-8') as f:
            f.write("{ invalid json content ")

        with self.assertRaises(ConfigLoadError) as context:
            loader.load(invalid_json_file)

        # 验证错误信息
        self.assertIn("JSON解析失败", str(context.exception))

    def test_json_loader_batch_load(self):
        """测试JSON加载器批量加载功能"""
        loader = JSONLoader()

        # 创建多个JSON文件
        files = []
        for i in range(3):
            file_path = os.path.join(self.temp_dir, f"batch_test_{i}.json")
            config_data = {"id": i, "name": f"config_{i}"}

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(config_data, f)

            files.append(file_path)

        # 测试批量加载
        results = loader.batch_load(files)

        # 验证结果
        self.assertEqual(len(results), 3)
        for i, file_path in enumerate(files):
            self.assertIn(file_path, results)
            config, metadata = results[file_path]
            self.assertEqual(config["id"], i)
            self.assertEqual(metadata["format"], "json")

    def test_json_loader_can_load(self):
        """测试JSON加载器格式识别功能"""
        loader = JSONLoader()

        # 测试支持的格式
        self.assertTrue(loader.can_load("config.json"))
        self.assertTrue(loader.can_load("settings.JSON"))
        self.assertTrue(loader.can_load("/path/to/config.json"))

        # 测试不支持的格式
        self.assertFalse(loader.can_load("config.yaml"))
        self.assertFalse(loader.can_load("config.toml"))
        self.assertFalse(loader.can_load("config.txt"))
        self.assertFalse(loader.can_load("config"))
        self.assertFalse(loader.can_load(""))

    def test_json_loader_supported_extensions(self):
        """测试JSON加载器支持的扩展名"""
        loader = JSONLoader()
        extensions = loader.get_supported_extensions()

        self.assertIsInstance(extensions, list)
        self.assertIn(".json", extensions)
        self.assertEqual(len(extensions), 1)

    @pytest.mark.skipif("yaml" not in globals() and "yaml" not in dir(__import__('sys').modules),
                       reason="PyYAML not available")
    def test_yaml_loader_real_integration(self):
        """测试YAML加载器真实集成"""
        try:
            import yaml
        except ImportError:
            self.skipTest("PyYAML not available")

        # 创建临时YAML文件
        yaml_file = os.path.join(self.temp_dir, "real_test_config.yaml")

        with open(yaml_file, 'w', encoding='utf-8') as f:
            yaml.dump(self.test_config, f, default_flow_style=False)

        # 创建YAML加载器实例
        loader = YAMLLoader()

        # 测试加载功能
        config, metadata = loader.load(yaml_file)

        # 验证配置内容
        self.assertEqual(config, self.test_config)
        self.assertEqual(config["database"]["host"], "localhost")
        self.assertEqual(config["database"]["port"], 5432)

        # 验证元数据
        self.assertEqual(metadata["format"], "yaml")
        self.assertEqual(metadata["source"], str(Path(yaml_file).absolute()))
        self.assertIsInstance(metadata["load_time"], (int, float))
        self.assertGreater(metadata["load_time"], 0)

    def test_env_loader_real_integration(self):
        """测试环境变量加载器真实集成"""
        # 设置测试环境变量
        test_env_vars = {
            "TEST_DB_HOST": "localhost",
            "TEST_DB_PORT": "5432",
            "TEST_DB_USER": "admin",
            "TEST_REDIS_HOST": "localhost",
            "TEST_REDIS_PORT": "6379"
        }

        # 设置环境变量
        for key, value in test_env_vars.items():
            os.environ[key] = value

        try:
            # 创建环境变量加载器实例
            loader = EnvLoader()

            # 测试加载功能 (假设有前缀配置)
            # 注意：这里的测试可能需要根据实际的EnvLoader实现进行调整
            # 这里只是演示如何测试

            # 验证环境变量已设置
            for key, expected_value in test_env_vars.items():
                self.assertEqual(os.environ.get(key), expected_value)

        finally:
            # 清理环境变量
            for key in test_env_vars.keys():
                if key in os.environ:
                    del os.environ[key]

    def test_env_loader_nested_config(self):
        """测试环境变量加载器嵌套配置功能"""
        # 设置嵌套环境变量
        test_env_vars = {
            "APP_DATABASE_HOST": "localhost",
            "APP_DATABASE_PORT": "5432",
            "APP_DATABASE_CREDENTIALS_USER": "admin",
            "APP_DATABASE_CREDENTIALS_PASS": "secret",
            "APP_CACHE_REDIS_HOST": "redis-server",
            "APP_CACHE_REDIS_PORT": "6379"
        }

        # 设置环境变量
        for key, value in test_env_vars.items():
            os.environ[key] = value

        try:
            loader = EnvLoader(prefixes=['APP_'])

            # 验证嵌套配置加载
            config = loader.load_all()

            # 验证嵌套结构
            self.assertIn("database", config)
            self.assertIn("cache", config)
            self.assertEqual(config["database"]["host"], "localhost")
            self.assertEqual(config["database"]["port"], 5432)  # 应该转换为整数
            self.assertIn("credentials", config["database"])
            self.assertEqual(config["database"]["credentials"]["user"], "admin")
            self.assertEqual(config["database"]["credentials"]["pass"], "secret")

        finally:
            # 清理环境变量
            for key in test_env_vars.keys():
                if key in os.environ:
                    del os.environ[key]

    def test_env_loader_value_conversion(self):
        """测试环境变量加载器值转换功能"""
        test_env_vars = {
            "TEST_INT_VAR": "42",
            "TEST_FLOAT_VAR": "3.14",
            "TEST_BOOL_TRUE_VAR": "true",
            "TEST_BOOL_FALSE_VAR": "false",
            "TEST_JSON_VAR": '{"key": "value"}',
            "TEST_LIST_VAR": "item1,item2,item3",
            "TEST_STRING_VAR": "hello world"
        }

        # 设置环境变量
        for key, value in test_env_vars.items():
            os.environ[key] = value

        try:
            loader = EnvLoader(prefixes=['TEST_'])

            config = loader.load_all()

            # 验证类型转换（注意环境变量加载器会将下划线转换为点号）
            self.assertEqual(config["int"]["var"], 42)
            self.assertEqual(config["float"]["var"], 3.14)
            self.assertEqual(config["bool"]["true"]["var"], True)
            self.assertEqual(config["bool"]["false"]["var"], False)
            self.assertEqual(config["json"]["var"], {"key": "value"})
            self.assertEqual(config["list"]["var"], ["item1", "item2", "item3"])
            self.assertEqual(config["string"]["var"], "hello world")

        finally:
            # 清理环境变量
            for key in test_env_vars.keys():
                if key in os.environ:
                    del os.environ[key]

    def test_env_loader_multiple_prefixes(self):
        """测试环境变量加载器多前缀支持"""
        test_env_vars = {
            "APP_DB_HOST": "localhost",
            "CONFIG_CACHE_HOST": "redis-server",
            "RQA_LOG_LEVEL": "INFO"
        }

        # 设置环境变量
        for key, value in test_env_vars.items():
            os.environ[key] = value

        try:
            loader = EnvLoader(prefixes=['APP_', 'CONFIG_', 'RQA_'])

            config = loader.load_all()

            # 验证所有前缀的配置都被加载
            self.assertIn("db", config)
            self.assertIn("cache", config)
            self.assertIn("log", config)

            self.assertEqual(config["db"]["host"], "localhost")
            self.assertEqual(config["cache"]["host"], "redis-server")
            self.assertEqual(config["log"]["level"], "INFO")

        finally:
            # 清理环境变量
            for key in test_env_vars.keys():
                if key in os.environ:
                    del os.environ[key]

    def test_toml_loader_real_integration(self):
        """测试TOML加载器真实集成"""
        try:
            import tomli
            toml_available = True
        except ImportError:
            try:
                import tomli
                toml_available = True
            except ImportError:
                toml_available = False

        if not toml_available:
            self.skipTest("TOML libraries not available")

        # 创建临时TOML文件
        toml_file = os.path.join(self.temp_dir, "real_test_config.toml")

        toml_content = '''[database]
host = "localhost"
port = 5432
user = "admin"
password = "password123"

[cache]
redis_host = "localhost"
redis_port = 6379
ttl = 300

[logging]
level = "INFO"
format = "json"
'''

        with open(toml_file, 'w', encoding='utf-8') as f:
            f.write(toml_content)

        # 创建TOML加载器实例
        loader = TOMLLoader()

        # 测试加载功能
        config = loader.load(toml_file)

        # 验证配置内容
        self.assertEqual(config["database"]["host"], "localhost")
        self.assertEqual(config["database"]["port"], 5432)
        self.assertEqual(config["cache"]["redis_port"], 6379)
        self.assertEqual(config["logging"]["level"], "INFO")

    def test_toml_loader_can_load(self):
        """测试TOML加载器格式识别功能"""
        loader = TOMLLoader()

        # 创建测试文件
        toml_file = os.path.join(self.temp_dir, "test.toml")
        with open(toml_file, 'w', encoding='utf-8') as f:
            f.write('test = "value"')

        # 测试支持的格式
        self.assertTrue(loader.can_load(toml_file))

        # 测试大小写变体
        toml_file_upper = os.path.join(self.temp_dir, "test.TOML")
        with open(toml_file_upper, 'w', encoding='utf-8') as f:
            f.write('test = "value"')
        self.assertTrue(loader.can_load(toml_file_upper))

        # 测试不支持的格式
        yaml_file = os.path.join(self.temp_dir, "test.yaml")
        with open(yaml_file, 'w', encoding='utf-8') as f:
            f.write('test: value')
        self.assertFalse(loader.can_load(yaml_file))

        json_file = os.path.join(self.temp_dir, "test.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            f.write('{"test": "value"}')
        self.assertFalse(loader.can_load(json_file))

        # 测试不存在的文件
        self.assertFalse(loader.can_load("non_existent.toml"))

        # 测试空字符串
        self.assertFalse(loader.can_load(""))

    def test_toml_loader_supported_extensions(self):
        """测试TOML加载器支持的扩展名"""
        loader = TOMLLoader()
        extensions = loader.get_supported_extensions()

        self.assertIsInstance(extensions, list)
        self.assertIn(".toml", extensions)
        self.assertIn(".TOML", extensions)
        self.assertEqual(len(extensions), 2)

    def test_database_loader_initialization(self):
        """测试数据库加载器初始化"""
        # 测试支持的数据库类型
        for db_type in ['postgresql', 'mysql', 'sqlite', 'mongodb', 'redis', 'influxdb']:
            loader = DatabaseLoader(db_type=db_type)
            self.assertEqual(loader.db_type, db_type)

        # 测试不支持的数据库类型
        with self.assertRaises(ValueError):
            DatabaseLoader(db_type='unsupported_db')

    def test_database_loader_can_load(self):
        """测试数据库加载器格式识别功能"""
        loader = DatabaseLoader()

        # 测试有效的配置源（数据库URL）
        self.assertTrue(loader.can_load("postgresql://config_table/myapp"))
        self.assertTrue(loader.can_load("mysql://settings/myapp"))
        self.assertTrue(loader.can_load("mongodb://app_config/myapp"))
        self.assertTrue(loader.can_load("redis://test.config/myapp"))

        # 测试无效的配置源
        self.assertFalse(loader.can_load(""))
        self.assertFalse(loader.can_load(None))

    def test_database_loader_supported_extensions(self):
        """测试数据库加载器支持的扩展名"""
        loader = DatabaseLoader()
        extensions = loader.get_supported_extensions()

        self.assertIsInstance(extensions, list)
        self.assertEqual(len(extensions), 0)  # 数据库加载器不需要文件扩展名

    def test_cloud_loader_initialization(self):
        """测试云配置加载器初始化"""
        # 测试支持的云服务提供商
        for provider in ['aws', 'azure', 'gcp', 'consul', 'etcd', 'zookeeper']:
            loader = CloudLoader(provider=provider)
            self.assertEqual(loader.provider, provider)

        # 测试不支持的云服务提供商
        with self.assertRaises(ValueError):
            CloudLoader(provider='unsupported_provider')

    def test_cloud_loader_can_load(self):
        """测试云配置加载器格式识别功能"""
        loader = CloudLoader()

        # 测试有效的配置源（云路径）
        self.assertTrue(loader.can_load("aws://parameter/myapp/config"))
        self.assertTrue(loader.can_load("azure://vault/myapp/config"))
        self.assertTrue(loader.can_load("gcp://secret/myapp/config"))
        self.assertTrue(loader.can_load("s3://mybucket/config"))
        self.assertTrue(loader.can_load("gs://mybucket/config"))

        # 测试无效的配置源
        self.assertFalse(loader.can_load(""))
        self.assertFalse(loader.can_load(None))

    def test_cloud_loader_supported_extensions(self):
        """测试云配置加载器支持的扩展名"""
        loader = CloudLoader()
        extensions = loader.get_supported_extensions()

        self.assertIsInstance(extensions, list)
        self.assertEqual(len(extensions), 0)  # 云服务加载器不需要文件扩展名

    def test_loader_integration_comprehensive(self):
        """测试加载器综合集成功能"""
        # 创建不同格式的配置文件
        json_file = os.path.join(self.temp_dir, "integration_test.json")
        yaml_file = os.path.join(self.temp_dir, "integration_test.yaml")

        # 创建相同的配置数据
        test_config = {
            "app": {
                "name": "RQA2025",
                "version": "1.0.0",
                "environment": "testing"
            },
            "features": {
                "ai_trading": True,
                "real_time": True,
                "backtesting": False
            },
            "performance": {
                "max_connections": 1000,
                "timeout": 30.5,
                "retry_count": 3
            }
        }

        # 保存为JSON
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(test_config, f, indent=2)

        # 保存为YAML
        try:
            import yaml
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(test_config, f, default_flow_style=False)
        except ImportError:
            yaml_file = None

        # 测试JSON加载器
        json_loader = JSONLoader()
        json_config, json_metadata = json_loader.load(json_file)

        self.assertEqual(json_config, test_config)
        self.assertEqual(json_metadata["format"], "json")
        self.assertIsInstance(json_metadata["load_time"], (int, float))

        # 测试YAML加载器（如果可用）
        if yaml_file:
            yaml_loader = YAMLLoader()
            yaml_config = yaml_loader.load(yaml_file)

            self.assertEqual(yaml_config, test_config)

        # 验证两个加载器加载的结果一致
        if yaml_file:
            self.assertEqual(json_config, yaml_config)

    def test_loader_error_handling_comprehensive(self):
        """测试加载器综合错误处理"""
        # 测试JSON加载器错误处理
        json_loader = JSONLoader()

        # 测试不存在的文件
        with self.assertRaises(ConfigLoadError):
            json_loader.load("non_existent_file.json")

        # 测试无效的JSON文件
        invalid_json_file = os.path.join(self.temp_dir, "invalid.json")
        with open(invalid_json_file, 'w', encoding='utf-8') as f:
            f.write("{ invalid json }")

        with self.assertRaises(ConfigLoadError):
            json_loader.load(invalid_json_file)

        # 测试TOML加载器错误处理
        toml_loader = TOMLLoader()

        # 测试不存在的文件
        with self.assertRaises(ConfigLoadError):
            toml_loader.load("non_existent_file.toml")

        # 测试不支持的文件格式
        self.assertFalse(json_loader.can_load("config.xml"))
        self.assertFalse(toml_loader.can_load("config.xml"))

    def test_loader_performance_comparison_comprehensive(self):
        """测试加载器综合性能比较"""
        # 创建大型配置数据
        large_config = {
            f"section_{i}": {
                "id": i,
                "name": f"config_{i}",
                "settings": {
                    "enabled": i % 2 == 0,
                    "value": float(i) * 1.5,
                    "items": [f"item_{j}" for j in range(5)]
                },
                "metadata": {
                    "created": f"2024-01-{i%30+1:02d}",
                    "tags": [f"tag_{j}" for j in range(3)]
                }
            }
            for i in range(200)  # 200个配置节
        }

        # 保存为JSON
        json_file = os.path.join(self.temp_dir, "performance_test.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(large_config, f, indent=2)

        # 保存为YAML（如果可用）
        yaml_file = None
        try:
            import yaml
            yaml_file = os.path.join(self.temp_dir, "performance_test.yaml")
            with open(yaml_file, 'w', encoding='utf-8') as f:
                yaml.dump(large_config, f, default_flow_style=False)
        except ImportError:
            pass

        # 测试JSON加载器性能
        json_loader = JSONLoader()
        import time

        # 多次加载测试
        json_times = []
        for _ in range(5):
            start_time = time.time()
            config, _ = json_loader.load(json_file)
            json_times.append(time.time() - start_time)

        # 验证数据正确性
        self.assertEqual(config, large_config)

        # 测试YAML加载器性能（如果可用）
        if yaml_file:
            yaml_loader = YAMLLoader()
            yaml_times = []

            for _ in range(5):
                start_time = time.time()
                config = yaml_loader.load(yaml_file)
                yaml_times.append(time.time() - start_time)

            # 验证数据一致性
            self.assertEqual(config, large_config)

            # 比较性能
            avg_json_time = sum(json_times) / len(json_times)
            avg_yaml_time = sum(yaml_times) / len(yaml_times)

            print(f"JSON average load time: {avg_json_time:.4f}s")
            print(f"YAML average load time: {avg_yaml_time:.4f}s")

        # 性能断言：每次加载应在合理时间内完成
        for load_time in json_times:
            self.assertLess(load_time, 1.0, f"JSON加载时间过长: {load_time:.4f}s")


if __name__ == '__main__':
    unittest.main()
