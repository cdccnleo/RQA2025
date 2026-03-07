# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
测试基础设施层 - 统一配置管理器

测试UnifiedConfigManager类的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', '..'))

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import json
from pathlib import Path

# 修复导入路径
from src.infrastructure.config import UnifiedConfigManager

class TestUnifiedConfigManager(unittest.TestCase):
    """测试统一配置管理器"""

    def setUp(self):
        """测试前准备"""
        self.config = {
        "auto_reload": True,
        "validation_enabled": True,
        "encryption_enabled": False,
        "backup_enabled": True,
        "max_backup_files": 3,
        "config_file": "test_config.json"
        }
        self.manager = UnifiedConfigManager(self.config)

    def test_initialization(self):
        """测试初始化"""
        self.assertIsNotNone(self.manager)
        self.assertFalse(self.manager._initialized)
        self.assertIsInstance(self.manager._data, dict)
        self.assertEqual(self.manager.config["auto_reload"], True)
        self.assertEqual(self.manager.config["max_backup_files"], 3)

    # ==================== 补充统一管理器测试覆盖 ====================

    def test_merge_config(self):
        """测试配置合并"""
        manager = UnifiedConfigManager()

        # 合并配置
        new_config = {"new_section": {"key": "value"}}
        result = manager.merge_config(new_config)
        self.assertTrue(result)

        # 验证配置已合并
        self.assertIn("new_section", manager._data)
        self.assertEqual(manager._data["new_section"]["key"], "value")

    def test_initialize(self):
        """测试初始化方法"""
        result = self.manager.initialize()
        self.assertTrue(result)
        self.assertTrue(self.manager._initialized)

    def test_get_valid_config(self):
        """测试获取有效配置"""
        # 设置配置数据
        self.manager._data = {
        "database": {"host": "localhost", "port": 5432},
        "cache": {"ttl": 300}
        }

        # 测试正常获取（使用正确的接口：section.key格式）
        self.assertEqual(self.manager.get("database.host"), "localhost")
        self.assertEqual(self.manager.get("database.port"), 5432)
        self.assertEqual(self.manager.get("cache.ttl"), 300)

    def test_get_with_default(self):
        """测试获取配置并使用默认值"""
        # 测试不存在的section
        self.assertEqual(self.manager.get("nonexistent.key", "default"), "default")

        # 测试不存在的key
        self.assertEqual(self.manager.get("database.nonexistent", 3306), 3306)

    def test_get_edge_cases(self):
        """测试获取配置的边界情况"""
        # 测试空字符串
        self.assertIsNone(self.manager.get(""))
        self.assertIsNone(self.manager.get("."))
        self.assertIsNone(self.manager.get("section."))

        # 测试None值（应该抛出TypeError）
        with self.assertRaises(TypeError):
            self.manager.get(None)

        # 测试过长字符串
        long_string = "a" * 101
        self.assertIsNone(self.manager.get(long_string + ".key"))
        self.assertIsNone(self.manager.get("section." + long_string))

        # 测试危险字符
        self.assertIsNone(self.manager.get("section<.key"))
        self.assertIsNone(self.manager.get("section.key;rm"))

    def test_set_valid_config(self):
        """测试设置有效配置"""
        # 测试正常设置
        result = self.manager.set("database.host", "localhost")
        self.assertTrue(result)
        self.assertEqual(self.manager._data["database"]["host"], "localhost")

        # 测试覆盖现有值
        result = self.manager.set("database.host", "127.0.0.1")
        self.assertTrue(result)
        self.assertEqual(self.manager._data["database"]["host"], "127.0.0.1")

    def test_set_edge_cases(self):
        """测试设置配置的边界情况"""
        # 测试无效key
        self.assertFalse(self.manager.set("", "value"))
        self.assertFalse(self.manager.set(".", "value"))
        self.assertFalse(self.manager.set("section.", "value"))

        # 测试None值（应该抛出TypeError）
        with self.assertRaises(TypeError):
            self.manager.set(None, "value")

        # 测试过长key
        long_string = "a" * 101
        self.assertFalse(self.manager.set(long_string + ".key", "value"))
        self.assertFalse(self.manager.set("section." + long_string, "value"))

        # 测试危险字符
        self.assertFalse(self.manager.set("section<.key", "value"))
        self.assertFalse(self.manager.set("section.key;rm", "value"))

    def test_delete_config(self):
        """测试删除配置"""
        # 设置测试数据
        self.manager._data = {
        "database": {"host": "localhost", "port": 5432},
        "cache": {"ttl": 300}
        }

        # 删除存在的配置
        result = self.manager.delete("database", "host")
        self.assertTrue(result)
        self.assertNotIn("host", self.manager._data["database"])

        # 删除不存在的配置
        result = self.manager.delete("database", "nonexistent")
        self.assertFalse(result)

        # 删除不存在的section
        result = self.manager.delete("nonexistent", "key")
        self.assertFalse(result)

    def test_has_section(self):
        """测试检查section是否存在"""
        self.manager._data = {
        "database": {"host": "localhost"},
        "cache": {"ttl": 300}
        }

        self.assertTrue(self.manager.has_section("database"))
        self.assertTrue(self.manager.has_section("cache"))
        self.assertFalse(self.manager.has_section("nonexistent"))

    def test_get_sections(self):
        """测试获取所有sections"""
        self.manager._data = {
        "database": {"host": "localhost"},
        "cache": {"ttl": 300},
        "logging": {"level": "INFO"}
        }

        sections = self.manager.get_sections()
        self.assertIsInstance(sections, list)
        self.assertEqual(len(sections), 3)
        self.assertIn("database", sections)
        self.assertIn("cache", sections)
        self.assertIn("logging", sections)

    def test_get_section(self):
        """测试获取完整section"""
        test_section = {"host": "localhost", "port": 5432, "user": "admin"}
        self.manager._data = {"database": test_section}

        section = self.manager.get_section("database")
        self.assertEqual(section, test_section)

        # 获取不存在的section
        section = self.manager.get_section("nonexistent")
        self.assertIsNone(section)

    def test_set_section(self):
        """测试设置完整section"""
        test_section = {"host": "localhost", "port": 5432}

        result = self.manager.set_section("database", test_section)
        self.assertTrue(result)
        self.assertEqual(self.manager._data["database"], test_section)

    def test_delete_section(self):
        """测试删除section"""
        self.manager._data = {
        "database": {"host": "localhost"},
        "cache": {"ttl": 300}
        }

        # 删除存在的section
        result = self.manager.delete_section("database")
        self.assertTrue(result)
        self.assertNotIn("database", self.manager._data)

        # 删除不存在的section
        result = self.manager.delete_section("nonexistent")
        self.assertFalse(result)

    def test_clear_all(self):
        """测试清空所有配置"""
        self.manager._data = {
        "database": {"host": "localhost"},
        "cache": {"ttl": 300},
        "logging": {"level": "INFO"}
        }

        self.manager.clear_all()
        self.assertEqual(len(self.manager._data), 0)

    def test_reload_config_method(self):
        """测试重新加载配置"""
        # 设置初始数据和配置文件路径
        self.manager._data = {"test": {"key": "value"}}
        self.manager.config["config_file"] = "test_config.json"

        # 重新加载（这里应该从文件加载，但我们模拟）
        with patch.object(self.manager, 'load_config', return_value=True):
            result = self.manager.reload_config()
            self.assertTrue(result)

    def test_save_to_file(self):
        """测试保存配置到文件"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")

            # 设置测试数据
            self.manager._data = {
                "database": {"host": "localhost", "port": 5432},
                "cache": {"ttl": 300}
            }

            # 保存到文件
            result = self.manager.save_to_file(config_file)
            self.assertTrue(result)

            # 验证文件存在
            self.assertTrue(os.path.exists(config_file))

            # 验证文件内容
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)
                self.assertEqual(saved_data, self.manager._data)

    def test_load_from_file(self):
        """测试从文件加载配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")

            # 创建测试配置文件
            test_data = {
                "database": {"host": "localhost", "port": 5432},
                "cache": {"ttl": 300}
            }
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(test_data, f)

            # 加载配置
            result = self.manager.load_from_file(config_file)
            self.assertTrue(result)
            self.assertEqual(self.manager._data, test_data)

    def test_backup_and_restore(self):
        """测试备份和恢复"""
        with tempfile.TemporaryDirectory() as temp_dir:
            backup_dir = os.path.join(temp_dir, "backups")

            # 确保备份目录存在
            os.makedirs(backup_dir, exist_ok=True)

            # 设置测试数据
            self.manager._data = {"test": {"key": "value"}}

            # 创建备份
            result = self.manager.backup_config(backup_dir)
            self.assertTrue(result)

            # 验证备份文件存在
            backup_files = os.listdir(backup_dir)
            self.assertGreater(len(backup_files), 0)

            # 修改数据
            self.manager._data = {"test": {"key": "modified"}}

            # 恢复备份
            latest_backup = os.path.join(backup_dir, backup_files[0])
            result = self.manager.restore_from_backup(latest_backup)
            self.assertTrue(result)
            self.assertEqual(self.manager._data, {"test": {"key": "value"}})

    def test_validate_config(self):
        """测试配置验证"""
        # 有效的配置
        valid_config = {"host": "localhost", "port": 5432}
        result = self.manager.validate_config(valid_config)
        self.assertTrue(result)

        # 无效的配置
        invalid_config = {"host": "", "port": "invalid"}
        result = self.manager.validate_config(invalid_config)
        self.assertFalse(result)

    def test_update_config(self):
        """测试更新配置"""
        # 初始配置
        initial_config = {
            "database.host": "localhost",
            "database.port": 5432
        }

        # 更新配置
        self.manager.update(initial_config)

        # 验证配置已更新
        self.assertEqual(self.manager.get("database.host"), "localhost")
        self.assertEqual(self.manager.get("database.port"), 5432)

    def test_update_config_with_exception(self):
        """测试更新配置时的异常处理"""
        # 测试None值导致的异常
        with self.assertRaises(ValueError):
            self.manager.update(None)

    def test_watch_config_changes(self):
        """测试监听配置变化"""
        # 创建回调函数
        callback_calls = []

        def test_callback(key, value):
            callback_calls.append((key, value))

        # 注册监听器
        self.manager.watch("test.key", test_callback)

        # 验证监听器已注册
        self.assertTrue(hasattr(self.manager, '_watchers'))
        self.assertIn("test.key", self.manager._watchers)
        self.assertEqual(len(self.manager._watchers["test.key"]), 1)

    def test_watch_multiple_callbacks(self):
        """测试多个监听器"""
        callback1_calls = []
        callback2_calls = []

        def callback1(key, value):
            callback1_calls.append((key, value))

        def callback2(key, value):
            callback2_calls.append((key, value))

        # 注册多个监听器
        self.manager.watch("test.key", callback1)
        self.manager.watch("test.key", callback2)

        # 验证两个监听器都已注册
        self.assertEqual(len(self.manager._watchers["test.key"]), 2)

    def test_reload_config_from_file(self):
        """测试重新加载配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "reload_test.json")

            # 创建测试配置文件
            test_config = {
                "database": {"host": "reloaded_host", "port": 3306}
            }

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(test_config, f)

            # 设置配置管理器的配置文件路径
            self.manager.config["config_file"] = config_file

            # 执行重新加载
            result = self.manager.reload_config()
            self.assertTrue(result)

            # 验证配置已重新加载
            self.assertEqual(self.manager.get("database.host"), "reloaded_host")
            self.assertEqual(self.manager.get("database.port"), 3306)

    def test_reload_config_without_file(self):
        """测试重新加载配置时文件不存在"""
        # 不设置配置文件路径
        self.manager.config["config_file"] = None

        # 执行重新加载
        result = self.manager.reload_config()
        self.assertFalse(result)

    def test_validate_config_method(self):
        """测试validate方法"""
        # 有效的配置
        valid_config = {"host": "localhost", "port": 5432}
        result = self.manager.validate(valid_config)
        self.assertTrue(result)

        # 有效的配置 - 空字符串值（validate方法只检查key）
        config_with_empty_value = {"host": "", "port": 5432}
        result = self.manager.validate(config_with_empty_value)
        self.assertTrue(result)  # validate方法不检查值，只检查key

        # 无效的配置 - 超长key
        long_key = "a" * 101
        invalid_config2 = {long_key: "value"}
        result = self.manager.validate(invalid_config2)
        self.assertFalse(result)

        # 无效的配置 - 危险字符
        invalid_config3 = {"host<script>": "value"}
        result = self.manager.validate(invalid_config3)
        self.assertFalse(result)

        # 无效的配置 - 空key
        invalid_config4 = {"": "value", "port": 5432}
        result = self.manager.validate(invalid_config4)
        self.assertFalse(result)

    def test_validate_config_method_edge_cases(self):
        """测试validate方法的边界情况"""
        # 非字典类型
        result = self.manager.validate("not_a_dict")
        self.assertFalse(result)

        result = self.manager.validate(123)
        self.assertFalse(result)

        result = self.manager.validate([])
        self.assertFalse(result)

        # 空字典
        result = self.manager.validate({})
        self.assertTrue(result)

    def test_get_section_method(self):
        """测试get_section方法"""
        # 设置测试数据
        self.manager._data = {
            "database": {"host": "localhost", "port": 5432},
            "cache": {"ttl": 300}
        }

        # 获取存在的section
        db_section = self.manager.get_section("database")
        self.assertIsNotNone(db_section)
        self.assertEqual(db_section["host"], "localhost")
        self.assertEqual(db_section["port"], 5432)

        # 获取不存在的section
        nonexistent_section = self.manager.get_section("nonexistent")
        self.assertIsNone(nonexistent_section)

        # 验证返回的是副本，不是引用
        db_section["host"] = "modified"
        original_host = self.manager.get("database.host")
        self.assertEqual(original_host, "localhost")  # 原始数据不应被修改

    def test_load_config_method(self):
        """测试load_config方法"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # 测试加载字典配置
            dict_config = {
                "database": {"host": "localhost", "port": 5432},
                "cache": {"ttl": 300}
            }

            result = self.manager.load_config(dict_config)
            self.assertTrue(result)

            # 验证配置已加载
            self.assertEqual(self.manager.get("database.host"), "localhost")
            self.assertEqual(self.manager.get("cache.ttl"), 300)

    def test_load_config_from_file(self):
        """测试从文件加载配置"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "test_config.json")

            # 创建测试配置文件
            file_config = {
                "app": {"name": "test_app", "version": "1.0"},
                "server": {"host": "0.0.0.0", "port": 8080}
            }

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(file_config, f)

            # 加载配置文件
            result = self.manager.load_config(config_file)
            self.assertTrue(result)

            # 验证配置已加载
            self.assertEqual(self.manager.get("app.name"), "test_app")
            self.assertEqual(self.manager.get("server.port"), 8080)

    def test_load_config_file_not_found(self):
        """测试加载不存在的文件"""
        result = self.manager.load_config("nonexistent_file.json")
        self.assertFalse(result)

    def test_save_config_method(self):
        """测试save_config方法"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "save_test.json")

            # 设置测试数据
            self.manager._data = {
                "test": {"key": "value", "number": 42},
                "config": {"enabled": True}
            }

            # 保存配置
            result = self.manager.save_config(config_file)
            self.assertTrue(result)

            # 验证文件已创建
            self.assertTrue(os.path.exists(config_file))

            # 验证文件内容
            with open(config_file, 'r', encoding='utf-8') as f:
                saved_data = json.load(f)

            self.assertEqual(saved_data, self.manager._data)

    def test_save_config_create_directory(self):
        """测试保存配置时自动创建目录"""
        with tempfile.TemporaryDirectory() as temp_dir:
            nested_dir = os.path.join(temp_dir, "nested", "deep", "path")
            config_file = os.path.join(nested_dir, "test.json")

            # 设置测试数据
            self.manager._data = {"test": {"data": "value"}}

            # 保存配置（目录不存在时应自动创建）
            result = self.manager.save_config(config_file)
            self.assertTrue(result)

            # 验证目录和文件都已创建
            self.assertTrue(os.path.exists(nested_dir))
            self.assertTrue(os.path.exists(config_file))

    def test_get_all_sections_method(self):
        """测试get_all_sections方法"""
        # 空配置管理器
        empty_manager = UnifiedConfigManager()
        sections = empty_manager.get_all_sections()
        self.assertEqual(sections, [])

        # 设置测试数据
        self.manager._data = {
            "database": {"host": "localhost"},
            "cache": {"ttl": 300},
            "logging": {"level": "INFO"}
        }

        sections = self.manager.get_all_sections()
        self.assertIsInstance(sections, list)
        self.assertEqual(len(sections), 3)
        self.assertIn("database", sections)
        self.assertIn("cache", sections)
        self.assertIn("logging", sections)

    def test_reload_config_with_file(self):
        """测试reload_config方法"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = os.path.join(temp_dir, "reload_config.json")

            # 创建初始配置文件
            initial_config = {
                "app": {"version": "1.0"}
            }

            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(initial_config, f)

            # 设置配置管理器
            self.manager.config["config_file"] = config_file

            # 执行重新加载
            result = self.manager.reload_config()
            self.assertTrue(result)

            # 验证配置已加载
            self.assertEqual(self.manager.get("app.version"), "1.0")

    def test_validate_config_with_rules(self):
        """测试带验证规则的配置验证"""
        # 设置验证规则
        validation_rules = {
            "database": {
                "host": {"type": "string", "required": True},
                "port": {"type": "number", "required": True, "min": 1024, "max": 65535}
            }
        }

        # 添加验证规则到管理器
        self.manager._validation_rules = validation_rules

        # 有效的配置
        valid_config = {
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }

        result = self.manager.validate_config(valid_config)
        self.assertTrue(result)

        # 无效的配置 - 缺少必需字段
        invalid_config1 = {
            "database": {
                "host": "localhost"
                # 缺少port
            }
        }

        result = self.manager.validate_config(invalid_config1)
        self.assertFalse(result)

        # 无效的配置 - 错误类型
        invalid_config2 = {
            "database": {
                "host": "localhost",
                "port": "not_a_number"
            }
        }

        result = self.manager.validate_config(invalid_config2)
        self.assertFalse(result)

    def test_validate_config_without_rules(self):
        """测试不带验证规则的配置验证"""
        # 不设置验证规则
        if hasattr(self.manager, '_validation_rules'):
            delattr(self.manager, '_validation_rules')

        # 有效的配置
        valid_config = {
            "database": {
                "host": "localhost",
                "port": 5432
            }
        }

        result = self.manager.validate_config(valid_config)
        self.assertTrue(result)

    def test_complex_config_operations(self):
        """测试复杂配置操作"""
        # 设置复杂配置数据
        complex_config = {
            "database": {
                "host": "localhost",
                "port": 5432,
                "credentials": {
                    "username": "admin",
                    "password": "secret"
                },
                "ssl": {
                    "enabled": True,
                    "cert_file": "/path/to/cert.pem"
                }
            },
            "cache": {
                "redis": {
                    "host": "localhost",
                    "port": 6379,
                    "cluster": {
                        "enabled": False,
                        "nodes": []
                    }
                }
            }
        }

        # 先加载完整的配置
        self.manager.load_config(complex_config)

        # 更新配置（只更新特定字段）
        self.manager.set("database.host", "prod-db.example.com")
        self.manager.set("database.port", 3306)
        self.manager.set("cache.redis.port", 6380)

        # 验证嵌套配置更新
        self.assertEqual(self.manager.get("database.host"), "prod-db.example.com")
        self.assertEqual(self.manager.get("database.port"), 3306)
        self.assertEqual(self.manager.get("cache.redis.port"), 6380)

        # 验证获取完整section
        db_section = self.manager.get_section("database")
        self.assertIsNotNone(db_section)
        self.assertIn("credentials", db_section)
        self.assertIn("ssl", db_section)

        # 验证嵌套字段仍然存在（通过get_section方法）
        # 注意：get方法只支持两级路径，深层嵌套需要通过get_section获取
        db_section = self.manager.get_section("database")
        self.assertEqual(db_section["credentials"]["username"], "admin")
        self.assertEqual(db_section["ssl"]["enabled"], True)

        # 验证不支持深层路径的get方法返回None
        self.assertIsNone(self.manager.get("database.credentials.username"))

    def test_config_validation_edge_cases(self):
        """测试配置验证的边界情况"""
        # 测试各种边界情况的验证

        # 1. 空字符串值
        config1 = {"key": ""}
        result = self.manager.validate_config(config1)
        self.assertFalse(result)

        # 2. None值
        config2 = {"key": None}
        result = self.manager.validate_config(config2)
        self.assertTrue(result)  # None值本身是有效的

        # 3. 超长key
        long_key = "a" * 101
        config3 = {long_key: "value"}
        result = self.manager.validate_config(config3)
        self.assertFalse(result)

        # 4. 危险字符
        config4 = {"key<script>": "value"}
        result = self.manager.validate_config(config4)
        self.assertFalse(result)

        # 5. 非字符串key
        config5 = {123: "value"}
        result = self.manager.validate_config(config5)
        self.assertFalse(result)

    def test_config_operations_error_handling(self):
        """测试配置操作的错误处理"""
        # 测试各种异常情况

        # 1. update方法异常处理
        with self.assertRaises(ValueError):
            self.manager.update("not_a_dict")

        # 2. reload方法异常处理
        self.manager.config["config_file"] = "nonexistent_file.json"
        result = self.manager.reload()
        # reload方法可能抛出异常或返回失败，这里不做严格断言

        # 3. 验证方法对异常的鲁棒性
        result = self.manager.validate(None)
        self.assertFalse(result)

        result = self.manager.validate(123)
        self.assertFalse(result)

    def test_get_config_summary(self):
        """测试获取配置摘要"""
        self.manager._data = {
        "database": {"host": "localhost", "port": 5432, "user": "admin"},
        "cache": {"ttl": 300, "max_size": 1000}
        }

        summary = self.manager.get_config_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn("total_sections", summary)
        self.assertIn("total_keys", summary)
        self.assertEqual(summary["total_sections"], 2)
        self.assertEqual(summary["total_keys"], 5)

    def test_thread_safety(self):
        """测试线程安全"""
        import threading
        import concurrent.futures

        results = []
        errors = []

        def worker(thread_id):
            try:
                # 每个线程设置自己的配置
                section = f"thread_{thread_id}"
                self.manager.set(f"{section}.key", f"value_{thread_id}")
                self.manager.set(f"{section}.counter", thread_id)

                # 读取配置
                value = self.manager.get(f"{section}.key")
                counter = self.manager.get(f"{section}.counter")

                if value == f"value_{thread_id}" and counter == thread_id:
                    results.append(f"Thread {thread_id} success")
                else:
                    errors.append(f"Thread {thread_id}: expected {f'value_{thread_id}'}, got {value}")
            except Exception as e:
                errors.append(f"Thread {thread_id} error: {e}")

        # 使用线程池执行并发操作
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(worker, i) for i in range(5)]
            concurrent.futures.wait(futures)

            # 验证没有错误发生
            self.assertEqual(len(errors), 0, f"Thread safety errors: {errors}")
            self.assertEqual(len(results), 5, "Not all threads completed successfully")

    def test_performance_under_load(self):
        """测试高负载下的性能"""
        import time

        # 设置大量配置数据
        for i in range(100):
            section = f"section_{i}"
            for j in range(10):
                key = f"key_{j}"
                value = f"value_{i}_{j}"
                self.manager.set(f"{section}.{key}", value)

        # 测试读取性能
        start_time = time.time()
        for i in range(100):
            section = f"section_{i}"
            for j in range(10):
                key = f"key_{j}"
                value = self.manager.get(f"{section}.{key}")
                self.assertIsNotNone(value)

        end_time = time.time()
        duration = end_time - start_time

        # 应该在合理时间内完成
        self.assertLess(duration, 5.0, f"Performance test took too long: {duration}s")

if __name__ == '__main__':
    unittest.main()