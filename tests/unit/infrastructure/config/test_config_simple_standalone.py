#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置系统简化独立测试

测试基本的配置功能，不依赖复杂的导入链
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import os
import json
import tempfile
from typing import Dict, Any, Optional


# 简化的配置异常类
class ConfigError(Exception):
    """配置异常"""
    pass


class ConfigKeyError(ConfigError):
    """配置键错误"""
    pass


class ConfigValueError(ConfigError):
    """配置值错误"""
    pass


# 简化的配置管理器
class SimpleConfigManager:
    """简化的配置管理器"""

    def __init__(self):
        self._config: Dict[str, Any] = {}
        self._file_path: Optional[str] = None

    def set(self, key: str, value: Any) -> bool:
        """设置配置项"""
        if not isinstance(key, str) or not key.strip():
            raise ConfigKeyError(f"Invalid key: {key}")

        self._config[key] = value
        return True

    def get(self, key: str, default: Any = None) -> Any:
        """获取配置项"""
        return self._config.get(key, default)

    def delete(self, key: str) -> bool:
        """删除配置项"""
        if key in self._config:
            del self._config[key]
            return True
        return False

    def exists(self, key: str) -> bool:
        """检查配置项是否存在"""
        return key in self._config

    def keys(self) -> list:
        """获取所有配置键"""
        return list(self._config.keys())

    def clear(self) -> bool:
        """清空所有配置"""
        self._config.clear()
        return True

    def save_to_file(self, file_path: str) -> bool:
        """保存配置到文件"""
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
            self._file_path = file_path
            return True
        except Exception as e:
            raise ConfigError(f"Failed to save config: {e}")

    def load_from_file(self, file_path: str) -> bool:
        """从文件加载配置"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self._config = json.load(f)
            self._file_path = file_path
            return True
        except Exception as e:
            raise ConfigError(f"Failed to load config: {e}")

    def get_all(self) -> Dict[str, Any]:
        """获取所有配置"""
        return self._config.copy()


class TestSimpleConfigManager:
    """测试简化配置管理器"""

    def setup_method(self, method):
        """测试前准备"""
        self.manager = SimpleConfigManager()

    def teardown_method(self, method):
        """测试后清理"""
        # 清理临时文件
        if hasattr(self, 'temp_file') and os.path.exists(self.temp_file):
            os.unlink(self.temp_file)

    def test_initialization(self):
        """测试初始化"""
        assert self.manager._config == {}
        assert self.manager._file_path is None

    def test_set_and_get_basic(self):
        """测试基本的设置和获取"""
        # 设置配置
        result = self.manager.set("test_key", "test_value")
        assert result is True

        # 获取配置
        value = self.manager.get("test_key")
        assert value == "test_value"

    def test_set_and_get_different_types(self):
        """测试不同类型的配置值"""
        # 字符串
        self.manager.set("str_key", "string_value")
        assert self.manager.get("str_key") == "string_value"

        # 整数
        self.manager.set("int_key", 42)
        assert self.manager.get("int_key") == 42

        # 浮点数
        self.manager.set("float_key", 3.14)
        assert self.manager.get("float_key") == 3.14

        # 布尔值
        self.manager.set("bool_key", True)
        assert self.manager.get("bool_key") is True

        # 列表
        self.manager.set("list_key", [1, 2, 3])
        assert self.manager.get("list_key") == [1, 2, 3]

        # 字典
        self.manager.set("dict_key", {"nested": "value"})
        assert self.manager.get("dict_key") == {"nested": "value"}

    def test_get_with_default(self):
        """测试带默认值的获取"""
        # 不存在的键应该返回默认值
        assert self.manager.get("nonexistent", "default") == "default"
        assert self.manager.get("nonexistent", None) is None
        assert self.manager.get("nonexistent", 0) == 0

    def test_delete_existing_key(self):
        """测试删除存在的键"""
        self.manager.set("delete_key", "delete_value")
        assert self.manager.exists("delete_key")

        result = self.manager.delete("delete_key")
        assert result is True
        assert not self.manager.exists("delete_key")
        assert self.manager.get("delete_key") is None

    def test_delete_nonexistent_key(self):
        """测试删除不存在的键"""
        result = self.manager.delete("nonexistent")
        assert result is False

    def test_exists_method(self):
        """测试存在性检查"""
        assert not self.manager.exists("test_key")

        self.manager.set("test_key", "test_value")
        assert self.manager.exists("test_key")

        self.manager.delete("test_key")
        assert not self.manager.exists("test_key")

    def test_keys_method(self):
        """测试获取所有键"""
        assert self.manager.keys() == []

        self.manager.set("key1", "value1")
        self.manager.set("key2", "value2")
        self.manager.set("key3", "value3")

        keys = self.manager.keys()
        assert len(keys) == 3
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" in keys

    def test_clear_method(self):
        """测试清空所有配置"""
        self.manager.set("key1", "value1")
        self.manager.set("key2", "value2")
        assert len(self.manager.keys()) == 2

        result = self.manager.clear()
        assert result is True
        assert len(self.manager.keys()) == 0
        assert self.manager.get("key1") is None
        assert self.manager.get("key2") is None

    def test_get_all_method(self):
        """测试获取所有配置"""
        assert self.manager.get_all() == {}

        self.manager.set("key1", "value1")
        self.manager.set("key2", "value2")

        all_config = self.manager.get_all()
        expected = {"key1": "value1", "key2": "value2"}
        assert all_config == expected

        # 验证返回的是副本
        all_config["key3"] = "value3"
        assert "key3" not in self.manager.get_all()

    def test_save_and_load_file(self):
        """测试文件保存和加载"""
        # 设置一些配置
        self.manager.set("file_key1", "file_value1")
        self.manager.set("file_key2", {"nested": "value"})
        self.manager.set("file_key3", [1, 2, 3])

        # 保存到临时文件
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            self.temp_file = f.name

        result = self.manager.save_to_file(self.temp_file)
        assert result is True
        assert self.manager._file_path == self.temp_file

        # 创建新的管理器并加载
        new_manager = SimpleConfigManager()
        result = new_manager.load_from_file(self.temp_file)
        assert result is True
        assert new_manager._file_path == self.temp_file

        # 验证加载的数据
        assert new_manager.get("file_key1") == "file_value1"
        assert new_manager.get("file_key2") == {"nested": "value"}
        assert new_manager.get("file_key3") == [1, 2, 3]

    def test_invalid_key_handling(self):
        """测试无效键的处理"""
        with pytest.raises(ConfigKeyError):
            self.manager.set("", "value")

        with pytest.raises(ConfigKeyError):
            self.manager.set(None, "value")

        with pytest.raises(ConfigKeyError):
            self.manager.set(123, "value")

    def test_file_operations_error_handling(self):
        """测试文件操作错误处理"""
        # 测试加载不存在的文件
        with pytest.raises(ConfigError):
            self.manager.load_from_file("/nonexistent/path/config.json")

        # 测试保存到无效路径
        # 使用mock来强制抛出异常，确保测试的可靠性
        from unittest.mock import patch, mock_open
        with patch("builtins.open", side_effect=OSError("Permission denied")):
            with pytest.raises(ConfigError):
                self.manager.save_to_file("/invalid/path/config.json")

    def test_overwrite_existing_key(self):
        """测试覆盖现有键"""
        self.manager.set("overwrite_key", "original_value")
        assert self.manager.get("overwrite_key") == "original_value"

        self.manager.set("overwrite_key", "new_value")
        assert self.manager.get("overwrite_key") == "new_value"

    def test_empty_config_operations(self):
        """测试空配置的操作"""
        # 空配置应该正常工作
        assert self.manager.get("any_key") is None
        assert self.manager.delete("any_key") is False
        assert self.manager.exists("any_key") is False
        assert self.manager.keys() == []

        # 清空空配置应该正常工作
        assert self.manager.clear() is True
        assert self.manager.get_all() == {}


if __name__ == '__main__':
    pytest.main([__file__])
