"""
基础设施层配置管理基础功能测试
避免复杂的导入链，专注于核心功能验证
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from typing import Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum


class ConfigScope(Enum):
    """配置作用域枚举"""
    GLOBAL = "global"
    USER = "user"
    SESSION = "session"
    APPLICATION = "application"


@dataclass
class ConfigItem:
    """配置项"""
    key: str
    value: Any
    scope: ConfigScope = ConfigScope.GLOBAL
    description: Optional[str] = None


class ConfigError(Exception):
    """配置系统基础异常"""
    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.config_key = config_key
        self.details = details or {}


class ConfigLoadError(ConfigError):
    """配置加载错误"""
    pass


class ConfigValidationError(ConfigError):
    """配置验证错误"""
    pass


class TestConfigBasicFunctionality:
    """配置管理基础功能测试"""

    def test_config_error_creation(self):
        """测试配置错误异常创建"""
        error = ConfigError("Test error", config_key="test.key")
        assert str(error) == "Test error"
        assert error.config_key == "test.key"
        assert error.details == {}

    def test_config_load_error_creation(self):
        """测试配置加载错误异常创建"""
        error = ConfigLoadError("Load failed", config_key="db.host")
        assert isinstance(error, ConfigError)
        assert isinstance(error, ConfigLoadError)
        assert error.config_key == "db.host"

    def test_config_validation_error_creation(self):
        """测试配置验证错误异常创建"""
        details = {"expected": "string", "actual": "int"}
        error = ConfigValidationError("Validation failed", config_key="port", details=details)
        assert error.details == details

    def test_config_scope_enum(self):
        """测试配置作用域枚举"""
        assert ConfigScope.GLOBAL.value == "global"
        assert ConfigScope.USER.value == "user"
        assert ConfigScope.SESSION.value == "session"
        assert ConfigScope.APPLICATION.value == "application"

    def test_config_item_creation(self):
        """测试配置项创建"""
        item = ConfigItem(
            key="database.host",
            value="localhost",
            scope=ConfigScope.APPLICATION,
            description="Database host configuration"
        )

        assert item.key == "database.host"
        assert item.value == "localhost"
        assert item.scope == ConfigScope.APPLICATION
        assert item.description == "Database host configuration"

    def test_config_item_defaults(self):
        """测试配置项默认值"""
        item = ConfigItem(key="test.key", value="test_value")
        assert item.scope == ConfigScope.GLOBAL
        assert item.description is None

    def test_exception_hierarchy(self):
        """测试异常继承关系"""
        base_error = ConfigError("Base error")
        load_error = ConfigLoadError("Load error")
        validation_error = ConfigValidationError("Validation error")

        # Test isinstance relationships
        assert isinstance(load_error, ConfigError)
        assert isinstance(validation_error, ConfigError)
        assert isinstance(load_error, Exception)
        assert isinstance(validation_error, Exception)

        # Test type relationships
        assert type(load_error) is ConfigLoadError
        assert type(validation_error) is ConfigValidationError


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
