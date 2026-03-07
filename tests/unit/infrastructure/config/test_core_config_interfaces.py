#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Core Config Interfaces 测试

测试 src/infrastructure/config/core/config_interfaces.py 文件的功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from unittest.mock import Mock

# 尝试导入模块
try:
    from src.infrastructure.config.core.config_interfaces import (
        ValidationResult,
        ValidationSeverity,
        BaseConfigValidator,
        TypedConfigBase
    )
    MODULE_AVAILABLE = True
    IMPORT_ERROR = None
except ImportError as e:
    MODULE_AVAILABLE = False
    IMPORT_ERROR = e


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestValidationResult:
    """测试ValidationResult类"""

    def test_validation_result_valid(self):
        """测试有效验证结果"""
        result = ValidationResult(True, "Configuration is valid")
        
        assert result.valid is True
        assert result.message == "Configuration is valid"
        assert result.errors == []

    def test_validation_result_invalid(self):
        """测试无效验证结果"""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult(False, "Configuration is invalid", errors)
        
        assert result.valid is False
        assert result.message == "Configuration is invalid"
        assert result.errors == errors

    def test_validation_result_default_errors(self):
        """测试默认错误列表"""
        result = ValidationResult(True)
        
        assert result.valid is True
        assert result.message == ""
        assert result.errors == []

    def test_validation_result_none_errors(self):
        """测试None错误列表"""
        result = ValidationResult(True, "Message", None)
        
        assert result.errors == []


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestValidationSeverity:
    """测试ValidationSeverity枚举"""

    def test_validation_severity_values(self):
        """测试验证严重程度值"""
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.CRITICAL.value == "critical"

    def test_validation_severity_membership(self):
        """测试枚举成员"""
        assert ValidationSeverity.INFO in ValidationSeverity
        assert ValidationSeverity.WARNING in ValidationSeverity
        assert ValidationSeverity.ERROR in ValidationSeverity
        assert ValidationSeverity.CRITICAL in ValidationSeverity

    def test_validation_severity_iteration(self):
        """测试枚举迭代"""
        severities = list(ValidationSeverity)
        assert len(severities) == 4
        assert ValidationSeverity.INFO in severities
        assert ValidationSeverity.WARNING in severities
        assert ValidationSeverity.ERROR in severities
        assert ValidationSeverity.CRITICAL in severities


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestBaseConfigValidator:
    """测试BaseConfigValidator抽象基类"""

    def test_base_config_validator_is_abstract(self):
        """测试基类是抽象的"""
        # 尝试直接实例化应该失败
        with pytest.raises(TypeError):
            BaseConfigValidator()

    def test_base_config_validator_subclass(self):
        """测试继承基类的实现"""
        class ConcreteValidator(BaseConfigValidator):
            def validate_config(self, config):
                return ValidationResult(True, "Valid")
        
        validator = ConcreteValidator()
        assert hasattr(validator, 'validate_config')
        assert callable(validator.validate_config)
        
        # 测试调用
        result = validator.validate_config({})
        assert isinstance(result, ValidationResult)


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestTypedConfigBase:
    """测试TypedConfigBase类"""

    def test_typed_config_base_initialization(self):
        """测试初始化"""
        config_base = TypedConfigBase()
        
        assert hasattr(config_base, '_config')
        assert isinstance(config_base._config, dict)
        assert len(config_base._config) == 0

    def test_set_config(self):
        """测试设置配置"""
        config_base = TypedConfigBase()
        
        config_base.set_config("key1", "value1")
        config_base.set_config("key2", 42)
        config_base.set_config("key3", {"nested": "value"})
        
        assert config_base._config["key1"] == "value1"
        assert config_base._config["key2"] == 42
        assert config_base._config["key3"] == {"nested": "value"}

    def test_get_config_existing(self):
        """测试获取已存在的配置"""
        config_base = TypedConfigBase()
        config_base.set_config("test_key", "test_value")
        
        value = config_base.get_config("test_key")
        assert value == "test_value"

    def test_get_config_nonexistent(self):
        """测试获取不存在的配置"""
        config_base = TypedConfigBase()
        
        value = config_base.get_config("nonexistent")
        assert value is None

    def test_get_config_with_default(self):
        """测试使用默认值获取配置"""
        config_base = TypedConfigBase()
        
        value = config_base.get_config("nonexistent", "default_value")
        assert value == "default_value"

    def test_get_config_different_default_types(self):
        """测试不同类型的默认值"""
        config_base = TypedConfigBase()
        
        # 字符串默认值
        string_value = config_base.get_config("missing", "default")
        assert string_value == "default"
        
        # 数字默认值
        int_value = config_base.get_config("missing", 123)
        assert int_value == 123
        
        # 列表默认值
        list_value = config_base.get_config("missing", [1, 2, 3])
        assert list_value == [1, 2, 3]
        
        # 字典默认值
        dict_value = config_base.get_config("missing", {"key": "value"})
        assert dict_value == {"key": "value"}

    def test_set_and_get_config_types(self):
        """测试设置和获取不同类型的配置"""
        config_base = TypedConfigBase()
        
        test_values = [
            ("string", "hello"),
            ("integer", 42),
            ("float", 3.14),
            ("boolean", True),
            ("list", [1, 2, 3]),
            ("dict", {"nested": "value"}),
            ("none", None)
        ]
        
        for key, value in test_values:
            config_base.set_config(key, value)
            retrieved = config_base.get_config(key)
            assert retrieved == value

    def test_validate_config(self):
        """测试配置验证"""
        config_base = TypedConfigBase()
        
        result = config_base.validate()
        
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.message == "Configuration is valid"

    def test_config_overwrite(self):
        """测试配置覆盖"""
        config_base = TypedConfigBase()
        
        # 设置初始值
        config_base.set_config("key", "initial_value")
        assert config_base.get_config("key") == "initial_value"
        
        # 覆盖值
        config_base.set_config("key", "new_value")
        assert config_base.get_config("key") == "new_value"

    def test_config_multiple_keys(self):
        """测试多个配置键"""
        config_base = TypedConfigBase()
        
        # 设置多个键
        keys_values = {
            "app_name": "TestApp",
            "version": "1.0.0",
            "debug": True,
            "port": 8080,
            "database_url": "sqlite:///test.db"
        }
        
        for key, value in keys_values.items():
            config_base.set_config(key, value)
        
        # 验证所有键
        for key, expected_value in keys_values.items():
            actual_value = config_base.get_config(key)
            assert actual_value == expected_value

    def test_config_empty_string_key(self):
        """测试空字符串键"""
        config_base = TypedConfigBase()
        
        config_base.set_config("", "empty_key_value")
        assert config_base.get_config("") == "empty_key_value"

    def test_config_none_value(self):
        """测试None值"""
        config_base = TypedConfigBase()
        
        config_base.set_config("none_key", None)
        assert config_base.get_config("none_key") is None
        assert config_base.get_config("none_key", "default") is None  # None值不会被默认值替换


@pytest.mark.skipif(not MODULE_AVAILABLE, reason=f"模块导入失败: {IMPORT_ERROR if IMPORT_ERROR else 'Unknown error'}")
class TestConfigInterfacesIntegration:
    """测试配置接口集成功能"""

    def test_validation_result_with_severity(self):
        """测试验证结果与严重程度的配合使用"""
        # 虽然ValidationResult没有severity字段，但我们可以测试它们的使用
        result = ValidationResult(False, "Error message", ["error1", "error2"])
        
        assert result.valid is False
        assert "Error" in result.message
        assert len(result.errors) == 2

    def test_typed_config_with_validator(self):
        """测试TypedConfigBase与验证器的配合"""
        class TestValidator(BaseConfigValidator):
            def validate_config(self, config):
                if "required_key" in config:
                    return ValidationResult(True, "Valid configuration")
                else:
                    return ValidationResult(False, "Missing required key", ["required_key is missing"])
        
        validator = TestValidator()
        config_base = TypedConfigBase()
        
        # 测试无效配置
        invalid_result = validator.validate_config(config_base._config)
        assert invalid_result.valid is False
        
        # 设置有效配置
        config_base.set_config("required_key", "value")
        
        # 测试有效配置
        valid_result = validator.validate_config(config_base._config)
        assert valid_result.valid is True

    def test_full_workflow(self):
        """测试完整工作流"""
        # 创建配置基类
        config = TypedConfigBase()
        
        # 设置配置
        config.set_config("database_host", "localhost")
        config.set_config("database_port", 5432)
        config.set_config("debug_mode", False)
        
        # 验证配置存在
        assert config.get_config("database_host") == "localhost"
        assert config.get_config("database_port") == 5432
        assert config.get_config("debug_mode") is False
        
        # 验证配置（默认总是返回True）
        validation_result = config.validate()
        assert validation_result.valid is True
        
        # 获取不存在的配置（使用默认值）
        timeout = config.get_config("timeout", 30)
        assert timeout == 30
