"""
测试配置系统核心接口

覆盖 ValidationResult, ValidationSeverity, BaseConfigValidator, TypedConfigBase 等接口
"""

import pytest
from abc import ABC
from src.infrastructure.config.core.config_interfaces import (
    ValidationResult,
    ValidationSeverity,
    BaseConfigValidator,
    TypedConfigBase
)


class TestValidationResult:
    """ValidationResult 单元测试"""

    def test_initialization_valid_only(self):
        """测试只传入valid参数的初始化"""
        result = ValidationResult(True)
        assert result.valid is True
        assert result.message == ""
        assert result.errors == []

    def test_initialization_with_message(self):
        """测试带消息的初始化"""
        result = ValidationResult(False, "Validation failed")
        assert result.valid is False
        assert result.message == "Validation failed"
        assert result.errors == []

    def test_initialization_with_errors(self):
        """测试带错误的初始化"""
        errors = ["Error 1", "Error 2"]
        result = ValidationResult(False, "Multiple errors", errors)
        assert result.valid is False
        assert result.message == "Multiple errors"
        assert result.errors == errors

    def test_initialization_empty_errors(self):
        """测试空错误列表的初始化"""
        result = ValidationResult(True, "All good", None)
        assert result.valid is True
        assert result.message == "All good"
        assert result.errors == []


class TestValidationSeverity:
    """ValidationSeverity 单元测试"""

    def test_enum_values(self):
        """测试枚举值"""
        assert ValidationSeverity.INFO.value == "info"
        assert ValidationSeverity.WARNING.value == "warning"
        assert ValidationSeverity.ERROR.value == "error"
        assert ValidationSeverity.CRITICAL.value == "critical"

    def test_enum_membership(self):
        """测试枚举成员"""
        assert ValidationSeverity.INFO in ValidationSeverity
        assert ValidationSeverity.WARNING in ValidationSeverity
        assert ValidationSeverity.ERROR in ValidationSeverity
        assert ValidationSeverity.CRITICAL in ValidationSeverity

    def test_enum_iteration(self):
        """测试枚举迭代"""
        severities = list(ValidationSeverity)
        assert len(severities) == 4
        assert ValidationSeverity.INFO in severities
        assert ValidationSeverity.CRITICAL in severities


class TestBaseConfigValidator:
    """BaseConfigValidator 单元测试"""

    def test_is_abstract_class(self):
        """测试是抽象类"""
        # Should not be able to instantiate directly
        with pytest.raises(TypeError):
            BaseConfigValidator()

    def test_abstract_method_exists(self):
        """测试抽象方法存在"""
        assert hasattr(BaseConfigValidator, 'validate_config')

        # Check that it's marked as abstract
        method = getattr(BaseConfigValidator, 'validate_config')
        assert hasattr(method, '__isabstractmethod__')

    def test_inheritance_requirement(self):
        """测试继承要求"""

        class ConcreteValidator(BaseConfigValidator):
            def validate_config(self, config):
                return ValidationResult(True, "Valid")

        # Should be able to instantiate concrete implementation
        validator = ConcreteValidator()
        assert isinstance(validator, BaseConfigValidator)

        # Should be able to call the method
        result = validator.validate_config({})
        assert isinstance(result, ValidationResult)
        assert result.valid is True


class TestTypedConfigBase:
    """TypedConfigBase 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        config = TypedConfigBase()
        assert config._config == {}
        assert isinstance(config._config, dict)

    def test_set_config(self):
        """测试设置配置"""
        config = TypedConfigBase()

        # Test setting string value
        config.set_config("key1", "value1")
        assert config._config["key1"] == "value1"

        # Test setting different types
        config.set_config("key2", 42)
        assert config._config["key2"] == 42

        config.set_config("key3", {"nested": "value"})
        assert config._config["key3"] == {"nested": "value"}

        config.set_config("key4", [1, 2, 3])
        assert config._config["key4"] == [1, 2, 3]

    def test_get_config_existing_key(self):
        """测试获取现有配置"""
        config = TypedConfigBase()
        config.set_config("key1", "value1")

        result = config.get_config("key1")
        assert result == "value1"

    def test_get_config_nonexistent_key(self):
        """测试获取不存在的配置"""
        config = TypedConfigBase()

        result = config.get_config("nonexistent")
        assert result is None

    def test_get_config_with_default(self):
        """测试获取配置并提供默认值"""
        config = TypedConfigBase()

        result = config.get_config("nonexistent", "default_value")
        assert result == "default_value"

    def test_get_config_overrides_default(self):
        """测试现有配置覆盖默认值"""
        config = TypedConfigBase()
        config.set_config("key1", "actual_value")

        result = config.get_config("key1", "default_value")
        assert result == "actual_value"

    def test_validate_default_implementation(self):
        """测试默认验证实现"""
        config = TypedConfigBase()

        result = config.validate()
        assert isinstance(result, ValidationResult)
        assert result.valid is True
        assert result.message == "Configuration is valid"
        assert result.errors == []

    def test_config_isolation(self):
        """测试配置隔离"""
        config1 = TypedConfigBase()
        config2 = TypedConfigBase()

        config1.set_config("key", "value1")
        config2.set_config("key", "value2")

        assert config1.get_config("key") == "value1"
        assert config2.get_config("key") == "value2"
        assert config1._config != config2._config

    def test_config_modification_through_methods(self):
        """测试通过方法修改配置"""
        config = TypedConfigBase()

        # Set multiple values
        config.set_config("app.name", "MyApp")
        config.set_config("app.version", "1.0.0")
        config.set_config("database.host", "localhost")
        config.set_config("database.port", 5432)

        # Verify all values are stored
        assert config.get_config("app.name") == "MyApp"
        assert config.get_config("app.version") == "1.0.0"
        assert config.get_config("database.host") == "localhost"
        assert config.get_config("database.port") == 5432

        # Verify the internal structure
        assert len(config._config) == 4

    def test_inheritance_extension(self):
        """测试继承扩展"""

        class ExtendedConfig(TypedConfigBase):
            def validate(self):
                # Custom validation logic
                if not self.get_config("required_field"):
                    return ValidationResult(False, "Required field missing")
                return ValidationResult(True, "Custom validation passed")

        config = ExtendedConfig()

        # Test without required field
        result = config.validate()
        assert result.valid is False
        assert "missing" in result.message

        # Test with required field
        config.set_config("required_field", "present")
        result = config.validate()
        assert result.valid is True
        assert "passed" in result.message
