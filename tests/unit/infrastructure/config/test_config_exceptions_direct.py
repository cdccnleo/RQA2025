"""
直接测试配置管理异常类的测试文件
测试src/infrastructure/config/config_exceptions.py中的实际代码
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.config.config_exceptions import (
    ConfigError,
    ConfigLoadError,
    ConfigValidationError,
    ConfigTypeError,
    ConfigAccessError,
    ConfigValueError
)


class TestConfigExceptionsDirect:
    """直接测试配置管理异常类"""

    def test_config_error_creation(self):
        """测试ConfigError异常创建"""
        error = ConfigError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_config_error_with_cause(self):
        """测试ConfigError异常带原因"""
        cause = ValueError("Original error")
        error = ConfigError("Test error", config_key="test.key", details={'cause': cause})
        assert str(error) == "Test error"
        assert error.details.get('cause') == cause

    def test_config_load_error_creation(self):
        """测试ConfigLoadError异常创建"""
        error = ConfigLoadError("Load failed", "test.json")
        assert str(error) == "Load failed"
        assert error.details.get('source') == "test.json"
        assert isinstance(error, ConfigError)

    def test_config_validation_error_creation(self):
        """测试ConfigValidationError异常创建"""
        error = ConfigValidationError("Validation failed", "database.host")
        assert str(error) == "Validation failed"
        assert error.config_key == "database.host"
        assert isinstance(error, ConfigError)

    def test_config_type_error_creation(self):
        """测试ConfigTypeError异常创建"""
        error = ConfigTypeError("Type mismatch", expected_type="str", actual_type="int")
        assert str(error) == "Type mismatch"
        assert error.details.get('expected_type') == "str"
        assert error.details.get('actual_type') == "int"
        assert isinstance(error, ConfigError)

    def test_config_access_error_creation(self):
        """测试ConfigAccessError异常创建"""
        error = ConfigAccessError("Access denied", "secret.key")
        assert str(error) == "Access denied"
        assert error.config_key == "secret.key"
        assert isinstance(error, ConfigError)

    def test_config_value_error_creation(self):
        """测试ConfigValueError异常创建"""
        error = ConfigValueError("Invalid value", config_key="port", value=99999)
        assert str(error) == "Invalid value"
        assert error.config_key == "port"
        assert error.details.get('value') == 99999
        assert isinstance(error, ConfigError)

