"""
测试配置异常类

覆盖 ConfigException 及其子类的功能
"""

import pytest
from src.infrastructure.config.core.exceptions import (
    ConfigException,
    ConfigLoadError,
    ConfigValidationError,
    ConfigTypeError,
    ConfigKeyError,
    ConfigNotFoundError,
    ConfigDuplicateError,
    ConfigSecurityError,
    ConfigEncryptionError,
    ConfigAccessError,
    ConfigQuotaError,
    ConfigVersionError,
    ConfigMergeError,
    ConfigTimeoutError,
    handle_config_validation_exception
)


class TestConfigException:
    """ConfigException 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigException("Test message")
        assert str(exc) == "Test message"
        assert exc.config_key is None
        assert exc.details == {}
        assert exc.error_type is None

    def test_initialization_with_config_key(self):
        """测试带配置键初始化"""
        exc = ConfigException("Test message", config_key="test.key")
        assert exc.config_key == "test.key"
        assert exc.error_type == "test.key"

    def test_initialization_with_details(self):
        """测试带详情初始化"""
        details = {"field": "value", "code": 123}
        exc = ConfigException("Test message", details=details)
        assert exc.details == details

    def test_initialization_with_error_type(self):
        """测试带错误类型初始化"""
        exc = ConfigException("Test message", error_type="custom_error")
        assert exc.error_type == "custom_error"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigException("Test")
        # Should inherit from InfrastructureException or Exception
        assert isinstance(exc, Exception)


class TestConfigLoadError:
    """ConfigLoadError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigLoadError("Load failed")
        assert str(exc) == "Load failed"
        assert exc.source is None
        # Context includes source by default
        assert 'source' in exc.context

    def test_initialization_with_source(self):
        """测试带源初始化"""
        exc = ConfigLoadError("Load failed", source="/path/to/config.json")
        assert exc.source == "/path/to/config.json"
        assert exc.context['source'] == "/path/to/config.json"

    def test_initialization_with_details(self):
        """测试带详情初始化"""
        details = {"line": 10, "column": 5}
        exc = ConfigLoadError("Load failed", details=details)
        # Context merges details with source
        assert exc.context["line"] == 10
        assert exc.context["column"] == 5
        assert 'source' in exc.context

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigLoadError("Test")
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigValidationError:
    """ConfigValidationError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigValidationError("Validation failed")
        assert str(exc) == "Validation failed"
        assert exc.expected_type is None
        assert exc.actual_type is None
        assert exc.value is None

    def test_initialization_with_types(self):
        """测试带类型信息初始化"""
        exc = ConfigValidationError(
            "Type mismatch",
            expected_type="str",
            actual_type="int",
            value=123
        )
        assert exc.expected_type == "str"
        assert exc.actual_type == "int"
        assert exc.value == 123

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigValidationError("Test")
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigTypeError:
    """ConfigTypeError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigTypeError("Type error")
        assert str(exc) == "Type error"

    def test_initialization_with_types(self):
        """测试带类型信息初始化"""
        exc = ConfigTypeError(
            "Type mismatch",
            expected_type="dict",
            actual_type="list",
            value=[1, 2, 3]
        )
        assert exc.expected_type == "dict"
        assert exc.actual_type == "list"
        assert exc.value == [1, 2, 3]

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigTypeError("Test")
        assert isinstance(exc, ConfigValidationError)
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigKeyError:
    """ConfigKeyError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigKeyError("Key error", key="invalid.key")
        assert str(exc) == "Key error"
        assert exc.config_key == "invalid.key"
        assert exc.key == "invalid.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigKeyError("Test", key="test")
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigNotFoundError:
    """ConfigNotFoundError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigNotFoundError("Not found", key="missing.key")
        assert str(exc) == "Not found"
        assert exc.config_key == "missing.key"
        assert exc.key == "missing.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigNotFoundError("Test", key="test")
        assert isinstance(exc, ConfigKeyError)
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigDuplicateError:
    """ConfigDuplicateError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigDuplicateError("Duplicate", key="duplicate.key")
        assert str(exc) == "Duplicate"
        assert exc.config_key == "duplicate.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigDuplicateError("Test", key="test")
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigSecurityError:
    """ConfigSecurityError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigSecurityError("Security error", config_key="secure.key")
        assert str(exc) == "Security error"
        assert exc.config_key == "secure.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigSecurityError("Test", config_key="test")
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigEncryptionError:
    """ConfigEncryptionError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigEncryptionError("Encryption error", config_key="encrypted.key")
        assert str(exc) == "Encryption error"
        assert exc.config_key == "encrypted.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigEncryptionError("Test", config_key="test")
        assert isinstance(exc, ConfigSecurityError)
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigAccessError:
    """ConfigAccessError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigAccessError("Access denied", config_key="restricted.key")
        assert str(exc) == "Access denied"
        assert exc.config_key == "restricted.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigAccessError("Test", config_key="test")
        assert isinstance(exc, ConfigSecurityError)
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigQuotaError:
    """ConfigQuotaError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigQuotaError("Quota exceeded", config_key="quota.key")
        assert str(exc) == "Quota exceeded"
        assert exc.config_key == "quota.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigQuotaError("Test", config_key="test")
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigVersionError:
    """ConfigVersionError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigVersionError("Version mismatch", config_key="versioned.key")
        assert str(exc) == "Version mismatch"
        assert exc.config_key == "versioned.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigVersionError("Test", config_key="test")
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigMergeError:
    """ConfigMergeError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigMergeError("Merge conflict", config_key="merge.key")
        assert str(exc) == "Merge conflict"
        assert exc.config_key == "merge.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigMergeError("Test", config_key="test")
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestConfigTimeoutError:
    """ConfigTimeoutError 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = ConfigTimeoutError("Operation timed out", config_key="timeout.key")
        assert str(exc) == "Operation timed out"
        assert exc.config_key == "timeout.key"

    def test_inheritance(self):
        """测试继承关系"""
        exc = ConfigTimeoutError("Test", config_key="test")
        assert isinstance(exc, ConfigException)
        assert isinstance(exc, Exception)


class TestHandleConfigValidationException:
    """handle_config_validation_exception 装饰器测试"""

    def test_decorator_with_exception(self):
        """测试装饰器处理异常"""

        @handle_config_validation_exception("test_field")
        def failing_function():
            raise ValueError("Test error")

        with pytest.raises(ConfigValidationError):
            failing_function()

    def test_decorator_success_case(self):
        """测试装饰器正常情况"""

        @handle_config_validation_exception("test_field")
        def success_function():
            return "success"

        result = success_function()
        assert result == "success"

    def test_decorator_with_config_exception(self):
        """测试装饰器处理配置异常"""

        @handle_config_validation_exception("test_field")
        def config_failing_function():
            raise ConfigValidationError("Config error")

        with pytest.raises(ConfigValidationError):
            config_failing_function()
