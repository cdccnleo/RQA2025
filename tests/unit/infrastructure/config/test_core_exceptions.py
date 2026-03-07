#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试核心配置异常类
测试 src.infrastructure.config.core.exceptions 模块的异常类
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from typing import Any, Dict, Optional

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
    ConfigConnectionError,
    ConfigStorageError,
    ConfigCacheError,
    ConfigMonitorError,
    ConfigPerformanceError,
    handle_config_exception,
    handle_config_load_exception,
    handle_config_validation_exception
)


class TestConfigException:
    """测试配置基础异常类"""

    def test_config_exception_basic(self):
        """测试基础异常初始化"""
        error = ConfigException("Test message")
        assert str(error) == "Test message"
        assert error.config_key is None
        assert error.details == {}
        assert error.message == "Test message"

    def test_config_exception_with_key(self):
        """测试带配置键的异常"""
        error = ConfigException("Test message", config_key="test.key")
        assert error.config_key == "test.key"
        assert error.details == {}

    def test_config_exception_with_details(self):
        """测试带详细信息的异常"""
        details = {"source": "file", "line": 10}
        error = ConfigException("Test message", config_key="test.key", details=details)
        assert error.details == details


class TestConfigLoadError:
    """测试配置加载错误"""

    def test_config_load_error_basic(self):
        """测试基础加载错误"""
        error = ConfigLoadError("Load failed")
        assert str(error) == "Load failed"
        assert error.source is None

    def test_config_load_error_with_source(self):
        """测试带源信息的加载错误"""
        error = ConfigLoadError("Load failed", source="config.json")
        assert error.source == "config.json"

    def test_config_load_error_with_kwargs(self):
        """测试带额外参数的加载错误"""
        error = ConfigLoadError("Load failed", source="config.json", config_key="app.name")
        assert error.source == "config.json"
        assert error.config_key == "app.name"


class TestConfigValidationError:
    """测试配置验证错误"""

    def test_config_validation_error_basic(self):
        """测试基础验证错误"""
        error = ConfigValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert error.expected_type is None
        assert error.actual_type is None
        assert error.value is None

    def test_config_validation_error_with_types(self):
        """测试带类型信息的验证错误"""
        error = ConfigValidationError(
            "Type mismatch",
            expected_type="int",
            actual_type="str",
            value="not_a_number"
        )
        assert error.expected_type == "int"
        assert error.actual_type == "str"
        assert error.value == "not_a_number"


class TestConfigTypeError:
    """测试配置类型错误"""

    def test_config_type_error_basic(self):
        """测试基础类型错误"""
        error = ConfigTypeError("Type error")
        assert isinstance(error, ConfigValidationError)
        assert str(error) == "Type error"


class TestConfigKeyError:
    """测试配置键错误"""

    def test_config_key_error_basic(self):
        """测试基础键错误"""
        error = ConfigKeyError("Key error", key="missing.key")
        assert str(error) == "Key error"
        assert error.key == "missing.key"
        assert error.config_key == "missing.key"


class TestConfigNotFoundError:
    """测试配置未找到错误"""

    def test_config_not_found_error_basic(self):
        """测试基础未找到错误"""
        error = ConfigNotFoundError("Not found", key="missing.key")
        assert isinstance(error, ConfigKeyError)
        assert error.key == "missing.key"


class TestConfigDuplicateError:
    """测试配置重复错误"""

    def test_config_duplicate_error_basic(self):
        """测试基础重复错误"""
        error = ConfigDuplicateError("Duplicate key", key="duplicate.key")
        assert error.key == "duplicate.key"


class TestConfigSecurityError:
    """测试配置安全错误"""

    def test_config_security_error_basic(self):
        """测试基础安全错误"""
        error = ConfigSecurityError("Security issue")
        assert error.security_issue is None

    def test_config_security_error_with_issue(self):
        """测试带安全问题的安全错误"""
        error = ConfigSecurityError("Security issue", security_issue="unauthorized_access")
        assert error.security_issue == "unauthorized_access"


class TestConfigEncryptionError:
    """测试配置加密错误"""

    def test_config_encryption_error_basic(self):
        """测试基础加密错误"""
        error = ConfigEncryptionError("Encryption failed")
        assert isinstance(error, ConfigSecurityError)
        assert error.algorithm is None

    def test_config_encryption_error_with_algorithm(self):
        """测试带算法的加密错误"""
        error = ConfigEncryptionError("Encryption failed", algorithm="AES-256")
        assert error.algorithm == "AES-256"


class TestConfigAccessError:
    """测试配置访问错误"""

    def test_config_access_error_basic(self):
        """测试基础访问错误"""
        error = ConfigAccessError("Access denied")
        assert isinstance(error, ConfigSecurityError)
        assert error.user is None
        assert error.permission is None

    def test_config_access_error_with_details(self):
        """测试带详细信息的访问错误"""
        error = ConfigAccessError("Access denied", user="user1", permission="read")
        assert error.user == "user1"
        assert error.permission == "read"


class TestConfigQuotaError:
    """测试配置配额错误"""

    def test_config_quota_error_basic(self):
        """测试基础配额错误"""
        error = ConfigQuotaError("Quota exceeded")
        assert error.quota_type is None
        assert error.current_usage is None
        assert error.limit is None

    def test_config_quota_error_with_details(self):
        """测试带详细信息的配额错误"""
        error = ConfigQuotaError(
            "Quota exceeded",
            quota_type="memory",
            current_usage=100,
            limit=50
        )
        assert error.quota_type == "memory"
        assert error.current_usage == 100
        assert error.limit == 50


class TestConfigVersionError:
    """测试配置版本错误"""

    def test_config_version_error_basic(self):
        """测试基础版本错误"""
        error = ConfigVersionError("Version mismatch")
        assert error.version is None
        assert error.expected_version is None

    def test_config_version_error_with_versions(self):
        """测试带版本信息的版本错误"""
        error = ConfigVersionError(
            "Version mismatch",
            version="1.0",
            expected_version="2.0"
        )
        assert error.version == "1.0"
        assert error.expected_version == "2.0"


class TestConfigMergeError:
    """测试配置合并错误"""

    def test_config_merge_error_basic(self):
        """测试基础合并错误"""
        error = ConfigMergeError("Merge conflict")
        assert error.conflict_keys == []

    def test_config_merge_error_with_conflicts(self):
        """测试带冲突键的合并错误"""
        error = ConfigMergeError("Merge conflict", conflict_keys=["key1", "key2"])
        assert error.conflict_keys == ["key1", "key2"]


class TestConfigTimeoutError:
    """测试配置超时错误"""

    def test_config_timeout_error_basic(self):
        """测试基础超时错误"""
        error = ConfigTimeoutError("Operation timed out")
        assert error.timeout is None
        assert error.operation is None

    def test_config_timeout_error_with_details(self):
        """测试带详细信息的超时错误"""
        error = ConfigTimeoutError(
            "Operation timed out",
            timeout=30.0,
            operation="load_config"
        )
        assert error.timeout == 30.0
        assert error.operation == "load_config"


class TestConfigConnectionError:
    """测试配置连接错误"""

    def test_config_connection_error_basic(self):
        """测试基础连接错误"""
        error = ConfigConnectionError("Connection failed")
        assert error.host is None
        assert error.port is None

    def test_config_connection_error_with_details(self):
        """测试带详细信息的连接错误"""
        error = ConfigConnectionError(
            "Connection failed",
            host="localhost",
            port=8080
        )
        assert error.host == "localhost"
        assert error.port == 8080


class TestConfigStorageError:
    """测试配置存储错误"""

    def test_config_storage_error_basic(self):
        """测试基础存储错误"""
        error = ConfigStorageError("Storage failed")
        assert error.storage_type is None
        assert error.operation is None

    def test_config_storage_error_with_details(self):
        """测试带详细信息的存储错误"""
        error = ConfigStorageError(
            "Storage failed",
            storage_type="redis",
            operation="save"
        )
        assert error.storage_type == "redis"
        assert error.operation == "save"


class TestConfigCacheError:
    """测试配置缓存错误"""

    def test_config_cache_error_basic(self):
        """测试基础缓存错误"""
        error = ConfigCacheError("Cache failed")
        assert error.cache_key is None
        assert error.operation is None

    def test_config_cache_error_with_details(self):
        """测试带详细信息的缓存错误"""
        error = ConfigCacheError(
            "Cache failed",
            cache_key="user:123",
            operation="get"
        )
        assert error.cache_key == "user:123"
        assert error.operation == "get"
        assert error.config_key == "user:123"  # 应该与cache_key相同


class TestConfigMonitorError:
    """测试配置监控错误"""

    def test_config_monitor_error_basic(self):
        """测试基础监控错误"""
        error = ConfigMonitorError("Monitor failed")
        assert error.metric is None

    def test_config_monitor_error_with_metric(self):
        """测试带指标的监控错误"""
        error = ConfigMonitorError("Monitor failed", metric="cpu_usage")
        assert error.metric == "cpu_usage"


class TestConfigPerformanceError:
    """测试配置性能错误"""

    def test_config_performance_error_basic(self):
        """测试基础性能错误"""
        error = ConfigPerformanceError("Performance issue")
        assert error.threshold is None
        assert error.actual_value is None

    def test_config_performance_error_with_details(self):
        """测试带详细信息的性能错误"""
        error = ConfigPerformanceError(
            "Performance issue",
            threshold=100.0,
            actual_value=150.0
        )
        assert error.threshold == 100.0
        assert error.actual_value == 150.0


class TestExceptionDecorators:
    """测试异常处理装饰器"""

    def test_handle_config_exception_success(self):
        """测试成功情况下的配置异常装饰器"""
        @handle_config_exception("test_operation")
        def successful_function():
            return "success"

        result = successful_function()
        assert result == "success"

    def test_handle_config_exception_config_error(self):
        """测试配置异常情况下的装饰器"""
        @handle_config_exception("test_operation")
        def failing_function():
            raise ConfigException("Test config error")

        with pytest.raises(ConfigException, match="Test config error"):
            failing_function()

    def test_handle_config_exception_generic_error(self):
        """测试一般异常情况下的装饰器"""
        @handle_config_exception("test_operation")
        def failing_function():
            raise ValueError("Test generic error")

        with pytest.raises(ConfigException) as exc_info:
            failing_function()

        assert "test_operation 失败" in str(exc_info.value)
        assert "Test generic error" in str(exc_info.value.details["original_error"])

    def test_handle_config_load_exception_success(self):
        """测试成功情况下的配置加载异常装饰器"""
        @handle_config_load_exception("test_source")
        def successful_function():
            return "loaded"

        result = successful_function()
        assert result == "loaded"

    def test_handle_config_load_exception_config_load_error(self):
        """测试配置加载异常情况下的装饰器"""
        @handle_config_load_exception("test_source")
        def failing_function():
            raise ConfigLoadError("Test load error")

        with pytest.raises(ConfigLoadError, match="Test load error"):
            failing_function()

    def test_handle_config_load_exception_generic_error(self):
        """测试一般异常情况下的配置加载装饰器"""
        @handle_config_load_exception("test_source")
        def failing_function():
            raise ConnectionError("Connection failed")

        with pytest.raises(ConfigLoadError) as exc_info:
            failing_function()

        assert "从 test_source 加载配置失败" in str(exc_info.value)
        assert "Connection failed" in str(exc_info.value.details["original_error"])

    def test_handle_config_validation_exception_success(self):
        """测试成功情况下的配置验证异常装饰器"""
        @handle_config_validation_exception("test_field")
        def successful_function():
            return "validated"

        result = successful_function()
        assert result == "validated"

    def test_handle_config_validation_exception_config_validation_error(self):
        """测试配置验证异常情况下的装饰器"""
        @handle_config_validation_exception("test_field")
        def failing_function():
            raise ConfigValidationError("Test validation error")

        with pytest.raises(ConfigValidationError, match="Test validation error"):
            failing_function()

    def test_handle_config_validation_exception_generic_error(self):
        """测试一般异常情况下的配置验证装饰器"""
        @handle_config_validation_exception("test_field")
        def failing_function():
            raise TypeError("Type mismatch")

        with pytest.raises(ConfigValidationError) as exc_info:
            failing_function()

        assert "验证字段 'test_field' 失败" in str(exc_info.value)
        assert "Type mismatch" in str(exc_info.value.details["original_error"])


class TestExceptionInheritance:
    """测试异常继承关系"""

    def test_exception_hierarchy(self):
        """测试异常类层次结构"""
        # 测试主要异常类的继承关系
        assert issubclass(ConfigLoadError, ConfigException)
        assert issubclass(ConfigValidationError, ConfigException)
        assert issubclass(ConfigTypeError, ConfigValidationError)
        assert issubclass(ConfigKeyError, ConfigException)
        assert issubclass(ConfigNotFoundError, ConfigKeyError)
        assert issubclass(ConfigDuplicateError, ConfigException)
        assert issubclass(ConfigSecurityError, ConfigException)
        assert issubclass(ConfigEncryptionError, ConfigSecurityError)
        assert issubclass(ConfigAccessError, ConfigSecurityError)
        assert issubclass(ConfigQuotaError, ConfigException)
        assert issubclass(ConfigVersionError, ConfigException)
        assert issubclass(ConfigMergeError, ConfigException)
        assert issubclass(ConfigTimeoutError, ConfigException)
        assert issubclass(ConfigConnectionError, ConfigException)
        assert issubclass(ConfigStorageError, ConfigException)
        assert issubclass(ConfigCacheError, ConfigException)
        assert issubclass(ConfigMonitorError, ConfigException)
        assert issubclass(ConfigPerformanceError, ConfigException)


class TestExceptionAttributes:
    """测试异常属性"""

    def test_all_exceptions_have_required_attributes(self):
        """测试所有异常类都有必需的属性"""
        exceptions_to_test = [
            (ConfigException, ["config_key", "details", "message"]),
            (ConfigLoadError, ["config_key", "details", "message", "source"]),
            (ConfigValidationError, ["config_key", "details", "message", "expected_type", "actual_type", "value"]),
            (ConfigTypeError, ["config_key", "details", "message", "expected_type", "actual_type", "value"]),
            (ConfigKeyError, ["config_key", "details", "message", "key"]),
            (ConfigNotFoundError, ["config_key", "details", "message", "key"]),
            (ConfigDuplicateError, ["config_key", "details", "message", "key"]),
            (ConfigSecurityError, ["config_key", "details", "message", "security_issue"]),
            (ConfigEncryptionError, ["config_key", "details", "message", "security_issue", "algorithm"]),
            (ConfigAccessError, ["config_key", "details", "message", "security_issue", "user", "permission"]),
            (ConfigQuotaError, ["config_key", "details", "message", "quota_type", "current_usage", "limit"]),
            (ConfigVersionError, ["config_key", "details", "message", "version", "expected_version"]),
            (ConfigMergeError, ["config_key", "details", "message", "conflict_keys"]),
            (ConfigTimeoutError, ["config_key", "details", "message", "timeout", "operation"]),
            (ConfigConnectionError, ["config_key", "details", "message", "host", "port"]),
            (ConfigStorageError, ["config_key", "details", "message", "storage_type", "operation"]),
            (ConfigCacheError, ["config_key", "details", "message", "cache_key", "operation"]),
            (ConfigMonitorError, ["config_key", "details", "message", "metric"]),
            (ConfigPerformanceError, ["config_key", "details", "message", "threshold", "actual_value"]),
        ]

        for exception_class, expected_attrs in exceptions_to_test:
            # 创建异常实例
            if exception_class == ConfigException:
                instance = exception_class("test")
            elif exception_class in [ConfigLoadError, ConfigKeyError, ConfigNotFoundError, ConfigDuplicateError]:
                instance = exception_class("test", **{expected_attrs[3]: "test_value"} if len(expected_attrs) > 3 else {})
            elif exception_class in [ConfigValidationError, ConfigTypeError]:
                instance = exception_class("test", expected_type="str", actual_type="int", value=123)
            elif exception_class == ConfigSecurityError:
                instance = exception_class("test", security_issue="test_issue")
            elif exception_class == ConfigEncryptionError:
                instance = exception_class("test", security_issue="test_issue", algorithm="AES")
            elif exception_class == ConfigAccessError:
                instance = exception_class("test", security_issue="test_issue", user="test_user", permission="read")
            elif exception_class == ConfigQuotaError:
                instance = exception_class("test", quota_type="memory", current_usage=100, limit=50)
            elif exception_class == ConfigVersionError:
                instance = exception_class("test", version="1.0", expected_version="2.0")
            elif exception_class == ConfigMergeError:
                instance = exception_class("test", conflict_keys=["key1"])
            elif exception_class == ConfigTimeoutError:
                instance = exception_class("test", timeout=30.0, operation="load")
            elif exception_class == ConfigConnectionError:
                instance = exception_class("test", host="localhost", port=8080)
            elif exception_class == ConfigStorageError:
                instance = exception_class("test", storage_type="redis", operation="save")
            elif exception_class == ConfigCacheError:
                instance = exception_class("test", cache_key="test:key", operation="get")
            elif exception_class == ConfigMonitorError:
                instance = exception_class("test", metric="cpu_usage")
            elif exception_class == ConfigPerformanceError:
                instance = exception_class("test", threshold=100.0, actual_value=150.0)
            else:
                instance = exception_class("test")

            # 检查所有必需属性是否存在
            for attr in expected_attrs:
                assert hasattr(instance, attr), f"{exception_class.__name__} should have attribute '{attr}'"
