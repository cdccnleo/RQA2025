#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Config核心异常深度测试"""

import pytest


# ============================================================================
# ConfigException基础异常测试
# ============================================================================

def test_config_exception_basic():
    """测试ConfigException基础创建"""
    from src.infrastructure.config.core.exceptions import ConfigException
    
    error = ConfigException("Test error")
    assert error.message == "Test error"
    assert error.config_key is None
    assert error.details == {}


def test_config_exception_with_key():
    """测试ConfigException包含配置键"""
    from src.infrastructure.config.core.exceptions import ConfigException
    
    error = ConfigException("Error", config_key="database.host")
    assert error.config_key == "database.host"


def test_config_exception_with_details():
    """测试ConfigException包含详情"""
    from src.infrastructure.config.core.exceptions import ConfigException
    
    details = {"line": 10, "file": "config.yml"}
    error = ConfigException("Error", details=details)
    assert error.details == details


# ============================================================================
# 配置加载错误测试
# ============================================================================

def test_config_load_error():
    """测试ConfigLoadError"""
    from src.infrastructure.config.core.exceptions import ConfigLoadError
    
    error = ConfigLoadError("Load failed", source="config.yaml", config_key="test")
    assert error.message == "Load failed"
    assert error.source == "config.yaml"
    assert error.config_key == "test"


def test_config_load_error_without_source():
    """测试ConfigLoadError无source"""
    from src.infrastructure.config.core.exceptions import ConfigLoadError
    
    error = ConfigLoadError("Load failed")
    assert error.source is None


# ============================================================================
# 配置验证错误测试
# ============================================================================

def test_config_validation_error():
    """测试ConfigValidationError"""
    from src.infrastructure.config.core.exceptions import ConfigValidationError
    
    error = ConfigValidationError(
        "Validation failed",
        expected_type="int",
        actual_type="str",
        value="abc"
    )
    assert error.expected_type == "int"
    assert error.actual_type == "str"
    assert error.value == "abc"


def test_config_type_error():
    """测试ConfigTypeError"""
    from src.infrastructure.config.core.exceptions import ConfigTypeError
    
    error = ConfigTypeError(
        "Type error",
        expected_type="float",
        actual_type="int",
        value=10
    )
    assert error.expected_type == "float"
    assert error.actual_type == "int"
    assert error.value == 10


# ============================================================================
# 配置键错误测试
# ============================================================================

def test_config_key_error():
    """测试ConfigKeyError"""
    from src.infrastructure.config.core.exceptions import ConfigKeyError
    
    error = ConfigKeyError("Key error", key="api.endpoint")
    assert error.key == "api.endpoint"
    assert error.config_key == "api.endpoint"


def test_config_not_found_error():
    """测试ConfigNotFoundError"""
    from src.infrastructure.config.core.exceptions import ConfigNotFoundError
    
    error = ConfigNotFoundError("Not found", key="missing.key")
    assert error.key == "missing.key"


# ============================================================================
# 配置重复错误测试
# ============================================================================

def test_config_duplicate_error():
    """测试ConfigDuplicateError"""
    from src.infrastructure.config.core.exceptions import ConfigDuplicateError
    
    error = ConfigDuplicateError("Duplicate config", key="database")
    assert error.key == "database"
    assert error.config_key == "database"


# ============================================================================
# 配置安全错误测试
# ============================================================================

def test_config_security_error():
    """测试ConfigSecurityError"""
    from src.infrastructure.config.core.exceptions import ConfigSecurityError
    
    error = ConfigSecurityError("Security issue", security_issue="unauthorized")
    assert error.security_issue == "unauthorized"


def test_config_encryption_error():
    """测试ConfigEncryptionError"""
    from src.infrastructure.config.core.exceptions import ConfigEncryptionError
    
    error = ConfigEncryptionError("Encryption failed", algorithm="AES-256")
    assert error.algorithm == "AES-256"


def test_config_access_error():
    """测试ConfigAccessError"""
    from src.infrastructure.config.core.exceptions import ConfigAccessError
    
    error = ConfigAccessError("Access denied", user="guest", permission="write")
    assert error.user == "guest"
    assert error.permission == "write"


# ============================================================================
# 配置配额错误测试
# ============================================================================

def test_config_quota_error():
    """测试ConfigQuotaError"""
    from src.infrastructure.config.core.exceptions import ConfigQuotaError
    
    error = ConfigQuotaError(
        "Quota exceeded",
        quota_type="storage",
        current_usage=1000,
        limit=800
    )
    assert error.quota_type == "storage"
    assert error.current_usage == 1000
    assert error.limit == 800


# ============================================================================
# 配置版本错误测试
# ============================================================================

def test_config_version_error():
    """测试ConfigVersionError"""
    from src.infrastructure.config.core.exceptions import ConfigVersionError
    
    error = ConfigVersionError(
        "Version mismatch",
        version="v2.0",
        expected_version="v1.0"
    )
    assert error.version == "v2.0"
    assert error.expected_version == "v1.0"


# ============================================================================
# 配置合并错误测试
# ============================================================================

def test_config_merge_error():
    """测试ConfigMergeError"""
    from src.infrastructure.config.core.exceptions import ConfigMergeError
    
    error = ConfigMergeError("Merge conflict", conflict_keys=["key1", "key2"])
    assert error.conflict_keys == ["key1", "key2"]


def test_config_merge_error_empty_conflicts():
    """测试ConfigMergeError无冲突键"""
    from src.infrastructure.config.core.exceptions import ConfigMergeError
    
    error = ConfigMergeError("Merge error")
    assert error.conflict_keys == []


# ============================================================================
# 配置超时错误测试
# ============================================================================

def test_config_timeout_error():
    """测试ConfigTimeoutError"""
    from src.infrastructure.config.core.exceptions import ConfigTimeoutError
    
    error = ConfigTimeoutError("Timeout", timeout=30.0, operation="load")
    assert error.timeout == 30.0
    assert error.operation == "load"


# ============================================================================
# 配置连接错误测试
# ============================================================================

def test_config_connection_error():
    """测试ConfigConnectionError"""
    from src.infrastructure.config.core.exceptions import ConfigConnectionError
    
    error = ConfigConnectionError("Connection failed", host="localhost", port=6379)
    assert error.host == "localhost"
    assert error.port == 6379


# ============================================================================
# 配置存储错误测试
# ============================================================================

def test_config_storage_error():
    """测试ConfigStorageError"""
    from src.infrastructure.config.core.exceptions import ConfigStorageError
    
    error = ConfigStorageError(
        "Storage error",
        storage_type="redis",
        operation="save"
    )
    assert error.storage_type == "redis"
    assert error.operation == "save"


# ============================================================================
# 配置缓存错误测试
# ============================================================================

def test_config_cache_error():
    """测试ConfigCacheError"""
    from src.infrastructure.config.core.exceptions import ConfigCacheError
    
    error = ConfigCacheError(
        "Cache error",
        cache_key="config:main",
        operation="get"
    )
    assert error.cache_key == "config:main"
    assert error.operation == "get"


# ============================================================================
# 配置监控错误测试
# ============================================================================

def test_config_monitor_error():
    """测试ConfigMonitorError"""
    from src.infrastructure.config.core.exceptions import ConfigMonitorError
    
    error = ConfigMonitorError("Monitor error", metric="config_changes")
    assert error.metric == "config_changes"


# ============================================================================
# 配置性能错误测试
# ============================================================================

def test_config_performance_error():
    """测试ConfigPerformanceError"""
    from src.infrastructure.config.core.exceptions import ConfigPerformanceError
    
    error = ConfigPerformanceError(
        "Performance issue",
        threshold=100.0,
        actual_value=150.0
    )
    assert error.threshold == 100.0
    assert error.actual_value == 150.0


# ============================================================================
# 兼容性异常测试
# ============================================================================

def test_configuration_error_compatibility():
    """测试ConfigurationError兼容性"""
    from src.infrastructure.config.core.exceptions import ConfigurationError
    
    error = ConfigurationError("Config error", config_key="test")
    assert error.config_key == "test"


def test_validation_error_compatibility():
    """测试ValidationError兼容性"""
    from src.infrastructure.config.core.exceptions import ValidationError
    
    error = ValidationError("Validation failed", field="email", value="invalid")
    assert error.field == "email"
    assert error.value == "invalid"


def test_infrastructure_exception_compatibility():
    """测试InfrastructureException兼容性"""
    from src.infrastructure.config.core.exceptions import InfrastructureException
    
    error = InfrastructureException("Infrastructure error", component="config")
    assert error.component == "config"


# ============================================================================
# 异常继承关系测试
# ============================================================================

def test_all_errors_inherit_config_exception():
    """测试所有异常都继承自ConfigException"""
    from src.infrastructure.config.core.exceptions import (
        ConfigException,
        ConfigLoadError,
        ConfigValidationError,
        ConfigTypeError,
        ConfigKeyError,
        ConfigNotFoundError
    )
    
    assert issubclass(ConfigLoadError, ConfigException)
    assert issubclass(ConfigValidationError, ConfigException)
    assert issubclass(ConfigTypeError, ConfigValidationError)
    assert issubclass(ConfigKeyError, ConfigException)
    assert issubclass(ConfigNotFoundError, ConfigKeyError)


# ============================================================================
# 装饰器测试
# ============================================================================

def test_handle_config_exception_decorator():
    """测试handle_config_exception装饰器"""
    from src.infrastructure.config.core.exceptions import handle_config_exception
    
    @handle_config_exception(operation="test")
    def test_func():
        return "success"
    
    result = test_func()
    assert result == "success"


def test_handle_config_exception_catches_error():
    """测试装饰器捕获错误"""
    from src.infrastructure.config.core.exceptions import (
        handle_config_exception,
        ConfigException
    )
    
    @handle_config_exception(operation="test")
    def test_func():
        raise ValueError("Test error")
    
    with pytest.raises(ConfigException):
        test_func()


def test_handle_config_load_exception_decorator():
    """测试handle_config_load_exception装饰器"""
    from src.infrastructure.config.core.exceptions import handle_config_load_exception
    
    @handle_config_load_exception(source="config.yaml")
    def load_config():
        return {"key": "value"}
    
    result = load_config()
    assert result == {"key": "value"}


def test_handle_config_validation_exception_decorator():
    """测试handle_config_validation_exception装饰器"""
    from src.infrastructure.config.core.exceptions import handle_config_validation_exception
    
    @handle_config_validation_exception(field="email")
    def validate_email():
        return True
    
    result = validate_email()
    assert result is True


# ============================================================================
# 异常使用场景测试
# ============================================================================

def test_exception_can_be_raised():
    """测试异常可以被raise"""
    from src.infrastructure.config.core.exceptions import ConfigLoadError
    
    with pytest.raises(ConfigLoadError) as exc_info:
        raise ConfigLoadError("Test", source="test.yaml")
    
    assert exc_info.value.source == "test.yaml"


def test_exception_catching_by_base_class():
    """测试通过基类捕获异常"""
    from src.infrastructure.config.core.exceptions import (
        ConfigException,
        ConfigLoadError
    )
    
    try:
        raise ConfigLoadError("Test")
    except ConfigException as e:
        assert isinstance(e, ConfigLoadError)

