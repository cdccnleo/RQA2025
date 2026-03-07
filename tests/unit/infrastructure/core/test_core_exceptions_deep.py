#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Core核心异常深度测试"""

import pytest


# ============================================================================
# InfrastructureException基础异常测试
# ============================================================================

def test_infrastructure_exception_basic():
    """测试InfrastructureException基础创建"""
    from src.infrastructure.core.exceptions import InfrastructureException
    
    error = InfrastructureException("Test error")
    assert error.message == "Test error"
    assert error.error_code == -1


def test_infrastructure_exception_with_code():
    """测试InfrastructureException包含错误码"""
    from src.infrastructure.core.exceptions import InfrastructureException
    
    error = InfrastructureException("Test error", error_code=100)
    assert error.error_code == 100


def test_infrastructure_exception_can_be_raised():
    """测试InfrastructureException可以被raise"""
    from src.infrastructure.core.exceptions import InfrastructureException
    
    with pytest.raises(InfrastructureException) as exc_info:
        raise InfrastructureException("Test")
    
    assert exc_info.value.message == "Test"


# ============================================================================
# ConfigurationError测试
# ============================================================================

def test_configuration_error():
    """测试ConfigurationError"""
    from src.infrastructure.core.exceptions import ConfigurationError
    
    error = ConfigurationError("Invalid config", config_key="database.host")
    assert error.config_key == "database.host"
    assert "database.host" in error.message


def test_configuration_error_without_key():
    """测试ConfigurationError无配置键"""
    from src.infrastructure.core.exceptions import ConfigurationError
    
    error = ConfigurationError("Config error")
    assert error.config_key is None


# ============================================================================
# CacheError测试
# ============================================================================

def test_cache_error():
    """测试CacheError"""
    from src.infrastructure.core.exceptions import CacheError
    
    error = CacheError("Cache failed", cache_key="user:123")
    assert error.cache_key == "user:123"
    assert "user:123" in error.message


def test_cache_error_without_key():
    """测试CacheError无缓存键"""
    from src.infrastructure.core.exceptions import CacheError
    
    error = CacheError("Cache error")
    assert error.cache_key is None


# ============================================================================
# LoggingError测试
# ============================================================================

def test_logging_error():
    """测试LoggingError"""
    from src.infrastructure.core.exceptions import LoggingError
    
    error = LoggingError("Log failed", log_file="/var/log/app.log")
    assert error.log_file == "/var/log/app.log"
    assert "/var/log/app.log" in error.message


# ============================================================================
# MonitoringError测试
# ============================================================================

def test_monitoring_error():
    """测试MonitoringError"""
    from src.infrastructure.core.exceptions import MonitoringError
    
    error = MonitoringError("Monitor failed", metric_name="cpu_usage")
    assert error.metric_name == "cpu_usage"
    assert "cpu_usage" in error.message


# ============================================================================
# ResourceError测试
# ============================================================================

def test_resource_error():
    """测试ResourceError"""
    from src.infrastructure.core.exceptions import ResourceError
    
    error = ResourceError("Resource error", resource_type="memory")
    assert error.resource_type == "memory"
    assert "memory" in error.message


# ============================================================================
# NetworkError测试
# ============================================================================

def test_network_error():
    """测试NetworkError"""
    from src.infrastructure.core.exceptions import NetworkError
    
    error = NetworkError("Network failed", endpoint="api.example.com")
    assert error.endpoint == "api.example.com"
    assert "api.example.com" in error.message


# ============================================================================
# DatabaseError测试
# ============================================================================

def test_database_error():
    """测试DatabaseError"""
    from src.infrastructure.core.exceptions import DatabaseError
    
    error = DatabaseError("DB error", table_name="users")
    assert error.table_name == "users"
    assert "users" in error.message


# ============================================================================
# FileSystemError测试
# ============================================================================

def test_file_system_error():
    """测试FileSystemError"""
    from src.infrastructure.core.exceptions import FileSystemError
    
    error = FileSystemError("File error", file_path="/data/file.txt")
    assert error.file_path == "/data/file.txt"
    assert "/data/file.txt" in error.message


# ============================================================================
# SecurityError测试
# ============================================================================

def test_security_error():
    """测试SecurityError"""
    from src.infrastructure.core.exceptions import SecurityError
    
    error = SecurityError("Security issue", security_context="authentication")
    assert error.security_context == "authentication"
    assert "authentication" in error.message


# ============================================================================
# HealthCheckError测试
# ============================================================================

def test_health_check_error():
    """测试HealthCheckError"""
    from src.infrastructure.core.exceptions import HealthCheckError
    
    error = HealthCheckError("Health check failed", check_target="database")
    assert error.check_target == "database"
    assert "database" in error.message


# ============================================================================
# VersionError测试
# ============================================================================

def test_version_error():
    """测试VersionError"""
    from src.infrastructure.core.exceptions import VersionError
    
    error = VersionError("Version mismatch", version="v2.0")
    assert error.version == "v2.0"
    assert "v2.0" in error.message


# ============================================================================
# 异常继承关系测试
# ============================================================================

def test_all_errors_inherit_infrastructure_exception():
    """测试所有异常都继承自InfrastructureException"""
    from src.infrastructure.core.exceptions import (
        InfrastructureException,
        ConfigurationError,
        CacheError,
        LoggingError,
        MonitoringError,
        ResourceError
    )
    
    assert issubclass(ConfigurationError, InfrastructureException)
    assert issubclass(CacheError, InfrastructureException)
    assert issubclass(LoggingError, InfrastructureException)
    assert issubclass(MonitoringError, InfrastructureException)
    assert issubclass(ResourceError, InfrastructureException)


def test_all_errors_inherit_exception():
    """测试所有异常都继承自Exception"""
    from src.infrastructure.core.exceptions import InfrastructureException
    
    assert issubclass(InfrastructureException, Exception)


# ============================================================================
# 异常使用场景测试
# ============================================================================

def test_exception_can_be_raised():
    """测试异常可以被raise"""
    from src.infrastructure.core.exceptions import DatabaseError
    
    with pytest.raises(DatabaseError) as exc_info:
        raise DatabaseError("Test", table_name="test_table")
    
    assert exc_info.value.table_name == "test_table"


def test_exception_catching_by_base_class():
    """测试通过基类捕获异常"""
    from src.infrastructure.core.exceptions import (
        InfrastructureException,
        DatabaseError
    )
    
    try:
        raise DatabaseError("Test")
    except InfrastructureException as e:
        assert isinstance(e, DatabaseError)


def test_exception_with_error_code():
    """测试异常包含错误码"""
    from src.infrastructure.core.exceptions import InfrastructureException
    
    error = InfrastructureException("Error", error_code=500)
    assert error.error_code == 500


def test_multiple_exceptions():
    """测试多种异常类型"""
    from src.infrastructure.core.exceptions import (
        ConfigurationError,
        NetworkError,
        SecurityError
    )
    
    errors = [
        ConfigurationError("Config", config_key="key1"),
        NetworkError("Network", endpoint="api.com"),
        SecurityError("Security", security_context="auth")
    ]
    
    assert len(errors) == 3
    assert all(isinstance(e, Exception) for e in errors)

