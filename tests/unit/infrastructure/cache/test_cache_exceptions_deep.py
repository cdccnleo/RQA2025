#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Cache核心异常深度测试"""

import pytest


# ============================================================================
# CacheException基础异常测试
# ============================================================================

def test_cache_exception_basic():
    """测试CacheException基础创建"""
    from src.infrastructure.cache.core.exceptions import CacheException
    
    error = CacheException("Test error")
    assert error.message == "Test error"
    assert error.cache_key is None
    assert error.operation is None
    assert error.details == {}


def test_cache_exception_with_key():
    """测试CacheException包含缓存键"""
    from src.infrastructure.cache.core.exceptions import CacheException
    
    error = CacheException("Error", cache_key="user:123")
    assert error.cache_key == "user:123"


def test_cache_exception_with_operation():
    """测试CacheException包含操作"""
    from src.infrastructure.cache.core.exceptions import CacheException
    
    error = CacheException("Error", operation="get")
    assert error.operation == "get"


def test_cache_exception_with_details():
    """测试CacheException包含详情"""
    from src.infrastructure.cache.core.exceptions import CacheException
    
    details = {"ttl": 3600, "size": 1024}
    error = CacheException("Error", details=details)
    assert error.details == details


# ============================================================================
# CacheNotFoundError测试
# ============================================================================

def test_cache_not_found_error():
    """测试CacheNotFoundError"""
    from src.infrastructure.cache.core.exceptions import CacheNotFoundError
    
    error = CacheNotFoundError("Key not found", cache_key="missing:key")
    assert error.cache_key == "missing:key"


# ============================================================================
# CacheExpiredError测试
# ============================================================================

def test_cache_expired_error():
    """测试CacheExpiredError"""
    from src.infrastructure.cache.core.exceptions import CacheExpiredError
    
    error = CacheExpiredError("Cache expired", cache_key="expired:key", ttl=3600)
    assert error.cache_key == "expired:key"
    assert error.ttl == 3600


# ============================================================================
# CacheFullError测试
# ============================================================================

def test_cache_full_error():
    """测试CacheFullError"""
    from src.infrastructure.cache.core.exceptions import CacheFullError
    
    error = CacheFullError("Cache is full", current_size=1000, max_size=1000)
    assert error.current_size == 1000
    assert error.max_size == 1000


# ============================================================================
# CacheSerializationError测试
# ============================================================================

def test_cache_serialization_error():
    """测试CacheSerializationError"""
    from src.infrastructure.cache.core.exceptions import CacheSerializationError
    
    error = CacheSerializationError(
        "Serialization failed",
        data_type="object",
        serialization_format="json"
    )
    assert error.data_type == "object"
    assert error.serialization_format == "json"


# ============================================================================
# CacheConnectionError测试
# ============================================================================

def test_cache_connection_error():
    """测试CacheConnectionError"""
    from src.infrastructure.cache.core.exceptions import CacheConnectionError
    
    error = CacheConnectionError("Connection failed", host="localhost", port=6379)
    assert error.host == "localhost"
    assert error.port == 6379


# ============================================================================
# CacheTimeoutError测试
# ============================================================================

def test_cache_timeout_error():
    """测试CacheTimeoutError"""
    from src.infrastructure.cache.core.exceptions import CacheTimeoutError
    
    error = CacheTimeoutError("Timeout", timeout=30.0, operation="get")
    assert error.timeout == 30.0
    assert error.operation == "get"


# ============================================================================
# CacheConsistencyError测试
# ============================================================================

def test_cache_consistency_error():
    """测试CacheConsistencyError"""
    from src.infrastructure.cache.core.exceptions import CacheConsistencyError
    
    error = CacheConsistencyError(
        "Data inconsistent",
        expected_value="v1",
        actual_value="v2"
    )
    assert error.expected_value == "v1"
    assert error.actual_value == "v2"


# ============================================================================
# CacheConfigurationError测试
# ============================================================================

def test_cache_configuration_error():
    """测试CacheConfigurationError"""
    from src.infrastructure.cache.core.exceptions import CacheConfigurationError
    
    error = CacheConfigurationError(
        "Config error",
        config_key="max_size",
        expected_value=1000,
        actual_value=500
    )
    assert error.config_key == "max_size"
    assert error.expected_value == 1000
    assert error.actual_value == 500


# ============================================================================
# CachePerformanceError测试
# ============================================================================

def test_cache_performance_error():
    """测试CachePerformanceError"""
    from src.infrastructure.cache.core.exceptions import CachePerformanceError
    
    error = CachePerformanceError(
        "Performance issue",
        threshold=100.0,
        actual_value=150.0
    )
    assert error.threshold == 100.0
    assert error.actual_value == 150.0


# ============================================================================
# CacheCorruptionError测试
# ============================================================================

def test_cache_corruption_error():
    """测试CacheCorruptionError"""
    from src.infrastructure.cache.core.exceptions import CacheCorruptionError
    
    error = CacheCorruptionError("Data corrupted", corruption_type="checksum_mismatch")
    assert error.corruption_type == "checksum_mismatch"


# ============================================================================
# CacheQuotaExceededError测试
# ============================================================================

def test_cache_quota_exceeded_error():
    """测试CacheQuotaExceededError"""
    from src.infrastructure.cache.core.exceptions import CacheQuotaExceededError
    
    error = CacheQuotaExceededError(
        "Quota exceeded",
        quota_type="memory",
        current_usage=1000,
        limit=800
    )
    assert error.quota_type == "memory"
    assert error.current_usage == 1000
    assert error.limit == 800


# ============================================================================
# DistributedCacheError测试
# ============================================================================

def test_distributed_cache_error():
    """测试DistributedCacheError"""
    from src.infrastructure.cache.core.exceptions import DistributedCacheError
    
    error = DistributedCacheError(
        "Distributed error",
        node_id="node-1",
        cluster_info={"nodes": 3}
    )
    assert error.node_id == "node-1"
    assert error.cluster_info["nodes"] == 3


# ============================================================================
# CacheMigrationError测试
# ============================================================================

def test_cache_migration_error():
    """测试CacheMigrationError"""
    from src.infrastructure.cache.core.exceptions import CacheMigrationError
    
    error = CacheMigrationError(
        "Migration failed",
        source_node="node-1",
        target_node="node-2",
        migration_phase="transfer"
    )
    assert error.source_node == "node-1"
    assert error.target_node == "node-2"
    assert error.migration_phase == "transfer"


# ============================================================================
# CacheBackupError测试
# ============================================================================

def test_cache_backup_error():
    """测试CacheBackupError"""
    from src.infrastructure.cache.core.exceptions import CacheBackupError
    
    error = CacheBackupError(
        "Backup failed",
        backup_type="full",
        backup_path="/backup/cache"
    )
    assert error.backup_type == "full"
    assert error.backup_path == "/backup/cache"


# ============================================================================
# CacheRestoreError测试
# ============================================================================

def test_cache_restore_error():
    """测试CacheRestoreError"""
    from src.infrastructure.cache.core.exceptions import CacheRestoreError
    
    error = CacheRestoreError(
        "Restore failed",
        restore_type="incremental",
        restore_path="/restore/cache"
    )
    assert error.restore_type == "incremental"
    assert error.restore_path == "/restore/cache"


# ============================================================================
# 异常继承关系测试
# ============================================================================

def test_all_errors_inherit_cache_exception():
    """测试所有异常都继承自CacheException"""
    from src.infrastructure.cache.core.exceptions import (
        CacheException,
        CacheNotFoundError,
        CacheExpiredError,
        CacheFullError,
        CacheSerializationError,
        CacheConnectionError
    )
    
    assert issubclass(CacheNotFoundError, CacheException)
    assert issubclass(CacheExpiredError, CacheException)
    assert issubclass(CacheFullError, CacheException)
    assert issubclass(CacheSerializationError, CacheException)
    assert issubclass(CacheConnectionError, CacheException)


# ============================================================================
# 装饰器测试
# ============================================================================

def test_handle_cache_exception_decorator():
    """测试handle_cache_exception装饰器"""
    from src.infrastructure.cache.core.exceptions import handle_cache_exception
    
    @handle_cache_exception(operation="test")
    def test_func():
        return "success"
    
    result = test_func()
    assert result == "success"


def test_handle_cache_exception_catches_error():
    """测试装饰器捕获错误"""
    from src.infrastructure.cache.core.exceptions import (
        handle_cache_exception,
        CacheException
    )
    
    @handle_cache_exception(operation="test")
    def test_func():
        raise ValueError("Test error")
    
    with pytest.raises(CacheException):
        test_func()


def test_handle_cache_connection_exception_decorator():
    """测试handle_cache_connection_exception装饰器"""
    from src.infrastructure.cache.core.exceptions import handle_cache_connection_exception
    
    @handle_cache_connection_exception(host="localhost", port=6379)
    def connect():
        return True
    
    result = connect()
    assert result is True


def test_handle_cache_timeout_exception_decorator():
    """测试handle_cache_timeout_exception装饰器"""
    from src.infrastructure.cache.core.exceptions import handle_cache_timeout_exception
    
    @handle_cache_timeout_exception(timeout=30.0, operation="get")
    def get_value():
        return "value"
    
    result = get_value()
    assert result == "value"


def test_handle_cache_performance_exception_decorator():
    """测试handle_cache_performance_exception装饰器"""
    from src.infrastructure.cache.core.exceptions import handle_cache_performance_exception
    
    @handle_cache_performance_exception(threshold=100.0, metric="response_time")
    def measure_performance():
        return 50.0
    
    result = measure_performance()
    assert result == 50.0


# ============================================================================
# 异常使用场景测试
# ============================================================================

def test_exception_can_be_raised():
    """测试异常可以被raise"""
    from src.infrastructure.cache.core.exceptions import CacheNotFoundError
    
    with pytest.raises(CacheNotFoundError) as exc_info:
        raise CacheNotFoundError("Test", cache_key="test:key")
    
    assert exc_info.value.cache_key == "test:key"


def test_exception_catching_by_base_class():
    """测试通过基类捕获异常"""
    from src.infrastructure.cache.core.exceptions import (
        CacheException,
        CacheNotFoundError
    )
    
    try:
        raise CacheNotFoundError("Test")
    except CacheException as e:
        assert isinstance(e, CacheNotFoundError)


def test_exception_with_all_fields():
    """测试异常包含所有字段"""
    from src.infrastructure.cache.core.exceptions import CacheException
    
    error = CacheException(
        "Complete error",
        cache_key="key:123",
        operation="set",
        details={"ttl": 3600, "size": 1024}
    )
    
    assert error.message == "Complete error"
    assert error.cache_key == "key:123"
    assert error.operation == "set"
    assert error.details["ttl"] == 3600
    assert error.details["size"] == 1024

