#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
缓存核心异常测试

测试缓存系统core模块的异常类和错误处理机制。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.cache.core.exceptions import (
    CacheException, CacheNotFoundError, CacheExpiredError, CacheFullError,
    CacheSerializationError, CacheConnectionError, CacheTimeoutError,
    CacheConsistencyError, CacheConfigurationError, CachePerformanceError,
    CacheCorruptionError, CacheQuotaExceededError, DistributedCacheError,
    CacheMigrationError, CacheBackupError, CacheRestoreError
)


class TestCoreCacheExceptions:
    """缓存核心异常测试类"""

    def test_cache_exception_basic(self):
        """测试基础缓存异常"""
        exc = CacheException("Test message")
        assert str(exc) == "Test message"
        assert exc.cache_key is None
        assert exc.operation is None
        assert exc.details == {}
        assert exc.message == "Test message"

    def test_cache_exception_with_params(self):
        """测试带参数的缓存异常"""
        details = {"key": "value"}
        exc = CacheException(
            "Test message",
            cache_key="test_key",
            operation="get",
            details=details
        )
        assert exc.cache_key == "test_key"
        assert exc.operation == "get"
        assert exc.details == details

    def test_cache_not_found_error(self):
        """测试缓存未找到错误"""
        exc = CacheNotFoundError("Key not found", cache_key="missing_key")
        assert isinstance(exc, CacheException)
        assert exc.cache_key == "missing_key"
        assert "Key not found" in str(exc)

    def test_cache_expired_error(self):
        """测试缓存过期错误"""
        exc = CacheExpiredError("Cache expired", cache_key="expired_key", ttl=300)
        assert isinstance(exc, CacheException)
        assert exc.cache_key == "expired_key"
        assert exc.ttl == 300

    def test_cache_expired_error_without_ttl(self):
        """测试缓存过期错误（无TTL）"""
        exc = CacheExpiredError("Cache expired", cache_key="expired_key")
        assert exc.ttl is None

    def test_cache_full_error(self):
        """测试缓存满错误"""
        exc = CacheFullError("Cache is full", current_size=1000, max_size=1000)
        assert isinstance(exc, CacheException)
        assert exc.current_size == 1000
        assert exc.max_size == 1000

    def test_cache_full_error_without_sizes(self):
        """测试缓存满错误（无大小信息）"""
        exc = CacheFullError("Cache is full")
        assert exc.current_size is None
        assert exc.max_size is None

    def test_cache_serialization_error(self):
        """测试缓存序列化错误"""
        exc = CacheSerializationError("Serialization failed", cache_key="complex_obj")
        assert isinstance(exc, CacheException)
        assert exc.cache_key == "complex_obj"

    def test_cache_connection_error(self):
        """测试缓存连接错误"""
        exc = CacheConnectionError("Connection failed", operation="connect")
        assert isinstance(exc, CacheException)
        assert exc.operation == "connect"

    def test_cache_timeout_error(self):
        """测试缓存超时错误"""
        exc = CacheTimeoutError("Operation timed out", cache_key="slow_key", operation="get")
        assert isinstance(exc, CacheException)
        assert exc.cache_key == "slow_key"
        assert exc.operation == "get"

    def test_cache_consistency_error(self):
        """测试缓存一致性错误"""
        details = {"expected": "value1", "actual": "value2"}
        exc = CacheConsistencyError("Data inconsistency", cache_key="inconsistent_key", details=details)
        assert isinstance(exc, CacheException)
        assert exc.details == details

    def test_cache_configuration_error(self):
        """测试缓存配置错误"""
        exc = CacheConfigurationError("Invalid configuration", config_key="cache.size", operation="init")
        assert isinstance(exc, CacheException)
        assert exc.operation == "init"

    def test_cache_performance_error(self):
        """测试缓存性能错误"""
        exc = CachePerformanceError("Performance degraded", cache_key="slow_key")
        assert isinstance(exc, CacheException)

    def test_cache_corruption_error(self):
        """测试缓存损坏错误"""
        exc = CacheCorruptionError("Data corruption detected", cache_key="corrupted_key")
        assert isinstance(exc, CacheException)

    def test_cache_quota_exceeded_error(self):
        """测试缓存配额超限错误"""
        exc = CacheQuotaExceededError("Quota exceeded", cache_key="large_key")
        assert isinstance(exc, CacheException)

    def test_distributed_cache_error(self):
        """测试分布式缓存错误"""
        exc = DistributedCacheError("Distributed operation failed", operation="sync")
        assert isinstance(exc, CacheException)
        assert exc.operation == "sync"

    def test_cache_migration_error(self):
        """测试缓存迁移错误"""
        exc = CacheMigrationError("Migration failed", operation="migrate")
        assert isinstance(exc, CacheException)

    def test_cache_backup_error(self):
        """测试缓存备份错误"""
        exc = CacheBackupError("Backup failed", operation="backup")
        assert isinstance(exc, CacheException)

    def test_cache_restore_error(self):
        """测试缓存恢复错误"""
        exc = CacheRestoreError("Restore failed", operation="restore")
        assert isinstance(exc, CacheException)

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        # 确保所有异常都继承自CacheException
        exceptions = [
            CacheNotFoundError, CacheExpiredError, CacheFullError,
            CacheSerializationError, CacheConnectionError, CacheTimeoutError,
            CacheConsistencyError, CacheConfigurationError, CachePerformanceError,
            CacheCorruptionError, CacheQuotaExceededError, DistributedCacheError,
            CacheMigrationError, CacheBackupError, CacheRestoreError
        ]

        for exc_class in exceptions:
            exc = exc_class("test")
            assert isinstance(exc, CacheException)
            assert isinstance(exc, Exception)

    def test_exception_messages(self):
        """测试异常消息"""
        test_message = "Test error message"
        exc = CacheException(test_message)
        assert str(exc) == test_message

    def test_exception_details_persistence(self):
        """测试异常详情持久化"""
        details = {"param1": "value1", "param2": 42, "param3": [1, 2, 3]}
        exc = CacheException("test", details=details)
        assert exc.details == details

    def test_exception_details_empty_by_default(self):
        """测试异常详情默认为空"""
        exc = CacheException("test")
        assert exc.details == {}

    def test_exception_with_complex_details(self):
        """测试异常的复杂详情"""
        complex_details = {
            "nested": {"key": "value"},
            "list": [1, 2, {"complex": "object"}],
            "numbers": [1.5, 2.7, 3.14]
        }
        exc = CacheException("Complex test", details=complex_details)
        assert exc.details == complex_details

    def test_exception_cache_key_none_by_default(self):
        """测试异常缓存键默认为空"""
        exc = CacheException("test")
        assert exc.cache_key is None

    def test_exception_operation_none_by_default(self):
        """测试异常操作默认为空"""
        exc = CacheException("test")
        assert exc.operation is None

    def test_exception_with_unicode_message(self):
        """测试异常Unicode消息"""
        unicode_message = "缓存错误：测试消息"
        exc = CacheException(unicode_message)
        assert str(exc) == unicode_message

    def test_exception_with_unicode_cache_key(self):
        """测试异常Unicode缓存键"""
        unicode_key = "测试键"
        exc = CacheException("test", cache_key=unicode_key)
        assert exc.cache_key == unicode_key
