"""
基础设施层缓存异常测试
测试缓存系统的异常类和错误处理机制
"""

import pytest
from pathlib import Path
import sys

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
src_path_str = str(project_root / "src")
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 导入异常类
from src.infrastructure.cache.exceptions.cache_exceptions import (
    CacheError,
    CacheKeyError,
    CacheValueError,
    CacheConnectionError,
    CacheTimeoutError,
    CacheCapacityError,
    CacheConsistencyError,
    CacheSerializationError,
    CacheQuotaError,
    CacheConfigurationError
)


class TestCacheExceptions:
    """缓存异常测试"""

    def test_cache_error_basic(self):
        """测试基础缓存异常"""
        message = "Cache operation failed"
        error_code = "CACHE_OP_FAILED"
        recovery_suggestion = "Check cache configuration"

        exception = CacheError(message, error_code=error_code, recovery_suggestion=recovery_suggestion)

        assert str(exception) == message
        assert exception.error_code == error_code
        assert exception.recovery_suggestion == recovery_suggestion
        assert exception.timestamp is not None

    def test_cache_key_error(self):
        """测试缓存键异常"""
        key = "invalid_key"
        operation = "get"

        exception = CacheKeyError(key=key, operation=operation)

        assert exception.key == key
        assert exception.operation == operation

    def test_cache_value_error(self):
        """测试缓存值异常"""
        key = "test_key"
        value_type = "dict"
        expected_type = "str"

        exception = CacheValueError(key=key, value_type=value_type)

        assert exception.key == key
        assert exception.value_type == value_type

    def test_cache_connection_error(self):
        """测试缓存连接异常"""
        host = "localhost"
        port = 6379
        timeout = 5.0

        exception = CacheConnectionError(host=host, port=port)

        assert exception.details['host'] == host
        assert exception.details['port'] == port

    def test_cache_timeout_error(self):
        """测试缓存超时异常"""
        operation = "set"
        timeout_seconds = 2.0
        actual_time = 3.5

        exception = CacheTimeoutError(operation=operation)

        assert exception.operation == operation

    def test_cache_capacity_error(self):
        """测试缓存容量异常"""
        capacity = 1000
        min_capacity = 100
        max_capacity = 5000

        exception = CacheCapacityError(capacity=capacity, min_capacity=min_capacity, max_capacity=max_capacity)

        assert "缓存容量" in str(exception)
        assert exception.capacity == capacity
        assert exception.min_capacity == min_capacity
        assert exception.max_capacity == max_capacity

    def test_cache_consistency_error(self):
        """测试缓存一致性异常"""
        key = "user:123"
        expected_version = "v2"
        actual_version = "v1"
        conflict_type = "version_mismatch"

        exception = CacheConsistencyError(key=key, expected_version=expected_version, actual_version=actual_version)

        assert exception.key == key
        assert exception.expected_version == expected_version
        assert exception.actual_version == actual_version

    def test_cache_serialization_error(self):
        """测试缓存序列化异常"""
        operation = "complex_object"
        data_type = "<class 'dict'>"
        format_type = "json"

        exception = CacheSerializationError(operation=operation, data_type=data_type, format=format_type)

        assert exception.operation == operation
        assert exception.data_type == data_type
        assert exception.format == format_type

    def test_cache_quota_error(self):
        """测试缓存配额异常"""
        current_usage = 900
        max_quota = 800

        exception = CacheQuotaError(current_usage=current_usage, max_quota=max_quota)

        assert "超出缓存配额" in str(exception)
        assert exception.current_usage == current_usage
        assert exception.max_quota == max_quota

    def test_cache_configuration_error(self):
        """测试缓存配置异常"""
        config_key = "cache.max_size"
        invalid_value = "invalid_number"
        expected_format = "integer > 0"

        exception = CacheConfigurationError(config_key=config_key, invalid_value=invalid_value)

        assert exception.config_key == config_key
        assert exception.invalid_value == invalid_value

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        cache_error = CacheError("test")
        assert isinstance(cache_error, Exception)

        key_error = CacheKeyError("test_key", "get")
        assert isinstance(key_error, CacheError)

        connection_error = CacheConnectionError("localhost", 6379)
        assert isinstance(connection_error, CacheError)

    def test_exception_with_details(self):
        """测试带详细信息的异常"""
        details = {
            "component": "RedisCache",
            "operation": "pipeline_execute",
            "thread_id": "Thread-123"
        }

        exception = CacheError("Operation failed", details=details, error_code="PIPELINE_ERROR")

        assert exception.details == details
        assert exception.error_code == "PIPELINE_ERROR"
        assert hasattr(exception, 'timestamp')

    def test_exception_with_cause(self):
        """测试带原因异常的异常"""
        cause = ValueError("Invalid value")
        exception = CacheError("Cache operation failed", cause=cause)

        assert exception.cause == cause
        assert isinstance(exception.cause, ValueError)
