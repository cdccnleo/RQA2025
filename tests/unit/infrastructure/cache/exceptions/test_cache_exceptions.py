"""
基础设施缓存层异常模块测试
"""

import pytest
from datetime import datetime
from src.infrastructure.cache.exceptions.cache_exceptions import *


class TestCacheExceptions:
    """测试基础设施缓存层异常模块"""

    def test_cache_error_inheritance(self):
        """测试CacheError继承关系"""
        assert issubclass(CacheError, Exception)

    def test_cache_error_creation_basic(self):
        """测试CacheError基本创建"""
        error = CacheError("测试错误")
        assert str(error) == "测试错误"
        assert error.error_code == "CACHE_ERROR"
        assert isinstance(error.details, dict)
        assert isinstance(error.timestamp, datetime)
        assert error.recovery_suggestion == ""

    def test_cache_error_creation_full(self):
        """测试CacheError完整创建"""
        details = {"key": "value"}
        error = CacheError(
            message="完整错误",
            error_code="TEST_ERROR",
            details=details,
            recovery_suggestion="请重试",
            extra_param="extra_value"
        )

        assert str(error) == "完整错误"
        assert error.error_code == "TEST_ERROR"
        assert error.details["key"] == "value"
        assert error.details["extra_param"] == "extra_value"
        assert error.recovery_suggestion == "请重试"

    def test_cache_error_to_dict(self):
        """测试CacheError转换为字典"""
        error = CacheError("测试错误", error_code="TEST_ERROR")
        error_dict = error.to_dict()

        assert isinstance(error_dict, dict)
        assert error_dict["error_code"] == "TEST_ERROR"
        assert error_dict["message"] == "测试错误"
        assert "timestamp" in error_dict

    def test_cache_error_create_error(self):
        """测试CacheError.create_error方法"""
        error = CacheError.create_error(
            message="创建的错误",
            error_code="CREATED_ERROR",
            recovery_suggestion="请检查配置",
            extra_detail="extra"
        )

        assert isinstance(error, CacheError)
        assert error.message == "创建的错误"
        assert error.error_code == "CREATED_ERROR"
        assert error.recovery_suggestion == "请检查配置"
        assert error.details["extra_detail"] == "extra"

    def test_cache_connection_error_creation(self):
        """测试CacheConnectionError创建"""
        error = CacheConnectionError(
            host="localhost",
            port=6379,
            retry_count=3
        )

        assert isinstance(error, CacheError)
        assert error.error_code == "CONNECTION_ERROR"
        assert error.host == "localhost"
        assert error.port == 6379
        assert error.retry_count == 3
        assert error.should_retry == True
        assert "host" in error.details
        assert "port" in error.details

    def test_cache_connection_error_defaults(self):
        """测试CacheConnectionError默认值"""
        error = CacheConnectionError()
        assert error.message == "无法连接到缓存服务器"
        assert error.error_code == "CONNECTION_ERROR"
        assert error.recovery_suggestion == "请检查网络连接、服务器状态和认证信息"

    def test_cache_serialization_error_creation(self):
        """测试CacheSerializationError创建"""
        error = CacheSerializationError(
            operation="serialize",
            data_type="dict",
            format="json"
        )

        assert isinstance(error, CacheError)
        assert error.error_code == "SERIALIZATION_ERROR"
        assert error.operation == "serialize"
        assert error.data_type == "dict"
        assert error.format == "json"

    def test_cache_key_error_creation(self):
        """测试CacheKeyError创建"""
        error = CacheKeyError(
            key="test_key",
            operation="get",
            key_type="string"
        )

        assert isinstance(error, CacheError)
        assert error.error_code == "KEY_ERROR"
        assert error.key == "test_key"
        assert error.operation == "get"
        assert error.key_type == "string"

    def test_cache_timeout_error_creation(self):
        """测试CacheTimeoutError创建"""
        error = CacheTimeoutError(
            operation="get",
            timeout_ms=5000,
            retryable=True
        )

        assert isinstance(error, CacheError)
        assert error.error_code == "TIMEOUT_ERROR"
        assert error.operation == "get"
        assert error.timeout_ms == 5000
        assert error.retryable == True

    def test_cache_consistency_error_creation(self):
        """测试CacheConsistencyError创建"""
        error = CacheConsistencyError(
            key="test_key",
            expected_version="v1",
            actual_version="v2",
            keys=["key1", "key2"]
        )

        assert isinstance(error, CacheError)
        assert error.error_code == "CONSISTENCY_ERROR"
        assert error.key == "test_key"
        assert error.expected_version == "v1"
        assert error.actual_version == "v2"
        assert error.keys == ["key1", "key2"]

    def test_cache_quota_error_creation(self):
        """测试CacheQuotaError创建"""
        error = CacheQuotaError(
            current_usage=80,
            max_quota=100
        )

        assert isinstance(error, CacheError)
        assert error.error_code == "QUOTA_ERROR"
        assert error.current_usage == 80
        assert error.max_quota == 100

    def test_cache_capacity_error_creation(self):
        """测试CacheCapacityError创建"""
        error = CacheCapacityError(
            capacity=1000,
            min_capacity=100,
            max_capacity=10000
        )

        assert isinstance(error, CacheError)
        assert error.error_code == "CAPACITY_ERROR"
        assert error.capacity == 1000
        assert error.min_capacity == 100
        assert error.max_capacity == 10000

    def test_cache_configuration_error_creation(self):
        """测试CacheConfigurationError创建"""
        error = CacheConfigurationError(
            config_key="max_size",
            invalid_value=-1,
            config_value="invalid"
        )

        assert isinstance(error, CacheError)
        assert error.error_code == "CONFIGURATION_ERROR"
        assert error.config_key == "max_size"
        assert error.invalid_value == -1
        assert error.config_value == "invalid"

    def test_cache_value_error_creation(self):
        """测试CacheValueError创建"""
        error = CacheValueError(
            value="test_value",
            value_type="string",
            size_limit=1024
        )

        assert isinstance(error, CacheError)
        assert error.error_code == "VALUE_ERROR"
        assert error.value == "test_value"
        assert error.value_type == "string"
        assert error.size_limit == 1024

    def test_exception_hierarchy(self):
        """测试异常继承层次"""
        # 测试所有异常都是CacheError的子类
        exceptions = [
            CacheConnectionError,
            CacheSerializationError,
            CacheKeyError,
            CacheTimeoutError,
            CacheConsistencyError,
            CacheQuotaError,
            CacheCapacityError,
            CacheConfigurationError,
            CacheValueError
        ]

        for exc_class in exceptions:
            assert issubclass(exc_class, CacheError)

    def test_exception_attributes_preserved(self):
        """测试异常属性保留"""
        error = CacheConnectionError(host="test.com", port=8080)
        # 确保基类属性仍然存在
        assert hasattr(error, 'message')
        assert hasattr(error, 'error_code')
        assert hasattr(error, 'details')
        assert hasattr(error, 'timestamp')
        # 确保子类特有属性存在
        assert hasattr(error, 'host')
        assert hasattr(error, 'port')
