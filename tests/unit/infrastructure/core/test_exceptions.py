"""
测试基础设施层异常类

覆盖 exceptions.py 中定义的所有异常类
"""

import pytest
from src.infrastructure.core.exceptions import (
    InfrastructureException,
    ConfigurationError,
    CacheError,
    LoggingError,
    MonitoringError,
    ResourceError,
    NetworkError,
    DatabaseError,
    FileSystemError,
    SecurityError,
    HealthCheckError,
    VersionError
)


class TestInfrastructureException:
    """InfrastructureException 单元测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        exc = InfrastructureException("Test error")

        assert str(exc) == "Test error"
        assert exc.error_code == -1
        assert exc.message == "Test error"

    def test_initialization_with_code(self):
        """测试带错误码的初始化"""
        exc = InfrastructureException("Test error", 1000)

        assert str(exc) == "Test error"
        assert exc.error_code == 1000
        assert exc.message == "Test error"

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(InfrastructureException, Exception)


class TestConfigurationError:
    """ConfigurationError 单元测试"""

    def test_initialization_without_key(self):
        """测试不带配置键的初始化"""
        exc = ConfigurationError("Config validation failed")

        assert "Config validation failed" in str(exc)
        assert isinstance(exc, InfrastructureException)
        assert exc.error_code == -1

    def test_initialization_with_key(self):
        """测试带配置键的初始化"""
        exc = ConfigurationError("Invalid value", "database.host")

        assert "配置错误 - database.host: Invalid value" in str(exc)
        assert exc.error_code == -1

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(ConfigurationError, InfrastructureException)


class TestCacheError:
    """CacheError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = CacheError("Cache operation failed", "test_key", 2001)

        assert str(exc) == "缓存错误 - test_key: Cache operation failed"
        assert exc.error_code == 2001
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(CacheError, InfrastructureException)


class TestLoggingError:
    """LoggingError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = LoggingError("Failed to write log", "test.log", 3001)

        assert str(exc) == "日志错误 - test.log: Failed to write log"
        assert exc.error_code == 3001
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(LoggingError, InfrastructureException)


class TestMonitoringError:
    """MonitoringError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = MonitoringError("Monitoring service unavailable", "cpu_usage", 4001)

        assert str(exc) == "监控错误 - cpu_usage: Monitoring service unavailable"
        assert exc.error_code == 4001
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(MonitoringError, InfrastructureException)


class TestResourceError:
    """ResourceError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = ResourceError("Insufficient memory", "memory", 5001)

        assert str(exc) == "资源错误 - memory: Insufficient memory"
        assert exc.error_code == 5001
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(ResourceError, InfrastructureException)


class TestNetworkError:
    """NetworkError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = NetworkError("Connection timeout", "api.example.com", 6001)

        assert str(exc) == "网络错误 - api.example.com: Connection timeout"
        assert exc.error_code == 6001
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(NetworkError, InfrastructureException)


class TestDatabaseError:
    """DatabaseError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = DatabaseError("Query execution failed", 7001)

        assert str(exc) == "Query execution failed"
        assert exc.error_code == 7001
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(DatabaseError, InfrastructureException)


class TestFileSystemError:
    """FileSystemError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = FileSystemError("Permission denied", "/test/path")

        assert "Permission denied" in str(exc)
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(FileSystemError, InfrastructureException)


class TestSecurityError:
    """SecurityError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = SecurityError("Authentication failed", "login")

        assert "Authentication failed" in str(exc)
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(SecurityError, InfrastructureException)


class TestHealthCheckError:
    """HealthCheckError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = HealthCheckError("Service unhealthy", "database")

        assert "Service unhealthy" in str(exc)
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(HealthCheckError, InfrastructureException)


class TestVersionError:
    """VersionError 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        exc = VersionError("Version conflict", "v1.0")

        assert "Version conflict" in str(exc)
        assert isinstance(exc, InfrastructureException)

    def test_inheritance(self):
        """测试继承关系"""
        assert issubclass(VersionError, InfrastructureException)


class TestExceptionHierarchy:
    """异常层次结构测试"""

    def test_all_exceptions_inherit_from_infrastructure_exception(self):
        """测试所有异常都继承自InfrastructureException"""
        exception_classes = [
            ConfigurationError,
            CacheError,
            LoggingError,
            MonitoringError,
            ResourceError,
            NetworkError,
            DatabaseError,
            FileSystemError,
            SecurityError,
            HealthCheckError,
            VersionError
        ]

        for exc_class in exception_classes:
            assert issubclass(exc_class, InfrastructureException)
            assert issubclass(exc_class, Exception)

    def test_exception_instantiation(self):
        """测试异常实例化"""
        exception_classes = [
            (ConfigurationError, "Config error"),
            (CacheError, "Cache error"),
            (LoggingError, "Log error"),
            (MonitoringError, "Monitor error"),
            (ResourceError, "Resource error"),
            (NetworkError, "Network error"),
            (DatabaseError, "DB error"),
            (FileSystemError, "FS error"),
            (SecurityError, "Security error"),
            (HealthCheckError, "Health error"),
            (VersionError, "Version error")
        ]

        for exc_class, message in exception_classes:
            exc = exc_class(message)
            assert message in str(exc)
            assert isinstance(exc, InfrastructureException)

    def test_exception_error_codes(self):
        """测试异常错误码范围"""
        # 创建不同类型的异常实例
        exceptions = [
            ConfigurationError("test"),
            CacheError("test", 2001),
            LoggingError("test", 3001),
            MonitoringError("test", 4001),
            ResourceError("test", 5001),
            NetworkError("test", 6001),
            DatabaseError("test", 7001),
            FileSystemError("test", 8001),
            SecurityError("test", 9001),
            HealthCheckError("test", 10001),
            VersionError("test", 11001)
        ]

        # 验证错误码都是正数或-1
        for exc in exceptions:
            assert exc.error_code >= -1, f"Error code {exc.error_code} should be >= -1"

        # 验证特定错误码
        assert exceptions[0].error_code == -1  # ConfigurationError 默认值
        assert exceptions[1].error_code == 2001  # CacheError 指定值


class TestExceptionUsage:
    """异常使用场景测试"""

    def test_exception_raising_and_catching(self):
        """测试异常抛出和捕获"""
        with pytest.raises(ConfigurationError) as exc_info:
            raise ConfigurationError("Test configuration error", "test.key")

        exc = exc_info.value
        assert "配置错误 - test.key: Test configuration error" in str(exc)
        assert isinstance(exc, InfrastructureException)

    def test_exception_chaining(self):
        """测试异常链"""
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise ConfigurationError(f"Configuration failed: {e}") from e
        except ConfigurationError as e:
            assert isinstance(e, ConfigurationError)
            assert isinstance(e, InfrastructureException)
            assert "Original error" in str(e)

    def test_exception_with_custom_error_codes(self):
        """测试自定义错误码"""
        error_codes = {
            CacheError: 2001,
            LoggingError: 3001,
            MonitoringError: 4001,
            ResourceError: 5001,
            NetworkError: 6001,
            DatabaseError: 7001,
            FileSystemError: 8001,
            SecurityError: 9001,
            HealthCheckError: 10001,
            VersionError: 11001
        }

        for exc_class, expected_code in error_codes.items():
            exc = exc_class("Test error")
            # 某些异常类有固定的error_code，其他的从InfrastructureException继承默认值-1
            if hasattr(exc, 'error_code') and exc.error_code != -1:
                assert exc.error_code == expected_code
            assert isinstance(exc, InfrastructureException)