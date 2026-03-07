"""
测试统一异常定义

覆盖 unified_exceptions.py 中的异常类和错误代码
"""

import pytest
from src.infrastructure.error.exceptions.unified_exceptions import (
    ErrorCode,
    InfrastructureError,
    DataLoaderError,
    DataProcessingError,
    DataValidationError,
    ConfigurationError,
    ConfigNotFoundError,
    ConfigInvalidError,
    NetworkError,
    ConnectionTimeoutError,
    NetworkUnavailableError,
    DatabaseError,
    DatabaseConnectionError,
    DatabaseQueryError,
    DatabaseTransactionError,
    CacheError,
    CacheMemoryError,
    CacheConnectionError,
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    PermissionDeniedError,
    SystemError,
    ResourceUnavailableError,
    PerformanceThresholdExceededError,
    CriticalError,
    WarningError,
    InfoLevelError,
    RetryableError,
    RetryError,
    TradingError,
    OrderRejectedError,
    InvalidPriceError,
    TradeError,
    CircuitBreakerOpenError
)


class TestErrorCode:
    """ErrorCode 枚举测试"""

    def test_error_code_values(self):
        """测试错误代码值"""
        # 数据相关错误
        assert ErrorCode.DATA_NOT_FOUND.value == 1001
        assert ErrorCode.DATA_INVALID.value == 1002
        assert ErrorCode.DATA_PROCESSING_ERROR.value == 1003
        assert ErrorCode.DATA_FETCH_ERROR.value == 1004
        assert ErrorCode.DATA_VALIDATION_ERROR.value == 1005

        # 配置相关错误
        assert ErrorCode.CONFIG_NOT_FOUND.value == 2001
        assert ErrorCode.CONFIG_INVALID.value == 2002
        assert ErrorCode.CONFIG_LOAD_ERROR.value == 2003

        # 网络相关错误
        assert ErrorCode.NETWORK_ERROR.value == 3001
        assert ErrorCode.CONNECTION_TIMEOUT.value == 3002

        # 数据库相关错误
        assert ErrorCode.DATABASE_ERROR.value == 4001
        assert ErrorCode.DATABASE_CONNECTION_ERROR.value == 4002

        # 缓存相关错误
        assert ErrorCode.CACHE_ERROR.value == 5001
        assert ErrorCode.CACHE_CONNECTION_ERROR.value == 5004

    def test_error_code_enum_membership(self):
        """测试错误代码枚举成员"""
        assert ErrorCode.DATA_NOT_FOUND in ErrorCode
        assert ErrorCode.NETWORK_ERROR in ErrorCode
        assert ErrorCode.DATABASE_ERROR in ErrorCode

    def test_error_code_iteration(self):
        """测试错误代码迭代"""
        codes = [member.value for member in ErrorCode]
        assert 1001 in codes  # DATA_NOT_FOUND
        assert 2001 in codes  # CONFIG_NOT_FOUND
        assert 3001 in codes  # NETWORK_CONNECTION_ERROR
        assert len(codes) > 30  # 应该有很多错误代码

    def test_error_code_ranges(self):
        """测试错误代码范围"""
        data_codes = [code.value for code in ErrorCode if 1000 <= code.value < 2000]
        config_codes = [code.value for code in ErrorCode if 2000 <= code.value < 3000]
        network_codes = [code.value for code in ErrorCode if 3000 <= code.value < 4000]

        assert len(data_codes) > 0
        assert len(config_codes) > 0
        assert len(network_codes) > 0

        # 验证范围正确性
        assert all(1000 <= code < 2000 for code in data_codes)
        assert all(2000 <= code < 3000 for code in config_codes)
        assert all(3000 <= code < 4000 for code in network_codes)


class TestInfrastructureError:
    """InfrastructureError 基类测试"""

    def test_initialization_basic(self):
        """测试基本初始化"""
        error = InfrastructureError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.error_code is None
        assert error.details is None
        assert error.context is None
        assert error.retryable == False

    def test_initialization_with_code(self):
        """测试带错误代码的初始化"""
        error = InfrastructureError(
            "Config error",
            error_code=ErrorCode.CONFIG_NOT_FOUND
        )

        assert error.message == "Config error"
        assert error.error_code == ErrorCode.CONFIG_NOT_FOUND
        assert error.details is None

    def test_initialization_complete(self):
        """测试完整初始化"""
        error = InfrastructureError(
            "Complex error",
            error_code=ErrorCode.DATA_PROCESSING_ERROR,
            details={"field": "username", "value": "invalid"},
            context={"user_id": 123},
            retryable=True
        )

        assert error.message == "Complex error"
        assert error.error_code == ErrorCode.DATA_PROCESSING_ERROR
        assert error.details == {"field": "username", "value": "invalid"}
        assert error.context == {"user_id": 123}
        assert error.retryable == True

    def test_str_representation(self):
        """测试字符串表示"""
        error = InfrastructureError("Test message")
        assert str(error) == "Test message"

    def test_repr_representation(self):
        """测试repr表示"""
        error = InfrastructureError("Test")
        repr_str = repr(error)
        assert "InfrastructureError" in repr_str
        assert "Test" in repr_str


class TestDataLoaderError:
    """DataLoaderError 测试"""

    def test_inheritance(self):
        """测试继承关系"""
        error = DataLoaderError("Data loader failed")
        assert isinstance(error, InfrastructureError)
        assert isinstance(error, Exception)

    def test_initialization(self):
        """测试初始化"""
        error = DataLoaderError(
            "File not found",
            error_code=ErrorCode.DATALOADER_FILE_NOT_FOUND,
            details={"file_path": "/data/file.csv"}
        )

        assert error.message == "File not found"
        assert error.error_code == ErrorCode.DATALOADER_FILE_NOT_FOUND
        assert error.details == {"file_path": "/data/file.csv"}


class TestDataProcessingError:
    """DataProcessingError 测试"""

    def test_initialization(self):
        """测试初始化"""
        error = DataProcessingError(
            "Data processing failed",
            error_code=ErrorCode.DATA_PROCESSING_ERROR,
            details={"operation": "transform", "record_count": 1000}
        )

        assert isinstance(error, InfrastructureError)
        assert error.message == "Data processing failed"
        assert error.error_code == ErrorCode.DATA_PROCESSING_ERROR


class TestDataValidationError:
    """DataValidationError 测试"""

    def test_initialization(self):
        """测试初始化"""
        error = DataValidationError(
            "Data validation failed",
            error_code=ErrorCode.DATA_VALIDATION_ERROR,
            details={"field": "email", "constraint": "format"}
        )

        assert isinstance(error, InfrastructureError)
        assert error.message == "Data validation failed"


class TestConfigurationError:
    """ConfigurationError 测试"""

    def test_initialization(self):
        """测试初始化"""
        error = ConfigurationError(
            "Configuration error",
            error_code=ErrorCode.CONFIG_INVALID
        )

        assert isinstance(error, InfrastructureError)
        assert error.error_code == ErrorCode.CONFIG_INVALID


class TestExceptionHierarchy:
    """异常层次结构测试"""

    def test_inheritance_chain(self):
        """测试继承链"""
        # 创建各种异常实例
        exceptions = [
            InfrastructureError("base"),
            DataLoaderError("data loader"),
            ConfigurationError("config"),
            ConfigNotFoundError("config not found"),
            NetworkError("network"),
            DatabaseError("database"),
            CacheError("cache"),
            SecurityError("security"),
            SystemError("system"),
            CriticalError("critical"),
            RetryableError("retryable"),
            TradingError("trading"),
        ]

        # 验证所有异常都是Exception的子类
        for exc in exceptions:
            assert isinstance(exc, Exception)

        # 验证InfrastructureError子类
        infrastructure_errors = [
            DataLoaderError("test"),
            ConfigurationError("test"),
            NetworkError("test"),
            DatabaseError("test"),
            CacheError("test"),
            SecurityError("test"),
            SystemError("test"),
            CriticalError("test"),
            RetryableError("test"),
            TradingError("test"),
        ]

        for exc in infrastructure_errors:
            assert isinstance(exc, InfrastructureError)

    def test_error_code_assignment(self):
        """测试错误代码分配"""
        error = InfrastructureError("test", error_code=ErrorCode.DATA_NOT_FOUND)
        assert error.error_code == ErrorCode.DATA_NOT_FOUND
        assert error.error_code.value == 1001

    def test_exception_raising_and_catching(self):
        """测试异常抛出和捕获"""
        with pytest.raises(InfrastructureError) as exc_info:
            raise InfrastructureError("Test error", error_code=ErrorCode.DATA_INVALID)

        error = exc_info.value
        assert error.message == "Test error"
        assert error.error_code == ErrorCode.DATA_INVALID

        # 测试更具体的异常类型
        with pytest.raises(ConfigNotFoundError) as exc_info:
            raise ConfigNotFoundError("Config missing", error_code=ErrorCode.CONFIG_NOT_FOUND)

        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, InfrastructureError)
        assert error.error_code == ErrorCode.CONFIG_NOT_FOUND


class TestDatabaseError:
    """DatabaseError 测试"""

    def test_initialization(self):
        """测试初始化"""
        error = DatabaseError(
            "Database error",
            error_code=ErrorCode.DATABASE_CONNECTION_ERROR,
            details={"database": "postgresql", "host": "db.example.com"}
        )

        assert isinstance(error, InfrastructureError)


class TestDatabaseConnectionError:
    """DatabaseConnectionError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "host": "db.example.com",
            "port": 5432,
            "database": "mydb"
        }
        error = DatabaseConnectionError(
            "Connection failed",
            details=details
        )

        assert isinstance(error, DatabaseError)
        assert error.details["host"] == "db.example.com"
        assert error.details["port"] == 5432
        assert error.details["database"] == "mydb"


class TestDatabaseQueryError:
    """DatabaseQueryError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "query": "SELECT * FROM users",
            "error_details": {"code": "SYNTAX_ERROR"}
        }
        error = DatabaseQueryError(
            "Query failed",
            details=details
        )

        assert isinstance(error, DatabaseError)
        assert error.details["query"] == "SELECT * FROM users"
        assert error.details["error_details"] == {"code": "SYNTAX_ERROR"}


class TestCacheError:
    """CacheError 测试"""

    def test_initialization(self):
        """测试初始化"""
        error = CacheError(
            "Cache error",
            error_code=ErrorCode.CACHE_CONNECTION_ERROR,
            details={"cache_type": "redis", "operation": "get"}
        )

        assert isinstance(error, InfrastructureError)


class TestCacheMemoryError:
    """CacheMemoryError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "current_usage": 900,
            "max_limit": 1000
        }
        error = CacheMemoryError(
            "Memory limit exceeded",
            details=details
        )

        assert isinstance(error, CacheError)
        assert error.details["current_usage"] == 900
        assert error.details["max_limit"] == 1000


class TestCacheConnectionError:
    """CacheConnectionError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "host": "redis.example.com",
            "port": 6379
        }
        error = CacheConnectionError(
            "Connection failed",
            details=details
        )

        assert isinstance(error, CacheError)
        assert error.details["host"] == "redis.example.com"
        assert error.details["port"] == 6379


class TestSecurityError:
    """SecurityError 测试"""

    def test_initialization(self):
        """测试初始化"""
        error = SecurityError(
            "Security error",
            error_code=ErrorCode.AUTHENTICATION_ERROR
        )

        assert isinstance(error, InfrastructureError)


class TestAuthenticationError:
    """AuthenticationError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {"username": "user123"}
        error = AuthenticationError(
            "Authentication failed",
            details=details
        )

        assert isinstance(error, SecurityError)
        assert error.details["username"] == "user123"


class TestAuthorizationError:
    """AuthorizationError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "resource": "admin/users",
            "action": "delete"
        }
        error = AuthorizationError(
            "Access denied",
            details=details
        )

        assert isinstance(error, SecurityError)
        assert error.details["resource"] == "admin/users"
        assert error.details["action"] == "delete"


class TestSystemError:
    """SystemError 测试"""

    def test_initialization(self):
        """测试初始化"""
        error = SystemError(
            "System error",
            error_code=ErrorCode.SYSTEM_RESOURCE_EXHAUSTED
        )

        assert isinstance(error, InfrastructureError)


class TestResourceUnavailableError:
    """ResourceUnavailableError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "resource_type": "cpu",
            "current_load": 95.5,
            "threshold": 90.0
        }
        error = ResourceUnavailableError(
            "Resource unavailable",
            details=details
        )

        assert isinstance(error, SystemError)
        assert error.details["resource_type"] == "cpu"
        assert error.details["current_load"] == 95.5
        assert error.details["threshold"] == 90.0


class TestCriticalError:
    """CriticalError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "component": "database",
            "impact": "system_down"
        }
        error = CriticalError(
            "Critical system error",
            details=details
        )

        assert isinstance(error, InfrastructureError)
        assert error.details["component"] == "database"
        assert error.details["impact"] == "system_down"


class TestRetryableError:
    """RetryableError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "retry_after_seconds": 30,
            "max_retries": 3
        }
        error = RetryableError(
            "Temporary error",
            details=details
        )

        assert isinstance(error, InfrastructureError)
        assert error.details["retry_after_seconds"] == 30
        assert error.details["max_retries"] == 3


class TestRetryError:
    """RetryError 测试"""

    def test_initialization(self):
        """测试初始化"""
        original_error = ConnectionError("Connection failed")
        details = {
            "original_error": original_error,
            "retry_count": 3,
            "total_delay": 90.5
        }
        error = RetryError(
            "Retry exhausted",
            details=details
        )

        assert isinstance(error, RetryableError)
        assert error.details["original_error"].args[0] == original_error.args[0]
        assert error.details["retry_count"] == 3
        assert error.details["total_delay"] == 90.5


class TestTradingError:
    """TradingError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "order_id": "ORD123",
            "symbol": "AAPL"
        }
        error = TradingError(
            "Trading error",
            details=details
        )

        assert isinstance(error, InfrastructureError)
        assert error.details["order_id"] == "ORD123"
        assert error.details["symbol"] == "AAPL"


class TestOrderRejectedError:
    """OrderRejectedError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "order_id": "ORD456",
            "reason": "INSUFFICIENT_FUNDS"
        }
        error = OrderRejectedError(
            "Order rejected",
            details=details
        )

        assert isinstance(error, TradingError)
        assert error.details["order_id"] == "ORD456"
        assert error.details["reason"] == "INSUFFICIENT_FUNDS"


class TestCircuitBreakerOpenError:
    """CircuitBreakerOpenError 测试"""

    def test_initialization(self):
        """测试初始化"""
        details = {
            "service_name": "payment-service",
            "failure_rate": 85.5,
            "recovery_timeout": 60
        }
        error = CircuitBreakerOpenError(
            "Circuit breaker open",
            details=details
        )

        assert isinstance(error, RetryableError)
        assert error.details["service_name"] == "payment-service"
        assert error.details["failure_rate"] == 85.5
        assert error.details["recovery_timeout"] == 60


class TestExceptionHierarchy:
    """异常层次结构测试"""

    def test_inheritance_chain(self):
        """测试继承链"""
        # 创建各种异常实例
        exceptions = [
            InfrastructureError("base"),
            DataLoaderError("data loader"),
            ConfigurationError("config"),
            ConfigNotFoundError("config not found"),
            NetworkError("network"),
            ConnectionTimeoutError("timeout", details={"host": "localhost", "port": 8080, "timeout_seconds": 10}),
            DatabaseError("database"),
            DatabaseConnectionError("db conn", details={"host": "db", "port": 5432, "database": "test"}),
            CacheError("cache"),
            CacheMemoryError("cache memory", details={"current_usage": 100, "max_limit": 200}),
            SecurityError("security"),
            AuthenticationError("auth", details={"username": "user"}),
            SystemError("system"),
            ResourceUnavailableError("resource", details={"resource_type": "memory", "current_load": 90, "threshold": 80}),
            CriticalError("critical", details={"component": "core", "impact": "high"}),
            RetryableError("retryable", details={"retry_after_seconds": 5, "max_retries": 3}),
            TradingError("trading", details={"order_id": "123", "symbol": "TEST"}),
        ]

        # 验证所有异常都是Exception的子类
        for exc in exceptions:
            assert isinstance(exc, Exception)

        # 验证InfrastructureError子类
        infrastructure_errors = [
            DataLoaderError("test"),
            ConfigurationError("test"),
            NetworkError("test"),
            DatabaseError("test"),
            CacheError("test"),
            SecurityError("test"),
            SystemError("test"),
            CriticalError("test", details={"component": "test", "impact": "test"}),
            RetryableError("test", details={"retry_after_seconds": 1, "max_retries": 1}),
            TradingError("test", details={"order_id": "test", "symbol": "test"}),
        ]

        for exc in infrastructure_errors:
            assert isinstance(exc, InfrastructureError)

    def test_error_code_assignment(self):
        """测试错误代码分配"""
        error = InfrastructureError("test", error_code=ErrorCode.DATA_NOT_FOUND)
        assert error.error_code == ErrorCode.DATA_NOT_FOUND
        assert error.error_code.value == 1001

    def test_exception_raising_and_catching(self):
        """测试异常抛出和捕获"""
        with pytest.raises(InfrastructureError) as exc_info:
            raise InfrastructureError("Test error", error_code=ErrorCode.DATA_INVALID)

        error = exc_info.value
        assert error.message == "Test error"
        assert error.error_code == ErrorCode.DATA_INVALID

        # 测试更具体的异常类型
        with pytest.raises(ConfigNotFoundError) as exc_info:
            raise ConfigNotFoundError("Config missing")

        error = exc_info.value
        assert isinstance(error, ConfigurationError)
        assert isinstance(error, InfrastructureError)

    def test_error_details_preservation(self):
        """测试错误详情保留"""
        original_details = {"user_id": 123, "action": "login", "timestamp": "2024-01-01"}
        error = InfrastructureError(
            "Authentication failed",
            error_code=ErrorCode.AUTHENTICATION_ERROR,
            details=original_details
        )

        assert error.details == original_details
        assert error.details is not original_details  # 应该是深拷贝

    def test_cause_chain(self):
        """测试原因链"""
        root_cause = ValueError("Invalid input")
        mid_cause = InfrastructureError("Processing failed", details={"cause": root_cause})
        top_error = CriticalError("System down", details={"cause": mid_cause})

        # 检查原因链是否正确存储和访问
        assert top_error.details["cause"].message == "Processing failed"
        assert top_error.details["cause"].details["cause"].args[0] == root_cause.args[0]
        assert isinstance(top_error.details["cause"].details["cause"], ValueError)
