"""
测试日志核心异常类

覆盖 exceptions.py 中的所有异常类和错误处理
"""

import pytest
from src.infrastructure.logging.core.exceptions import (
    LoggingException,
    LogConfigurationError,
    LogHandlerError,
    LogFormatterError,
    LogFileError,
    LogRotationError,
    LogCompressionError,
    LogNetworkError,
    LogQueueError,
    LogFilterError,
    LogSecurityError,
    LogMonitorError,
    LogPerformanceError,
    LogStorageError,
    LogAsyncError,
    LogBatchError,
    LogValidationError,
    LogTimeoutError,
    ResourceError,
    HTTP_OK,
    HTTP_BAD_REQUEST,
    HTTP_UNAUTHORIZED,
    HTTP_FORBIDDEN,
    HTTP_NOT_FOUND,
    HTTP_INTERNAL_ERROR,
    DEFAULT_PAGE_SIZE,
    MAX_PAGE_SIZE
)


class TestHTTPConstants:
    """HTTP常量测试"""

    def test_http_status_codes(self):
        """测试HTTP状态码常量"""
        assert HTTP_OK == 200
        assert HTTP_BAD_REQUEST == 400
        assert HTTP_UNAUTHORIZED == 401
        assert HTTP_FORBIDDEN == 403
        assert HTTP_NOT_FOUND == 404
        assert HTTP_INTERNAL_ERROR == 500

    def test_pagination_constants(self):
        """测试分页常量"""
        assert DEFAULT_PAGE_SIZE == 20
        assert MAX_PAGE_SIZE == 100
        assert MAX_PAGE_SIZE >= DEFAULT_PAGE_SIZE


class TestLoggingException:
    """LoggingException 测试"""

    def test_basic_exception(self):
        """测试基础异常"""
        exc = LoggingException("Test error")
        assert str(exc) == "Test error"
        assert isinstance(exc, Exception)

    def test_exception_with_details(self):
        """测试带详细信息的异常"""
        details = {"key": "value", "code": 500}
        exc = LoggingException("Test error", logger_name="test_logger",
                              log_level="ERROR", details=details)

        assert exc.logger_name == "test_logger"
        assert exc.log_level == "ERROR"
        assert exc.details == details
        assert exc.message == "Test error"

    def test_exception_with_kwargs(self):
        """测试带额外参数的异常"""
        exc = LoggingException("Test error", extra_param="extra_value",
                              another_param=123)

        assert exc.details["extra_param"] == "extra_value"
        assert exc.details["another_param"] == 123


class TestLogConfigurationError:
    """LogConfigurationError 测试"""

    def test_config_error_basic(self):
        """测试配置错误基础功能"""
        exc = LogConfigurationError("Config error")
        assert isinstance(exc, LoggingException)
        assert str(exc) == "Config error"

    def test_config_error_with_details(self):
        """测试配置错误详细信息"""
        exc = LogConfigurationError("Config error", config_key="log_level",
                                   expected_value="INFO", actual_value="INVALID")

        assert exc.config_key == "log_level"
        assert exc.expected_value == "INFO"
        assert exc.actual_value == "INVALID"


class TestLogHandlerError:
    """LogHandlerError 测试"""

    def test_handler_error(self):
        """测试处理器错误"""
        exc = LogHandlerError("Handler error", handler_type="file")
        assert isinstance(exc, LoggingException)
        assert exc.handler_type == "file"


class TestLogFormatterError:
    """LogFormatterError 测试"""

    def test_formatter_error(self):
        """测试格式化器错误"""
        exc = LogFormatterError("Formatter error", formatter_type="json")
        assert isinstance(exc, LoggingException)
        assert exc.formatter_type == "json"


class TestLogFileError:
    """LogFileError 测试"""

    def test_file_error(self):
        """测试文件错误"""
        exc = LogFileError("File error", file_path="/var/log/app.log")
        assert isinstance(exc, LoggingException)
        assert exc.file_path == "/var/log/app.log"


class TestLogRotationError:
    """LogRotationError 测试"""

    def test_rotation_error(self):
        """测试轮转错误"""
        exc = LogRotationError("Rotation error", rotation_type="time")
        assert isinstance(exc, LogFileError)
        assert exc.rotation_type == "time"


class TestLogCompressionError:
    """LogCompressionError 测试"""

    def test_compression_error(self):
        """测试压缩错误"""
        exc = LogCompressionError("Compression error", compression_type="gzip",
                                 original_size=1000)
        assert isinstance(exc, LogFileError)
        assert exc.details["compression_type"] == "gzip"
        assert exc.details["original_size"] == 1000


class TestLogNetworkError:
    """LogNetworkError 测试"""

    def test_network_error(self):
        """测试网络错误"""
        exc = LogNetworkError("Network error", host="localhost", port=514,
                             protocol="tcp")
        assert isinstance(exc, LoggingException)
        assert exc.host == "localhost"
        assert exc.port == 514
        assert exc.protocol == "tcp"


class TestLogQueueError:
    """LogQueueError 测试"""

    def test_queue_error(self):
        """测试队列错误"""
        exc = LogQueueError("Queue error", queue_size=1000, current_size=500,
                           queue_name="log_queue", max_size=1000)
        assert isinstance(exc, LoggingException)
        assert exc.queue_size == 1000
        assert exc.current_size == 500
        assert exc.details["queue_name"] == "log_queue"
        assert exc.details["max_size"] == 1000


class TestLogFilterError:
    """LogFilterError 测试"""

    def test_filter_error(self):
        """测试过滤器错误"""
        exc = LogFilterError("Filter error", filter_type="security",
                            filter_rule="invalid_rule")
        assert isinstance(exc, LoggingException)
        assert exc.filter_type == "security"
        assert exc.filter_rule == "invalid_rule"


class TestLogSecurityError:
    """LogSecurityError 测试"""

    def test_security_error(self):
        """测试安全错误"""
        exc = LogSecurityError("Security error", security_level="high",
                              violation_type="unauthorized_access")
        assert isinstance(exc, LoggingException)
        assert exc.details["security_level"] == "high"
        assert exc.details["violation_type"] == "unauthorized_access"


class TestLogMonitorError:
    """LogMonitorError 测试"""

    def test_monitor_error(self):
        """测试监控错误"""
        exc = LogMonitorError("Monitor error", metric_name="response_time",
                             threshold=1.0, actual_value=2.5, monitor_type="performance")
        assert isinstance(exc, LoggingException)
        assert exc.metric_name == "response_time"
        assert exc.threshold == 1.0
        assert exc.actual_value == 2.5
        assert exc.details["monitor_type"] == "performance"


class TestLogPerformanceError:
    """LogPerformanceError 测试"""

    def test_performance_error(self):
        """测试性能错误"""
        exc = LogPerformanceError("Performance error", operation="log_write",
                                 duration=5.2, threshold=1.0)
        assert isinstance(exc, LoggingException)
        assert exc.operation == "log_write"
        assert exc.duration == 5.2
        assert exc.threshold == 1.0


class TestLogStorageError:
    """LogStorageError 测试"""

    def test_storage_error(self):
        """测试存储错误"""
        exc = LogStorageError("Storage error", storage_type="file",
                             storage_path="/var/log", error_code="EACCES")
        assert isinstance(exc, LoggingException)
        assert exc.storage_type == "file"
        assert exc.storage_path == "/var/log"
        assert exc.details["error_code"] == "EACCES"


class TestLogAsyncError:
    """LogAsyncError 测试"""

    def test_async_error(self):
        """测试异步错误"""
        exc = LogAsyncError("Async error", task_id="12345", worker_id="worker-1",
                           task_name="log_processor", error_type="TimeoutError")
        assert isinstance(exc, LoggingException)
        assert exc.task_id == "12345"
        assert exc.worker_id == "worker-1"
        assert exc.details["task_name"] == "log_processor"
        assert exc.details["error_type"] == "TimeoutError"


class TestLogBatchError:
    """LogBatchError 测试"""

    def test_batch_error(self):
        """测试批处理错误"""
        exc = LogBatchError("Batch error", batch_size=100, processed_count=50,
                           failed_count=10)
        assert isinstance(exc, LoggingException)
        assert exc.batch_size == 100
        assert exc.processed_count == 50
        assert exc.failed_count == 10


class TestLogValidationError:
    """LogValidationError 测试"""

    def test_validation_error(self):
        """测试验证错误"""
        exc = LogValidationError("Validation error", field_name="log_level",
                                field_value="INVALID", expected_type="str")
        assert isinstance(exc, LoggingException)
        assert exc.details["field_name"] == "log_level"
        assert exc.details["field_value"] == "INVALID"
        assert exc.details["expected_type"] == "str"


class TestLogTimeoutError:
    """LogTimeoutError 测试"""

    def test_timeout_error(self):
        """测试超时错误"""
        exc = LogTimeoutError("Timeout error", timeout=5.0, operation="network_send",
                             elapsed_time=7.2)
        assert isinstance(exc, LoggingException)
        assert exc.timeout == 5.0
        assert exc.operation == "network_send"
        assert exc.details["elapsed_time"] == 7.2


class TestResourceError:
    """ResourceError 测试"""

    def test_resource_error(self):
        """测试资源错误"""
        exc = ResourceError("Resource error", resource_type="memory", resource_id="mem-1",
                           available=100, required=200, unit="MB")
        assert isinstance(exc, LoggingException)
        assert exc.resource_type == "memory"
        assert exc.resource_id == "mem-1"
        assert exc.details["available"] == 100
        assert exc.details["required"] == 200
        assert exc.details["unit"] == "MB"


class TestExceptionHierarchy:
    """异常层次结构测试"""

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        # 测试所有异常都是LoggingException的子类
        exceptions = [
            LogConfigurationError("test"),
            LogHandlerError("test"),
            LogFormatterError("test"),
            LogFileError("test"),
            LogRotationError("test"),
            LogCompressionError("test"),
            LogNetworkError("test"),
            LogQueueError("test"),
            LogFilterError("test"),
            LogSecurityError("test"),
            LogMonitorError("test"),
            LogPerformanceError("test"),
            LogStorageError("test"),
            LogAsyncError("test"),
            LogBatchError("test"),
            LogValidationError("test"),
            LogTimeoutError("test"),
            ResourceError("test")
        ]

        for exc in exceptions:
            assert isinstance(exc, LoggingException)
            assert isinstance(exc, Exception)

    def test_file_error_hierarchy(self):
        """测试文件错误层次结构"""
        rotation_error = LogRotationError("test")
        compression_error = LogCompressionError("test")

        assert isinstance(rotation_error, LogFileError)
        assert isinstance(rotation_error, LoggingException)
        assert isinstance(compression_error, LogFileError)
        assert isinstance(compression_error, LoggingException)


class TestExceptionDetails:
    """异常详细信息测试"""

    def test_exception_details_preservation(self):
        """测试异常详细信息保留"""
        original_details = {"key1": "value1", "key2": 42}
        exc = LoggingException("test", details=original_details)

        assert exc.details == original_details
        assert exc.details is not original_details  # 应该是副本

    def test_exception_kwargs_merge(self):
        """测试异常kwargs合并"""
        initial_details = {"initial": "value"}
        exc = LoggingException("test", details=initial_details,
                              additional="param", number=123)

        expected_details = {"initial": "value", "additional": "param", "number": 123}
        assert exc.details == expected_details
