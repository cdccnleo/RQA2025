#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - Core日志系统异常处理

测试logging/core/exceptions.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch

# 确保模块被导入以进行覆盖率检测
import src.infrastructure.logging.core.exceptions


class TestCoreExceptions:
    """测试Core日志系统异常处理"""

    def setup_method(self):
        """测试前准备"""
        from src.infrastructure.logging.core.exceptions import (
            LoggingException, LogConfigurationError, LogHandlerError,
            LogFormatterError, LogFilterError, LogStorageError,
            LogNetworkError, LogSecurityError, LogPerformanceError,
            LogValidationError, LogTimeoutError, ResourceError,
            LogFileError, LogRotationError, LogCompressionError,
            LogQueueError, LogMonitorError, LogAsyncError, LogBatchError,
            handle_logging_exception, handle_log_file_exception,
            handle_log_network_exception, handle_log_performance_exception,
            HTTP_OK, HTTP_BAD_REQUEST, HTTP_UNAUTHORIZED,
            HTTP_FORBIDDEN, HTTP_NOT_FOUND, HTTP_INTERNAL_ERROR,
            DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
        )
        self.LoggingException = LoggingException
        self.LogConfigurationError = LogConfigurationError
        self.LogHandlerError = LogHandlerError
        self.LogFormatterError = LogFormatterError
        self.LogFilterError = LogFilterError
        self.LogStorageError = LogStorageError
        self.LogNetworkError = LogNetworkError
        self.LogSecurityError = LogSecurityError
        self.LogPerformanceError = LogPerformanceError
        self.LogValidationError = LogValidationError
        self.LogTimeoutError = LogTimeoutError
        self.LogResourceError = ResourceError
        self.LogFileError = LogFileError
        self.LogRotationError = LogRotationError
        self.LogCompressionError = LogCompressionError
        self.LogQueueError = LogQueueError
        self.LogMonitorError = LogMonitorError
        self.LogAsyncError = LogAsyncError
        self.LogBatchError = LogBatchError
        self.handle_logging_exception = handle_logging_exception
        self.handle_log_file_exception = handle_log_file_exception
        self.handle_log_network_exception = handle_log_network_exception
        self.handle_log_performance_exception = handle_log_performance_exception
        self.HTTP_OK = HTTP_OK
        self.HTTP_BAD_REQUEST = HTTP_BAD_REQUEST
        self.HTTP_UNAUTHORIZED = HTTP_UNAUTHORIZED
        self.HTTP_FORBIDDEN = HTTP_FORBIDDEN
        self.HTTP_NOT_FOUND = HTTP_NOT_FOUND
        self.HTTP_INTERNAL_ERROR = HTTP_INTERNAL_ERROR
        self.DEFAULT_PAGE_SIZE = DEFAULT_PAGE_SIZE
        self.MAX_PAGE_SIZE = MAX_PAGE_SIZE

    def test_http_constants(self):
        """测试HTTP常量"""
        assert self.HTTP_OK == 200
        assert self.HTTP_BAD_REQUEST == 400
        assert self.HTTP_UNAUTHORIZED == 401
        assert self.HTTP_FORBIDDEN == 403
        assert self.HTTP_NOT_FOUND == 404
        assert self.HTTP_INTERNAL_ERROR == 500

    def test_pagination_constants(self):
        """测试分页常量"""
        assert self.DEFAULT_PAGE_SIZE == 20
        assert self.MAX_PAGE_SIZE == 100

    def test_logging_exception_creation(self):
        """测试基础日志异常创建"""
        exception = self.LoggingException("Test error message")

        assert str(exception) == "Test error message"
        assert exception.message == "Test error message"
        assert exception.logger_name is None
        assert exception.log_level is None
        assert exception.details == {}

    def test_logging_exception_with_details(self):
        """测试带详细信息的日志异常"""
        details = {"key": "value", "code": 123}
        exception = self.LoggingException(
            "Test error with details",
            logger_name="test_logger",
            log_level="ERROR",
            details=details
        )

        assert str(exception) == "Test error with details"
        assert exception.logger_name == "test_logger"
        assert exception.log_level == "ERROR"
        assert exception.details == details

    def test_log_configuration_error(self):
        """测试配置错误异常"""
        exception = self.LogConfigurationError(
            "Invalid configuration",
            config_key="log_level",
            expected_value="INFO",
            actual_value="INVALID"
        )

        assert str(exception) == "Invalid configuration"
        assert exception.config_key == "log_level"
        assert exception.expected_value == "INFO"
        assert exception.actual_value == "INVALID"

    def test_log_handler_error(self):
        """测试处理器错误异常"""
        exception = self.LogHandlerError(
            "Handler initialization failed",
            handler_name="file_handler",
            handler_type="FileHandler"
        )

        assert str(exception) == "Handler initialization failed"
        assert exception.handler_name == "file_handler"
        assert exception.handler_type == "FileHandler"

    def test_log_formatter_error(self):
        """测试格式化器错误异常"""
        exception = self.LogFormatterError(
            "Format string invalid",
            formatter_type="json_formatter",
            original_message='{"invalid": json}'
        )

        assert str(exception) == "Format string invalid"
        assert exception.formatter_type == "json_formatter"
        assert exception.original_message == '{"invalid": json}'

    def test_log_filter_error(self):
        """测试过滤器错误异常"""
        exception = self.LogFilterError(
            "Filter condition invalid",
            filter_type="level_filter",
            filter_rule="invalid > condition"
        )

        assert str(exception) == "Filter condition invalid"
        assert exception.filter_type == "level_filter"
        assert exception.filter_rule == "invalid > condition"

    def test_log_storage_error(self):
        """测试存储错误异常"""
        exception = self.LogStorageError(
            "Storage write failed",
            storage_type="file",
            storage_path="/var/log/app.log",
            error_code="EACCES"
        )

        assert str(exception) == "Storage write failed"
        assert exception.storage_type == "file"
        assert exception.storage_path == "/var/log/app.log"
        assert exception.details.get("error_code") == "EACCES"

    def test_log_network_error(self):
        """测试网络错误异常"""
        exception = self.LogNetworkError(
            "Network connection failed",
            host="log.example.com",
            port=514,
            protocol="tcp",
            timeout=30
        )

        assert str(exception) == "Network connection failed"
        assert exception.host == "log.example.com"
        assert exception.port == 514
        assert exception.protocol == "tcp"
        assert exception.details.get("timeout") == 30

    def test_log_security_error(self):
        """测试安全错误异常"""
        exception = self.LogSecurityError(
            "Security violation detected",
            security_issue="unauthorized_access",
            sensitive_data="/admin/logs",
            user_id="user123",
            resource="/admin/logs"
        )

        assert str(exception) == "Security violation detected"
        assert exception.security_issue == "unauthorized_access"
        assert exception.sensitive_data == "/admin/logs"
        assert exception.details.get("user_id") == "user123"
        assert exception.details.get("resource") == "/admin/logs"

    def test_log_performance_error(self):
        """测试性能错误异常"""
        exception = self.LogPerformanceError(
            "Performance threshold exceeded",
            operation="log_processing",
            duration=2.5,
            threshold=1000,
            metric="logs_per_second",
            actual_value=1200
        )

        assert str(exception) == "Performance threshold exceeded"
        assert exception.operation == "log_processing"
        assert exception.duration == 2.5
        assert exception.threshold == 1000
        assert exception.details.get("metric") == "logs_per_second"
        assert exception.details.get("actual_value") == 1200


    def test_log_validation_error(self):
        """测试验证错误异常"""
        exception = self.LogValidationError(
            "Log entry validation failed",
            validation_rule="timestamp_format",
            invalid_value="invalid_string",
            field="timestamp",
            expected_type="datetime",
            actual_value="invalid_string"
        )

        assert str(exception) == "Log entry validation failed"
        assert exception.validation_rule == "timestamp_format"
        assert exception.invalid_value == "invalid_string"
        assert exception.details.get("field") == "timestamp"
        assert exception.details.get("expected_type") == "datetime"
        assert exception.details.get("actual_value") == "invalid_string"

    def test_log_timeout_error(self):
        """测试超时错误异常"""
        exception = self.LogTimeoutError(
            "Operation timed out",
            timeout=30.0,
            operation="log_flush",
            timeout_seconds=30,
            elapsed_seconds=35
        )

        assert str(exception) == "Operation timed out"
        assert exception.timeout == 30.0
        assert exception.operation == "log_flush"
        assert exception.details.get("timeout_seconds") == 30
        assert exception.details.get("elapsed_seconds") == 35

    def test_log_resource_error(self):
        """测试资源错误异常"""
        exception = self.LogResourceError(
            "Resource limit exceeded",
            resource_type="memory",
            resource_id="main_process",
            limit_mb=1024,
            used_mb=1200
        )

        assert str(exception) == "Resource limit exceeded"
        assert exception.resource_type == "memory"
        assert exception.resource_id == "main_process"
        assert exception.details.get("limit_mb") == 1024
        assert exception.details.get("used_mb") == 1200

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        # 所有异常都应该继承自LoggingException
        exceptions = [
            self.LogConfigurationError("test"),
            self.LogHandlerError("test"),
            self.LogFormatterError("test"),
            self.LogFilterError("test"),
            self.LogStorageError("test"),
            self.LogNetworkError("test"),
            self.LogSecurityError("test"),
            self.LogPerformanceError("test"),
            self.LogValidationError("test"),
            self.LogTimeoutError("test"),
            self.LogResourceError("test")
        ]

        for exc in exceptions:
            assert isinstance(exc, self.LoggingException)
            assert isinstance(exc, Exception)

    def test_exception_details_preservation(self):
        """测试异常详细信息保留"""
        original_details = {"custom": "data", "trace_id": "abc123"}
        exception = self.LoggingException(
            "Test message",
            logger_name="test_logger",
            details=original_details
        )

        # 确保原始details被保留
        assert exception.details["custom"] == "data"
        assert exception.details["trace_id"] == "abc123"

        # 添加更多details
        exception.details["additional"] = "info"
        assert exception.details["additional"] == "info"

    def test_log_file_error(self):
        """测试日志文件错误异常"""
        exception = self.LogFileError(
            "File access denied",
            file_path="/var/log/app.log",
            operation="write"
        )

        assert str(exception) == "File access denied"
        assert exception.file_path == "/var/log/app.log"
        assert exception.operation == "write"

    def test_log_rotation_error(self):
        """测试日志轮转错误异常"""
        exception = self.LogRotationError(
            "Rotation failed",
            rotation_type="size_based",
            file_path="/var/log/app.log"
        )

        assert str(exception) == "Rotation failed"
        assert exception.rotation_type == "size_based"
        assert exception.file_path == "/var/log/app.log"

    def test_log_compression_error(self):
        """测试日志压缩错误异常"""
        exception = self.LogCompressionError(
            "Compression failed",
            compression_algorithm="gzip",
            file_path="/var/log/app.log"
        )

        assert str(exception) == "Compression failed"
        assert exception.compression_algorithm == "gzip"
        assert exception.file_path == "/var/log/app.log"

    def test_log_queue_error(self):
        """测试日志队列错误异常"""
        exception = self.LogQueueError(
            "Queue overflow",
            queue_size=1000,
            current_size=1500
        )

        assert str(exception) == "Queue overflow"
        assert exception.queue_size == 1000
        assert exception.current_size == 1500

    def test_log_monitor_error(self):
        """测试日志监控错误异常"""
        exception = self.LogMonitorError(
            "Threshold exceeded",
            metric_name="error_rate",
            threshold=5.0,
            actual_value=8.5
        )

        assert str(exception) == "Threshold exceeded"
        assert exception.metric_name == "error_rate"
        assert exception.threshold == 5.0
        assert exception.actual_value == 8.5

    def test_log_async_error(self):
        """测试日志异步错误异常"""
        exception = self.LogAsyncError(
            "Async task failed",
            task_id="task_123",
            worker_id="worker_456"
        )

        assert str(exception) == "Async task failed"
        assert exception.task_id == "task_123"
        assert exception.worker_id == "worker_456"

    def test_log_batch_error(self):
        """测试日志批量错误异常"""
        exception = self.LogBatchError(
            "Batch processing failed",
            batch_size=100,
            processed_count=75,
            failed_count=25
        )

        assert str(exception) == "Batch processing failed"
        assert exception.batch_size == 100
        assert exception.processed_count == 75
        assert exception.failed_count == 25

    def test_handle_logging_exception_decorator(self):
        """测试日志异常处理装饰器"""
        @self.handle_logging_exception("test_operation")
        def test_function():
            raise ValueError("Test error")

        # 测试装饰器正确包装异常
        with pytest.raises(self.LoggingException) as exc_info:
            test_function()
        
        assert "test_operation 失败" in str(exc_info.value)
        assert exc_info.value.details["original_error"] == "Test error"

    def test_handle_logging_exception_decorator_passthrough(self):
        """测试装饰器对LoggingException的透传"""
        @self.handle_logging_exception("test_operation")
        def test_function():
            raise self.LoggingException("Direct logging exception")

        # LoggingException应该直接透传
        with pytest.raises(self.LoggingException) as exc_info:
            test_function()
        
        assert str(exc_info.value) == "Direct logging exception"

    def test_handle_log_file_exception_decorator(self):
        """测试日志文件异常处理装饰器"""
        @self.handle_log_file_exception("/test/file.log", "read")
        def test_file_operation():
            raise IOError("File not found")

        with pytest.raises(self.LogFileError) as exc_info:
            test_file_operation()
        
        assert "文件操作失败 read" in str(exc_info.value)
        assert exc_info.value.file_path == "/test/file.log"
        assert exc_info.value.operation == "read"

    def test_handle_log_network_exception_decorator(self):
        """测试日志网络异常处理装饰器"""
        @self.handle_log_network_exception("example.com", 514, "tcp")
        def test_network_operation():
            raise ConnectionError("Connection failed")

        with pytest.raises(self.LogNetworkError) as exc_info:
            test_network_operation()
        
        assert "网络连接失败 tcp://example.com:514" in str(exc_info.value)
        assert exc_info.value.host == "example.com"
        assert exc_info.value.port == 514
        assert exc_info.value.protocol == "tcp"

    def test_handle_log_performance_exception_decorator_success(self):
        """测试性能异常处理装饰器 - 成功情况"""
        @self.handle_log_performance_exception("fast_operation", 1.0)
        def fast_function():
            time.sleep(0.1)  # 快速操作
            return "success"

        # 应该正常执行，不抛出异常
        result = fast_function()
        assert result == "success"

    def test_handle_log_performance_exception_decorator_timeout(self):
        """测试性能异常处理装饰器 - 超时情况"""
        @self.handle_log_performance_exception("slow_operation", 0.1)
        def slow_function():
            time.sleep(0.2)  # 慢操作
            return "success"

        # 应该抛出性能异常
        with pytest.raises(self.LogPerformanceError) as exc_info:
            slow_function()
        
        assert "操作性能超出阈值 slow_operation" in str(exc_info.value)
        assert exc_info.value.operation == "slow_operation"
        assert exc_info.value.duration > 0.1
        assert exc_info.value.threshold == 0.1

    def test_handle_log_performance_exception_decorator_error(self):
        """测试性能异常处理装饰器 - 错误情况"""
        @self.handle_log_performance_exception("error_operation", 1.0)
        def error_function():
            raise ValueError("Function error")

        with pytest.raises(self.LoggingException) as exc_info:
            error_function()
        
        assert "操作执行失败 error_operation" in str(exc_info.value)
        assert "duration" in exc_info.value.details

    def test_all_remaining_exceptions_inheritance(self):
        """测试剩余异常类的继承关系"""
        remaining_exceptions = [
            self.LogFileError("test"),
            self.LogRotationError("test"),
            self.LogCompressionError("test"),
            self.LogQueueError("test"),
            self.LogMonitorError("test"),
            self.LogAsyncError("test"),
            self.LogBatchError("test")
        ]

        for exc in remaining_exceptions:
            assert isinstance(exc, self.LoggingException)
            assert isinstance(exc, Exception)

    def test_logging_exception_with_kwargs(self):
        """测试LoggingException处理kwargs参数"""
        exception = self.LoggingException(
            "Test with kwargs",
            extra_param="extra_value",
            another_param=123
        )

        assert str(exception) == "Test with kwargs"
        assert exception.details["extra_param"] == "extra_value"
        assert exception.details["another_param"] == 123
