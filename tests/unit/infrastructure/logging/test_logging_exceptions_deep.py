#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Logging核心异常深度测试"""

import pytest


# ============================================================================
# 常量测试
# ============================================================================

def test_http_status_constants():
    """测试HTTP状态码常量"""
    from src.infrastructure.logging.core.exceptions import (
        HTTP_OK, HTTP_BAD_REQUEST, HTTP_UNAUTHORIZED,
        HTTP_FORBIDDEN, HTTP_NOT_FOUND, HTTP_INTERNAL_ERROR
    )
    
    assert HTTP_OK == 200
    assert HTTP_BAD_REQUEST == 400
    assert HTTP_UNAUTHORIZED == 401
    assert HTTP_FORBIDDEN == 403
    assert HTTP_NOT_FOUND == 404
    assert HTTP_INTERNAL_ERROR == 500


def test_pagination_constants():
    """测试分页常量"""
    from src.infrastructure.logging.core.exceptions import (
        DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE
    )
    
    assert DEFAULT_PAGE_SIZE == 20
    assert MAX_PAGE_SIZE == 100
    assert DEFAULT_PAGE_SIZE < MAX_PAGE_SIZE


# ============================================================================
# LoggingException基础异常测试
# ============================================================================

def test_logging_exception_basic():
    """测试LoggingException基础创建"""
    from src.infrastructure.logging.core.exceptions import LoggingException
    
    error = LoggingException("Test error")
    assert error.message == "Test error"
    assert error.logger_name is None
    assert error.log_level is None
    assert error.details == {}


def test_logging_exception_with_logger_name():
    """测试LoggingException包含logger名称"""
    from src.infrastructure.logging.core.exceptions import LoggingException
    
    error = LoggingException("Error", logger_name="app.logger")
    assert error.logger_name == "app.logger"


def test_logging_exception_with_log_level():
    """测试LoggingException包含日志级别"""
    from src.infrastructure.logging.core.exceptions import LoggingException
    
    error = LoggingException("Error", log_level="ERROR")
    assert error.log_level == "ERROR"


def test_logging_exception_with_kwargs():
    """测试LoggingException合并kwargs到details"""
    from src.infrastructure.logging.core.exceptions import LoggingException
    
    error = LoggingException("Error", custom_field="value", count=10)
    assert error.details["custom_field"] == "value"
    assert error.details["count"] == 10


# ============================================================================
# 各类异常测试
# ============================================================================

def test_log_configuration_error():
    """测试LogConfigurationError"""
    from src.infrastructure.logging.core.exceptions import LogConfigurationError
    
    error = LogConfigurationError(
        "Config error",
        config_key="log_level",
        expected_value="INFO",
        actual_value="INVALID"
    )
    assert error.config_key == "log_level"
    assert error.expected_value == "INFO"
    assert error.actual_value == "INVALID"


def test_log_handler_error():
    """测试LogHandlerError"""
    from src.infrastructure.logging.core.exceptions import LogHandlerError
    
    error = LogHandlerError(
        "Handler error",
        handler_type="FileHandler",
        handler_name="main_handler"
    )
    assert error.handler_type == "FileHandler"
    assert error.handler_name == "main_handler"


def test_log_formatter_error():
    """测试LogFormatterError"""
    from src.infrastructure.logging.core.exceptions import LogFormatterError
    
    error = LogFormatterError(
        "Format error",
        formatter_type="JSONFormatter",
        original_message="test message"
    )
    assert error.formatter_type == "JSONFormatter"
    assert error.original_message == "test message"


def test_log_file_error():
    """测试LogFileError"""
    from src.infrastructure.logging.core.exceptions import LogFileError
    
    error = LogFileError(
        "File error",
        file_path="/var/log/app.log",
        operation="write"
    )
    assert error.file_path == "/var/log/app.log"
    assert error.operation == "write"


def test_log_rotation_error():
    """测试LogRotationError"""
    from src.infrastructure.logging.core.exceptions import LogRotationError
    
    error = LogRotationError("Rotation failed", rotation_type="size_based")
    assert error.rotation_type == "size_based"


def test_log_compression_error():
    """测试LogCompressionError"""
    from src.infrastructure.logging.core.exceptions import LogCompressionError
    
    error = LogCompressionError("Compression failed", compression_algorithm="gzip")
    assert error.compression_algorithm == "gzip"


def test_log_network_error():
    """测试LogNetworkError"""
    from src.infrastructure.logging.core.exceptions import LogNetworkError
    
    error = LogNetworkError(
        "Network error",
        host="log-server.com",
        port=514,
        protocol="TCP"
    )
    assert error.host == "log-server.com"
    assert error.port == 514
    assert error.protocol == "TCP"


def test_log_queue_error():
    """测试LogQueueError"""
    from src.infrastructure.logging.core.exceptions import LogQueueError
    
    error = LogQueueError(
        "Queue full",
        queue_size=1000,
        current_size=1000
    )
    assert error.queue_size == 1000
    assert error.current_size == 1000


def test_log_filter_error():
    """测试LogFilterError"""
    from src.infrastructure.logging.core.exceptions import LogFilterError
    
    error = LogFilterError(
        "Filter error",
        filter_type="regex",
        filter_rule=".*ERROR.*"
    )
    assert error.filter_type == "regex"
    assert error.filter_rule == ".*ERROR.*"


def test_log_security_error():
    """测试LogSecurityError"""
    from src.infrastructure.logging.core.exceptions import LogSecurityError
    
    error = LogSecurityError(
        "Security issue",
        security_issue="sensitive_data_leak",
        sensitive_data="password"
    )
    assert error.security_issue == "sensitive_data_leak"
    assert error.sensitive_data == "password"


def test_log_monitor_error():
    """测试LogMonitorError"""
    from src.infrastructure.logging.core.exceptions import LogMonitorError
    
    error = LogMonitorError(
        "Monitor error",
        metric_name="log_rate",
        threshold=1000.0,
        actual_value=1500.0
    )
    assert error.metric_name == "log_rate"
    assert error.threshold == 1000.0
    assert error.actual_value == 1500.0


def test_log_performance_error():
    """测试LogPerformanceError"""
    from src.infrastructure.logging.core.exceptions import LogPerformanceError
    
    error = LogPerformanceError(
        "Performance issue",
        operation="write",
        duration=5.0,
        threshold=1.0
    )
    assert error.operation == "write"
    assert error.duration == 5.0
    assert error.threshold == 1.0


def test_log_storage_error():
    """测试LogStorageError"""
    from src.infrastructure.logging.core.exceptions import LogStorageError
    
    error = LogStorageError(
        "Storage error",
        storage_type="elasticsearch",
        storage_path="/logs/2025"
    )
    assert error.storage_type == "elasticsearch"
    assert error.storage_path == "/logs/2025"


def test_log_async_error():
    """测试LogAsyncError"""
    from src.infrastructure.logging.core.exceptions import LogAsyncError
    
    error = LogAsyncError(
        "Async error",
        task_id="task-123",
        worker_id="worker-1"
    )
    assert error.task_id == "task-123"
    assert error.worker_id == "worker-1"


def test_log_batch_error():
    """测试LogBatchError"""
    from src.infrastructure.logging.core.exceptions import LogBatchError
    
    error = LogBatchError(
        "Batch processing failed",
        batch_size=100,
        processed_count=75,
        failed_count=25
    )
    assert error.batch_size == 100
    assert error.processed_count == 75
    assert error.failed_count == 25


def test_log_validation_error():
    """测试LogValidationError"""
    from src.infrastructure.logging.core.exceptions import LogValidationError
    
    error = LogValidationError(
        "Validation failed",
        validation_rule="max_length",
        invalid_value="x" * 1000
    )
    assert error.validation_rule == "max_length"
    assert len(error.invalid_value) == 1000


def test_log_timeout_error():
    """测试LogTimeoutError"""
    from src.infrastructure.logging.core.exceptions import LogTimeoutError
    
    error = LogTimeoutError(
        "Timeout",
        timeout=30.0,
        operation="flush"
    )
    assert error.timeout == 30.0
    assert error.operation == "flush"


def test_resource_error():
    """测试ResourceError"""
    from src.infrastructure.logging.core.exceptions import ResourceError
    
    error = ResourceError(
        "Resource error",
        resource_type="file_handle",
        resource_id="handle-123"
    )
    assert error.resource_type == "file_handle"
    assert error.resource_id == "handle-123"


# ============================================================================
# 异常继承关系测试
# ============================================================================

def test_all_errors_inherit_logging_exception():
    """测试所有异常都继承自LoggingException"""
    from src.infrastructure.logging.core.exceptions import (
        LoggingException,
        LogConfigurationError,
        LogHandlerError,
        LogFileError,
        LogRotationError
    )
    
    assert issubclass(LogConfigurationError, LoggingException)
    assert issubclass(LogHandlerError, LoggingException)
    assert issubclass(LogFileError, LoggingException)
    assert issubclass(LogRotationError, LogFileError)


# ============================================================================
# 装饰器测试
# ============================================================================

def test_handle_logging_exception_decorator():
    """测试handle_logging_exception装饰器"""
    from src.infrastructure.logging.core.exceptions import handle_logging_exception
    
    @handle_logging_exception(operation="test")
    def test_func():
        return "success"
    
    result = test_func()
    assert result == "success"


def test_handle_logging_exception_catches_error():
    """测试装饰器捕获错误"""
    from src.infrastructure.logging.core.exceptions import (
        handle_logging_exception,
        LoggingException
    )
    
    @handle_logging_exception(operation="test")
    def test_func():
        raise ValueError("Test error")
    
    with pytest.raises(LoggingException):
        test_func()


# ============================================================================
# 异常使用场景测试
# ============================================================================

def test_exception_can_be_raised():
    """测试异常可以被raise"""
    from src.infrastructure.logging.core.exceptions import LogHandlerError
    
    with pytest.raises(LogHandlerError) as exc_info:
        raise LogHandlerError("Test", handler_type="FileHandler")
    
    assert exc_info.value.handler_type == "FileHandler"


def test_exception_catching_by_base_class():
    """测试通过基类捕获异常"""
    from src.infrastructure.logging.core.exceptions import (
        LoggingException,
        LogHandlerError
    )
    
    try:
        raise LogHandlerError("Test")
    except LoggingException as e:
        assert isinstance(e, LogHandlerError)

