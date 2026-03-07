"""
Error Handler 单元测试

测试错误处理模块的核心功能和组件。
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, call
from datetime import datetime
import traceback
import uuid

from src.infrastructure.logging.core.error_handler import (
    ErrorSeverity,
    ErrorType,
    ErrorInfo,
    ErrorClassifier,
    SeverityAnalyzer,
    ErrorProcessor,
    ErrorHandler,
)
from src.infrastructure.logging.core.exceptions import ResourceError


class TestErrorSeverity:
    """测试错误严重级别枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_enum_str(self):
        """测试枚举字符串表示"""
        assert str(ErrorSeverity.LOW) == "ErrorSeverity.LOW"
        assert str(ErrorSeverity.CRITICAL) == "ErrorSeverity.CRITICAL"


class TestErrorType:
    """测试错误类型枚举"""

    def test_enum_values(self):
        """测试枚举值"""
        assert ErrorType.VALIDATION.value == "validation"
        assert ErrorType.CONNECTION.value == "connection"
        assert ErrorType.TIMEOUT.value == "timeout"
        assert ErrorType.PERMISSION.value == "permission"
        assert ErrorType.RESOURCE.value == "resource"
        assert ErrorType.SYSTEM.value == "system"
        assert ErrorType.BUSINESS.value == "business"
        assert ErrorType.UNKNOWN.value == "unknown"

    def test_enum_str(self):
        """测试枚举字符串表示"""
        assert str(ErrorType.VALIDATION) == "ErrorType.VALIDATION"
        assert str(ErrorType.UNKNOWN) == "ErrorType.UNKNOWN"


class TestErrorInfo:
    """测试错误信息数据类"""

    def test_init_minimal(self):
        """测试最小化初始化"""
        error_info = ErrorInfo(
            error_type=ErrorType.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Validation failed"
        )

        assert error_info.error_type == ErrorType.VALIDATION
        assert error_info.severity == ErrorSeverity.LOW
        assert error_info.message == "Validation failed"
        assert error_info.details is None
        assert error_info.stack_trace is None
        assert error_info.context is None
        assert error_info.event_id is None

    def test_init_full(self):
        """测试完整初始化"""
        event_id = str(uuid.uuid4())
        stack_trace = "Traceback (most recent call last)..."
        details = {"field": "username", "value": None}
        context = {"user_id": 123, "request_id": "req-456"}

        error_info = ErrorInfo(
            error_type=ErrorType.PERMISSION,
            severity=ErrorSeverity.HIGH,
            message="Access denied",
            details=details,
            stack_trace=stack_trace,
            context=context,
            event_id=event_id
        )

        assert error_info.error_type == ErrorType.PERMISSION
        assert error_info.severity == ErrorSeverity.HIGH
        assert error_info.message == "Access denied"
        assert error_info.details == details
        assert error_info.stack_trace == stack_trace
        assert error_info.context == context
        assert error_info.event_id == event_id


class TestErrorClassifier:
    """测试错误分类器"""

    @pytest.fixture
    def error_classifier(self):
        """创建错误分类器实例"""
        return ErrorClassifier()

    def test_init(self, error_classifier):
        """测试初始化"""
        assert error_classifier is not None

    def test_classify_validation_error(self, error_classifier):
        """测试分类验证错误"""
        # ValueError通常是验证错误
        error = ValueError("Invalid input")
        result = error_classifier.classify_error(error)

        assert result == ErrorType.VALIDATION

    def test_classify_timeout_error(self, error_classifier):
        """测试分类超时错误"""
        # TimeoutError
        error = TimeoutError("Connection timed out")
        result = error_classifier.classify_error(error)

        assert result == ErrorType.TIMEOUT

    def test_classify_connection_error(self, error_classifier):
        """测试分类连接错误"""
        # ConnectionError
        error = ConnectionError("Failed to connect")
        result = error_classifier.classify_error(error)

        assert result == ErrorType.CONNECTION

    def test_classify_permission_error(self, error_classifier):
        """测试分类权限错误"""
        # PermissionError
        error = PermissionError("Access denied")
        result = error_classifier.classify_error(error)

        assert result == ErrorType.PERMISSION

    def test_classify_resource_error(self, error_classifier):
        """测试分类资源错误"""
        # OSError被分类为SYSTEM，让我们测试MemoryError
        error = MemoryError("Out of memory")
        result = error_classifier.classify_error(error)

        assert result == ErrorType.RESOURCE

    def test_classify_system_error(self, error_classifier):
        """测试分类系统错误"""
        # SystemError
        error = SystemError("Internal system error")
        result = error_classifier.classify_error(error)

        assert result == ErrorType.SYSTEM

    def test_classify_business_error(self, error_classifier):
        """测试分类业务错误"""
        # 自定义业务异常
        class BusinessException(Exception):
            pass

        error = BusinessException("Business logic error")
        result = error_classifier.classify_error(error)

        assert result == ErrorType.BUSINESS

    def test_classify_unknown_error(self, error_classifier):
        """测试分类未知错误"""
        # 完全未知的异常类型
        error = Exception("Unknown error occurred")
        result = error_classifier.classify_error(error)

        assert result == ErrorType.UNKNOWN


class TestSeverityAnalyzer:
    """测试严重程度分析器"""

    @pytest.fixture
    def severity_analyzer(self):
        """创建严重程度分析器实例"""
        return SeverityAnalyzer()

    def test_init(self, severity_analyzer):
        """测试初始化"""
        assert severity_analyzer is not None

    def test_determine_severity_by_error_type(self, severity_analyzer):
        """测试通过错误类型确定严重程度"""
        # 系统错误 -> 严重
        severity = severity_analyzer._get_severity_by_error_type(ErrorType.SYSTEM)
        assert severity == ErrorSeverity.CRITICAL

        # 权限错误 -> 高
        severity = severity_analyzer._get_severity_by_error_type(ErrorType.PERMISSION)
        assert severity == ErrorSeverity.HIGH

        # 连接错误 -> 高
        severity = severity_analyzer._get_severity_by_error_type(ErrorType.CONNECTION)
        assert severity == ErrorSeverity.HIGH

        # 超时错误 -> 中
        severity = severity_analyzer._get_severity_by_error_type(ErrorType.TIMEOUT)
        assert severity == ErrorSeverity.MEDIUM

        # 验证错误 -> 中
        severity = severity_analyzer._get_severity_by_error_type(ErrorType.VALIDATION)
        assert severity == ErrorSeverity.MEDIUM

    def test_determine_severity_by_message(self, severity_analyzer):
        """测试通过消息内容确定严重程度"""
        # 包含"critical"关键词
        severity = severity_analyzer._get_severity_by_message_content("critical system failure occurred")
        assert severity == ErrorSeverity.CRITICAL

        # 包含"fatal"关键词
        severity = severity_analyzer._get_severity_by_message_content("fatal database crash")
        assert severity == ErrorSeverity.CRITICAL

        # 包含"error"关键词
        severity = severity_analyzer._get_severity_by_message_content("database connection error")
        assert severity == ErrorSeverity.HIGH

        # 包含"warning"关键词
        severity = severity_analyzer._get_severity_by_message_content("low disk space warning")
        assert severity == ErrorSeverity.MEDIUM

        # 普通消息
        severity = severity_analyzer._get_severity_by_message_content("user logged in successfully")
        assert severity == ErrorSeverity.LOW

    def test_determine_severity_comprehensive(self, severity_analyzer):
        """测试综合严重程度确定"""
        # 系统错误类型
        error = SystemError("Internal error")
        severity = severity_analyzer.determine_severity(error, ErrorType.SYSTEM)
        assert severity == ErrorSeverity.CRITICAL

        # 严重消息内容会提升严重程度
        error = ValueError("Critical validation failure")
        severity = severity_analyzer.determine_severity(error, ErrorType.VALIDATION)
        assert severity == ErrorSeverity.CRITICAL


class TestErrorProcessor:
    """测试错误处理器"""

    @pytest.fixture
    def error_processor(self):
        """创建错误处理器实例"""
        return ErrorProcessor()

    @pytest.fixture
    def mock_logger(self):
        """创建Mock日志器"""
        return Mock()

    def test_init(self, error_processor):
        """测试初始化"""
        assert error_processor is not None
        assert hasattr(error_processor, 'process_error')

    def test_process_error_basic(self, error_processor):
        """测试基本错误处理"""
        error_info = ErrorInfo(
            error_type=ErrorType.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Test error",
            context={"operation": "test"}
        )

        error_processor.process_error(error_info)

        # 验证错误已添加到历史记录
        assert len(error_processor.error_history) == 1
        assert error_processor.error_history[0] == error_info

    def test_register_error_handler(self, error_processor):
        """测试注册错误处理器"""
        def custom_handler(error_info: ErrorInfo):
            pass

        error_processor.register_error_handler(ErrorType.SYSTEM, custom_handler)
        assert ErrorType.SYSTEM in error_processor.error_handlers

    def test_register_global_error_handler(self, error_processor):
        """测试注册全局错误处理器"""
        def global_handler(error_info: ErrorInfo):
            pass

        error_processor.register_global_error_handler(global_handler)
        assert error_processor.global_error_handler == global_handler

    def test_process_error_with_handler(self, error_processor):
        """测试使用处理器处理错误"""
        handler_called = []

        def test_handler(error_info: ErrorInfo):
            handler_called.append(error_info)

        error_processor.register_error_handler(ErrorType.VALIDATION, test_handler)

        error_info = ErrorInfo(
            error_type=ErrorType.VALIDATION,
            severity=ErrorSeverity.LOW,
            message="Validation error"
        )

        error_processor.process_error(error_info)

        assert len(handler_called) == 1
        assert handler_called[0] == error_info

    def test_process_error_with_global_handler(self, error_processor):
        """测试使用全局处理器处理错误"""
        handler_called = []

        def global_handler(error_info: ErrorInfo):
            handler_called.append(error_info)

        error_processor.register_global_error_handler(global_handler)

        error_info = ErrorInfo(
            error_type=ErrorType.UNKNOWN,
            severity=ErrorSeverity.LOW,
            message="Unknown error"
        )

        error_processor.process_error(error_info)

        assert len(handler_called) == 1
        assert handler_called[0] == error_info

    def test_error_history_management(self, error_processor):
        """测试错误历史记录管理"""
        # 添加多个错误
        for i in range(5):
            error_info = ErrorInfo(
                error_type=ErrorType.VALIDATION,
                severity=ErrorSeverity.LOW,
                message=f"Error {i}"
            )
            error_processor.process_error(error_info)

        assert len(error_processor.error_history) == 5

        # 测试清空历史
        error_processor.clear_error_history()
        assert len(error_processor.error_history) == 0
        assert error_processor.error_count == 0


class TestErrorHandler:
    """测试错误处理器"""

    @pytest.fixture
    def error_handler(self):
        """创建错误处理器实例"""
        return ErrorHandler()

    @pytest.fixture
    def mock_logger(self):
        """创建Mock日志器"""
        return Mock()

    def test_init(self, error_handler):
        """测试初始化"""
        assert error_handler is not None
        assert hasattr(error_handler, '_classifier')
        assert hasattr(error_handler, '_severity_analyzer')
        assert hasattr(error_handler, '_processor')

    def test_handle_error(self, error_handler):
        """测试错误处理"""
        error = TimeoutError("Request timed out")

        result = error_handler.handle_error(error)

        assert isinstance(result, ErrorInfo)
        assert result.error_type == ErrorType.TIMEOUT
        assert result.severity == ErrorSeverity.MEDIUM
        assert result.message == "Request timed out"

    def test_handle_error_with_context(self, error_handler):
        """测试带上下文的错误处理"""
        error = ResourceError("Disk full")
        context = {"path": "/var/log", "available_space": "0MB"}

        result = error_handler.handle_error(error, context=context)

        assert isinstance(result, ErrorInfo)
        assert result.context == context

    def test_register_error_handler(self, error_handler):
        """测试注册错误处理器"""
        def custom_handler(error_info: ErrorInfo) -> None:
            pass

        error_handler.register_error_handler(ErrorType.SYSTEM, custom_handler)

        # 验证处理器已注册（通过_processor）
        assert ErrorType.SYSTEM in error_handler._processor.error_handlers

    def test_register_global_error_handler(self, error_handler):
        """测试注册全局错误处理器"""
        def global_handler(error_info: ErrorInfo) -> None:
            pass

        error_handler.register_global_error_handler(global_handler)

        # 验证全局处理器已注册
        assert error_handler._processor.global_error_handler == global_handler

    def test_get_error_statistics(self, error_handler):
        """测试获取错误统计信息"""
        # 先处理一些错误
        error_handler.handle_error(ValueError("test"))
        error_handler.handle_error(ConnectionError("test"))

        stats = error_handler.get_error_statistics()

        assert isinstance(stats, dict)
        assert "total_errors" in stats
        assert "error_count" in stats
        assert "error_types" in stats
        assert "severity_distribution" in stats
        assert "recent_errors" in stats

    def test_clear_error_history(self, error_handler):
        """测试清除错误历史"""
        # 先处理一些错误
        error_handler.handle_error(ValueError("test"))
        error_handler.handle_error(ConnectionError("test"))

        # 清除历史
        error_handler.clear_error_history()

        # 验证历史已清空
        stats = error_handler.get_error_statistics()
        assert stats["total_errors"] == 0
        assert len(stats["recent_errors"]) == 0

    def test_is_healthy(self, error_handler):
        """测试健康检查"""
        # 初始状态应该是健康的
        assert error_handler.is_healthy()

        # 处理一些严重错误
        for i in range(10):
            error_handler.handle_error(SystemError(f"Critical error {i}"))

        # 现在应该不健康
        assert not error_handler.is_healthy()

    def test_handle_errors_batch(self, error_handler):
        """测试批量错误处理"""
        errors = [
            ValueError("Error 1"),
            ConnectionError("Error 2"),
            TimeoutError("Error 3")
        ]

        results = error_handler.handle_errors_batch(errors)

        assert len(results) == 3
        assert all(isinstance(result, ErrorInfo) for result in results)

    def test_handle_errors_batch_empty(self, error_handler):
        """测试批量处理空错误列表"""
        results = error_handler.handle_errors_batch([])
        assert results == []
