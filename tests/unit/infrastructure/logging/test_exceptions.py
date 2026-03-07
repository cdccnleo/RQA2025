#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试基础设施层 - 日志系统异常处理

测试logging/exceptions.py中的所有类和方法
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import time
from unittest.mock import Mock, patch


class TestExceptions:
    """测试日志系统异常处理"""

    def setup_method(self):
        """测试前准备"""
        try:
            from src.infrastructure.logging.exceptions import (
                ErrorContext, EngineException, ErrorCode, create_error_context
            )
            self.ErrorContext = ErrorContext
            self.EngineException = EngineException
            self.ErrorCode = ErrorCode
            self.create_error_context = create_error_context
        except ImportError as e:
            pytest.skip(f"Exception components not available: {e}")

    def test_error_context_creation(self):
        """测试错误上下文创建"""
        if not hasattr(self, 'ErrorContext'):
            pytest.skip("ErrorContext not available")

        context = self.ErrorContext(
            module="test_module",
            function="test_function",
            line_number=42,
            timestamp=1234567890.0,
            additional_info={"key": "value"}
        )

        assert context.module == "test_module"
        assert context.function == "test_function"
        assert context.line_number == 42
        assert context.timestamp == 1234567890.0
        assert context.additional_info["key"] == "value"

    def test_engine_exception_creation(self):
        """测试引擎异常创建"""
        if not hasattr(self, 'EngineException'):
            pytest.skip("EngineException not available")

        exception = self.EngineException(
            message="Test error message",
            error_code=500
        )

        assert str(exception) == "Test error message"
        assert hasattr(exception, 'error_code')

    def test_engine_exception_with_context(self):
        """测试带上下文的引擎异常"""
        if not all(hasattr(self, cls) for cls in ['EngineException', 'ErrorContext']):
            pytest.skip("Required exception components not available")

        context = self.ErrorContext(
            module="test_module",
            function="test_function",
            line_number=100
        )

        exception = self.EngineException(
            message="Context test error",
            context=context,
            severity="high"
        )

        assert str(exception) == "Context test error"
        assert exception.context == context
        assert exception.details["severity"] == "high"

    def test_engine_exception_to_dict(self):
        """测试引擎异常转换为字典"""
        if not all(hasattr(self, cls) for cls in ['EngineException', 'ErrorContext']):
            pytest.skip("Required exception components not available")

        context = self.ErrorContext(
            module="test_module",
            function="test_function",
            line_number=50
        )

        exception = self.EngineException(
            message="Dict test error",
            context=context,
            category="validation"
        )

        result = exception.to_dict()

        assert isinstance(result, dict)
        assert result["message"] == "Dict test error"
        assert result["context"]["module"] == "test_module"
        assert result["details"]["category"] == "validation"

    def test_engine_exception_without_context(self):
        """测试不带上下文的引擎异常"""
        if not hasattr(self, 'EngineException'):
            pytest.skip("EngineException not available")

        exception = self.EngineException("Simple error message")

        assert str(exception) == "Simple error message"
        assert exception.context is None
        assert exception.details == {}

        result = exception.to_dict()
        assert result["context"] is None
        assert result["details"] == {}

    def test_error_code_constants(self):
        """测试错误代码常量"""
        if not hasattr(self, 'ErrorCode'):
            pytest.skip("ErrorCode not available")

        # 验证错误代码常量存在
        assert hasattr(self.ErrorCode, 'SYSTEM_ERROR')
        assert hasattr(self.ErrorCode, 'UNKNOWN_ERROR')

        assert self.ErrorCode.SYSTEM_ERROR == 9000
        assert self.ErrorCode.UNKNOWN_ERROR == 9999

    def test_create_error_context_basic(self):
        """测试基本错误上下文创建"""
        if not hasattr(self, 'create_error_context'):
            pytest.skip("create_error_context not available")

        context = self.create_error_context(
            module="test_module",
            function="test_function",
            line_number=25
        )

        assert context.module == "test_module"
        assert context.function == "test_function"
        assert context.line_number == 25
        assert isinstance(context.timestamp, float)
        assert context.additional_info is not None

    def test_create_error_context_with_kwargs(self):
        """测试带额外参数的错误上下文创建"""
        if not hasattr(self, 'create_error_context'):
            pytest.skip("create_error_context not available")

        context = self.create_error_context(
            module="test_module",
            function="test_function",
            line_number=30,
            error_type="validation_error",
            severity="high",
            user_id="user123"
        )

        assert context.module == "test_module"
        assert context.function == "test_function"
        assert context.line_number == 30
        assert context.additional_info["error_type"] == "validation_error"
        assert context.additional_info["severity"] == "high"
        assert context.additional_info["user_id"] == "user123"

    def test_exception_inheritance(self):
        """测试异常继承关系"""
        if not hasattr(self, 'EngineException'):
            pytest.skip("EngineException not available")

        exception = self.EngineException("Inheritance test")

        # 验证继承关系
        assert isinstance(exception, Exception)
        assert isinstance(exception, self.EngineException)

    def test_exception_details_storage(self):
        """测试异常详细信息存储"""
        if not hasattr(self, 'EngineException'):
            pytest.skip("EngineException not available")

        details = {
            "operation": "database_query",
            "table": "users",
            "query_time": 2.5,
            "error_code": "CONNECTION_TIMEOUT"
        }

        exception = self.EngineException(
            message="Database operation failed",
            **details
        )

        assert exception.details == details
        assert exception.details["operation"] == "database_query"
        assert exception.details["query_time"] == 2.5

    def test_error_context_timestamp(self):
        """测试错误上下文时间戳"""
        if not hasattr(self, 'create_error_context'):
            pytest.skip("create_error_context not available")

        before_time = time.time()
        context = self.create_error_context(
            module="test",
            function="test_func",
            line_number=1
        )
        after_time = time.time()

        # 验证时间戳在合理范围内
        assert before_time <= context.timestamp <= after_time

    def test_exception_serialization(self):
        """测试异常序列化"""
        if not all(hasattr(self, cls) for cls in ['EngineException', 'ErrorContext']):
            pytest.skip("Required exception components not available")

        context = self.ErrorContext(
            module="serialization_test",
            function="test_method",
            line_number=123,
            timestamp=1609459200.0,  # 2021-01-01 00:00:00
            additional_info={"test": "data"}
        )

        exception = self.EngineException(
            message="Serialization test error",
            context=context,
            error_type="test_error"
        )

        result = exception.to_dict()

        # 验证序列化结果
        assert result["message"] == "Serialization test error"
        assert result["context"]["module"] == "serialization_test"
        assert result["context"]["line_number"] == 123
        assert result["details"]["error_type"] == "test_error"

    def test_exception_message_formatting(self):
        """测试异常消息格式化"""
        if not hasattr(self, 'EngineException'):
            pytest.skip("EngineException not available")

        # 测试简单消息
        exception1 = self.EngineException("Simple message")
        assert str(exception1) == "Simple message"

        # 测试复杂消息
        exception2 = self.EngineException(
            message="Complex error in module {module} at line {line}",
            module="test_module",
            line=42
        )
        assert str(exception2) == "Complex error in module {module} at line {line}"

    def test_error_context_dict_conversion(self):
        """测试错误上下文字典转换"""
        if not hasattr(self, 'ErrorContext'):
            pytest.skip("ErrorContext not available")

        context = self.ErrorContext(
            module="dict_test",
            function="test_func",
            line_number=99,
            timestamp=1609459200.0,
            additional_info={"key1": "value1", "key2": 42}
        )

        # 测试上下文对象的字典表示
        context_dict = context.__dict__

        assert context_dict["module"] == "dict_test"
        assert context_dict["function"] == "test_func"
        assert context_dict["line_number"] == 99
        assert context_dict["timestamp"] == 1609459200.0
        assert context_dict["additional_info"]["key1"] == "value1"
        assert context_dict["additional_info"]["key2"] == 42

    def test_exception_with_empty_details(self):
        """测试空详细信息的异常"""
        if not hasattr(self, 'EngineException'):
            pytest.skip("EngineException not available")

        exception = self.EngineException("Empty details test")

        assert exception.details == {}
        assert exception.context is None

        result = exception.to_dict()
        assert result["details"] == {}
        assert result["context"] is None

    def test_error_context_with_none_values(self):
        """测试包含None值的错误上下文"""
        if not hasattr(self, 'ErrorContext'):
            pytest.skip("ErrorContext not available")

        context = self.ErrorContext(
            module="none_test",
            function="test_func",
            line_number=0,
            timestamp=None,
            additional_info=None
        )

        assert context.module == "none_test"
        assert context.function == "test_func"
        assert context.line_number == 0
        assert context.timestamp is None
        assert context.additional_info is None

    def test_create_error_context_timestamp_range(self):
        """测试错误上下文创建的时间戳范围"""
        if not hasattr(self, 'create_error_context'):
            pytest.skip("create_error_context not available")

        start_time = time.time()

        # 创建多个上下文，确保时间戳递增
        contexts = []
        for i in range(5):
            context = self.create_error_context(
                module=f"module_{i}",
                function=f"func_{i}",
                line_number=i
            )
            contexts.append(context)
            time.sleep(0.001)  # 短暂延迟

        end_time = time.time()

        # 验证所有时间戳都在合理范围内
        for context in contexts:
            assert start_time <= context.timestamp <= end_time

        # 验证时间戳是递增的
        timestamps = [ctx.timestamp for ctx in contexts]
        assert timestamps == sorted(timestamps)


if __name__ == '__main__':
    pytest.main([__file__])
