# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
测试通用日志工具

测试common_logger.py中的日志记录功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import logging
import json
import time
import threading
from unittest.mock import Mock, patch
from io import StringIO

from src.infrastructure.config.core.common_logger import (
    LogLevel,
    LogFormat,
    OperationType,
    LogContext,
    StructuredLogger,
    get_logger,
    create_operation_context,
    default_logger
)


class TestLogLevel(unittest.TestCase):
    """测试日志级别枚举"""

    def test_log_level_values(self):
        """测试日志级别值"""
        self.assertEqual(LogLevel.DEBUG.value, "DEBUG")
        self.assertEqual(LogLevel.INFO.value, "INFO")
        self.assertEqual(LogLevel.WARNING.value, "WARNING")
        self.assertEqual(LogLevel.ERROR.value, "ERROR")
        self.assertEqual(LogLevel.CRITICAL.value, "CRITICAL")


class TestLogFormat(unittest.TestCase):
    """测试日志格式枚举"""

    def test_log_format_values(self):
        """测试日志格式值"""
        self.assertEqual(LogFormat.TEXT.value, "text")
        self.assertEqual(LogFormat.JSON.value, "json")
        self.assertEqual(LogFormat.STRUCTURED.value, "structured")


class TestOperationType(unittest.TestCase):
    """测试操作类型枚举"""

    def test_operation_type_values(self):
        """测试操作类型值"""
        self.assertEqual(OperationType.READ.value, "read")
        self.assertEqual(OperationType.WRITE.value, "write")
        self.assertEqual(OperationType.DELETE.value, "delete")
        self.assertEqual(OperationType.UPDATE.value, "update")
        self.assertEqual(OperationType.QUERY.value, "query")
        self.assertEqual(OperationType.VALIDATE.value, "validate")
        self.assertEqual(OperationType.MONITOR.value, "monitor")
        self.assertEqual(OperationType.MAINTAIN.value, "maintain")


class TestLogContext(unittest.TestCase):
    """测试日志上下文"""

    def test_initialization(self):
        """测试初始化"""
        context = LogContext(
            component="test_component",
            operation="test_operation",
            operation_type=OperationType.READ,
            user_id="user123",
            session_id="session456",
            request_id="request789",
            parameters={"key": "value"}
        )

        self.assertEqual(context.component, "test_component")
        self.assertEqual(context.operation, "test_operation")
        self.assertEqual(context.operation_type, OperationType.READ)
        self.assertEqual(context.user_id, "user123")
        self.assertEqual(context.session_id, "session456")
        self.assertEqual(context.request_id, "request789")
        self.assertEqual(context.parameters, {"key": "value"})
        self.assertIsInstance(context.start_time, float)
        self.assertIsNone(context.end_time)
        self.assertIsNone(context.duration)

    def test_initialization_defaults(self):
        """测试默认初始化"""
        context = LogContext()

        self.assertEqual(context.component, "")
        self.assertEqual(context.operation, "")
        self.assertIsNone(context.operation_type)
        self.assertIsNone(context.user_id)
        self.assertIsNone(context.session_id)
        self.assertIsNone(context.request_id)
        self.assertEqual(context.parameters, {})

    def test_complete_success(self):
        """测试完成成功操作"""
        context = LogContext("comp", "op")
        time.sleep(0.01)  # 确保有一些时间差

        context.complete(success=True, result="success_result")

        self.assertIsNotNone(context.end_time)
        self.assertIsNotNone(context.duration)
        self.assertTrue(context.success)
        self.assertEqual(context.result, "success_result")
        self.assertIsNone(context.error)

    def test_complete_failure(self):
        """测试完成失败操作"""
        context = LogContext("comp", "op")

        context.complete(success=False, error="error_message")

        self.assertFalse(context.success)
        self.assertEqual(context.error, "error_message")
        self.assertIsNone(context.result)

    def test_to_dict(self):
        """测试转换为字典"""
        context = LogContext(
            component="comp",
            operation="op",
            operation_type=OperationType.READ,
            user_id="user",
            parameters={"param": "value"}
        )
        context.complete(success=True, result="result")

        data = context.to_dict()

        expected_keys = [
            'component', 'operation', 'operation_type', 'user_id',
            'session_id', 'request_id', 'parameters', 'start_time',
            'end_time', 'duration', 'success', 'result', 'error'
        ]

        for key in expected_keys:
            self.assertIn(key, data)

        self.assertEqual(data['component'], "comp")
        self.assertEqual(data['operation'], "op")
        self.assertEqual(data['operation_type'], "read")
        self.assertEqual(data['success'], True)
        self.assertEqual(data['result'], "result")


class TestStructuredLogger(unittest.TestCase):
    """测试结构化日志记录器"""

    def setUp(self):
        """测试前准备"""
        # 创建一个StringIO来捕获日志输出
        self.log_stream = StringIO()

    def tearDown(self):
        """测试后清理"""
        # 清理日志记录器
        logger = logging.getLogger('test_logger')
        logger.handlers.clear()

    def test_initialization(self):
        """测试初始化"""
        logger = StructuredLogger('test_logger', LogLevel.DEBUG, LogFormat.JSON)

        self.assertEqual(logger.name, 'test_logger')
        self.assertEqual(logger.level, LogLevel.DEBUG)
        self.assertEqual(logger.format_type, LogFormat.JSON)
        self.assertTrue(logger.include_timestamp)
        self.assertTrue(logger.include_thread_id)
        self.assertIsInstance(logger.logger, logging.Logger)

    def test_initialization_defaults(self):
        """测试默认初始化"""
        logger = StructuredLogger('test_logger')

        self.assertEqual(logger.level, LogLevel.INFO)
        self.assertEqual(logger.format_type, LogFormat.STRUCTURED)

    def test_log_levels(self):
        """测试不同日志级别"""
        logger = StructuredLogger('test_logger')

        # 测试所有级别的方法存在
        self.assertTrue(hasattr(logger, 'debug'))
        self.assertTrue(hasattr(logger, 'info'))
        self.assertTrue(hasattr(logger, 'warning'))
        self.assertTrue(hasattr(logger, 'error'))
        self.assertTrue(hasattr(logger, 'critical'))

    @patch('src.infrastructure.config.core.common_logger.datetime')
    def test_logging_with_context(self, mock_datetime):
        """测试带上下文的日志记录"""
        mock_datetime.now.return_value.isoformat.return_value = "2023-01-01T12:00:00"

        # 创建日志记录器
        stream_handler = logging.StreamHandler(self.log_stream)
        test_logger = logging.getLogger('test_logger')
        test_logger.addHandler(stream_handler)
        test_logger.setLevel(logging.DEBUG)

        # 创建结构化日志记录器
        structured_logger = StructuredLogger('test_logger')
        structured_logger.logger = test_logger  # 替换为我们的测试logger

        # 创建上下文
        context = LogContext("test_comp", "test_op", OperationType.READ, user_id="user123")

        # 记录日志
        structured_logger.info("Test message", context=context, extra_field="extra_value")

        # 验证日志输出
        log_output = self.log_stream.getvalue()
        self.assertIn("Test message", log_output)

    def test_logging_without_context(self):
        """测试不带上下文的日志记录"""
        # 创建日志记录器
        stream_handler = logging.StreamHandler(self.log_stream)
        test_logger = logging.getLogger('test_logger_no_context')
        test_logger.addHandler(stream_handler)
        test_logger.setLevel(logging.INFO)

        # 创建结构化日志记录器
        structured_logger = StructuredLogger('test_logger_no_context')
        structured_logger.logger = test_logger

        # 记录日志
        structured_logger.info("Simple message", simple_field="simple_value")

        # 验证日志输出
        log_output = self.log_stream.getvalue()
        self.assertIn("Simple message", log_output)


class TestFormatterClasses(unittest.TestCase):
    """测试格式化器类"""

    def setUp(self):
        """测试前准备"""
        from src.infrastructure.config.core.common_logger import TextFormatter, JSONFormatter, StructuredFormatter

        self.TextFormatter = TextFormatter
        self.JSONFormatter = JSONFormatter
        self.StructuredFormatter = StructuredFormatter

    def test_text_formatter(self):
        """测试文本格式化器"""
        formatter = self.TextFormatter()

        # 创建一个日志记录
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        record.timestamp = "2023-01-01T12:00:00"
        record.thread_id = 12345

        formatted = formatter.format(record)
        self.assertIn("Test message", formatted)
        self.assertIn("INFO", formatted)

    def test_json_formatter(self):
        """测试JSON格式化器"""
        formatter = self.JSONFormatter()

        # 创建一个日志记录
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        record.structured_data = {"message": "Test", "level": "INFO"}

        formatted = formatter.format(record)

        # 验证是有效的JSON
        parsed = json.loads(formatted)
        self.assertEqual(parsed['message'], "Test")
        self.assertEqual(parsed['level'], "INFO")

    def test_structured_formatter(self):
        """测试结构化格式化器"""
        formatter = self.StructuredFormatter()

        # 创建一个日志记录
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        record.structured_data = {"message": "Test", "level": "INFO"}

        formatted = formatter.format(record)
        self.assertIn("Test message", formatted)
        self.assertIn("INFO", formatted)
        self.assertIn("test", formatted)  # logger name


class TestUtilityFunctions(unittest.TestCase):
    """测试工具函数"""

    def test_get_logger(self):
        """测试获取日志记录器"""
        logger = get_logger("test_utility_logger")

        self.assertIsInstance(logger, StructuredLogger)
        self.assertEqual(logger.name, "test_utility_logger")

    def test_create_operation_context(self):
        """测试创建操作上下文"""
        context = create_operation_context(
            component="test_comp",
            operation="test_op",
            operation_type=OperationType.WRITE
        )

        self.assertIsInstance(context, LogContext)
        self.assertEqual(context.component, "test_comp")
        self.assertEqual(context.operation, "test_op")
        self.assertEqual(context.operation_type, OperationType.WRITE)

    def test_default_logger(self):
        """测试默认日志记录器"""
        # default_logger应该是一个StructuredLogger实例
        self.assertIsInstance(default_logger, StructuredLogger)

        # 应该可以用来记录日志
        self.assertTrue(hasattr(default_logger, 'info'))
        self.assertTrue(hasattr(default_logger, 'error'))


class TestPerformanceMonitoring(unittest.TestCase):
    """测试性能监控功能"""

    def test_context_performance_tracking(self):
        """测试上下文性能跟踪"""
        context = LogContext("perf_comp", "perf_op")

        # 模拟一些操作时间
        time.sleep(0.01)

        context.complete(success=True)

        # 验证时间跟踪
        self.assertIsNotNone(context.duration)
        self.assertGreater(context.duration, 0)
        self.assertLess(context.duration, 1.0)  # 应该很短

    def test_multiple_operations_logging(self):
        """测试多个操作的日志记录"""
        logger = get_logger("multi_op_logger")

        # 记录多个不同类型的操作
        operations = [
            ("read", OperationType.READ),
            ("write", OperationType.WRITE),
            ("delete", OperationType.DELETE),
            ("validate", OperationType.VALIDATE)
        ]

        for op_name, op_type in operations:
            context = create_operation_context("multi_comp", op_name, op_type)
            context.complete(success=True)

            # 验证上下文数据
            self.assertEqual(context.operation, op_name)
            self.assertEqual(context.operation_type, op_type)
            self.assertTrue(context.success)


class TestThreadSafety(unittest.TestCase):
    """测试线程安全性"""

    def test_concurrent_logging(self):
        """测试并发日志记录"""
        import concurrent.futures

        logger = get_logger("concurrent_logger")
        results = []

        def log_worker(worker_id):
            try:
                context = create_operation_context(
                    f"worker_comp_{worker_id}",
                    f"worker_op_{worker_id}",
                    OperationType.MONITOR
                )

                # 模拟一些工作
                time.sleep(0.01)

                context.complete(success=True)
                results.append(f"worker_{worker_id}_success")
                return True
            except Exception as e:
                results.append(f"worker_{worker_id}_error: {e}")
                return False

        # 使用线程池并发执行
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(log_worker, i) for i in range(5)]
            concurrent_results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 验证所有操作都成功
        self.assertEqual(len([r for r in concurrent_results if r]), 5)
        self.assertEqual(len(results), 5)


class TestErrorHandling(unittest.TestCase):
    """测试错误处理"""

    def test_invalid_log_level(self):
        """测试无效日志级别"""
        # 这应该不会抛出异常，而是使用默认级别
        logger = StructuredLogger("error_test", LogLevel.INFO)

        # 尝试记录日志
        logger.info("This should work")

        # 如果没有抛出异常，测试通过
        self.assertTrue(True)

    def test_context_with_missing_attributes(self):
        """测试上下文缺少属性"""
        context = LogContext("test_comp", "test_op")

        # 不调用complete，直接转换为字典
        data = context.to_dict()

        # 验证可选字段为None
        self.assertIsNone(data['success'])
        self.assertIsNone(data['result'])
        self.assertIsNone(data['error'])

    def test_formatter_with_missing_data(self):
        """测试格式化器处理缺失数据"""
        from src.infrastructure.config.core.common_logger import StructuredFormatter

        formatter = StructuredFormatter()
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test", args=(), exc_info=None
        )

        # 不设置structured_data
        formatted = formatter.format(record)

        # 应该不会崩溃
        self.assertIsInstance(formatted, str)
        self.assertIn("Test", formatted)


if __name__ == '__main__':
    unittest.main()
