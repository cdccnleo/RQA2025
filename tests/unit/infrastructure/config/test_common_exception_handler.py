# -*- coding: utf-8 -*-
#!/usr/bin/env python3

"""
测试通用异常处理工具

测试common_exception_handler.py中的异常处理功能
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
from unittest.mock import Mock, patch

from src.infrastructure.config.core.common_exception_handler import (
    ExceptionHandlingStrategy,
    LogLevel,
    ExceptionContext,
    ExceptionCollector,
    handle_exceptions,
    handle_config_exceptions,
    handle_cache_exceptions,
    handle_monitoring_exceptions,
    handle_validation_exceptions,
    global_exception_collector
)


class TestExceptionHandlingStrategy(unittest.TestCase):
    """测试异常处理策略枚举"""

    def test_exception_handling_strategy_values(self):
        """测试异常处理策略值"""
        self.assertEqual(ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT.value, "log_and_return_default")
        self.assertEqual(ExceptionHandlingStrategy.LOG_AND_RERAISE.value, "log_and_reraise")
        self.assertEqual(ExceptionHandlingStrategy.SILENT_RETURN_DEFAULT.value, "silent_return_default")
        self.assertEqual(ExceptionHandlingStrategy.COLLECT_AND_RETURN.value, "collect_and_return")


class TestLogLevel(unittest.TestCase):
    """测试日志级别枚举"""

    def test_log_level_values(self):
        """测试日志级别值"""
        self.assertEqual(LogLevel.DEBUG.value, 10)
        self.assertEqual(LogLevel.INFO.value, 20)
        self.assertEqual(LogLevel.WARNING.value, 30)
        self.assertEqual(LogLevel.ERROR.value, 40)
        self.assertEqual(LogLevel.CRITICAL.value, 50)


class TestExceptionContext(unittest.TestCase):
    """测试异常上下文"""

    def test_initialization(self):
        """测试初始化"""
        start_time = time.time()
        context = ExceptionContext(
            operation="test_operation",
            component="test_component",
            parameters={"key": "value"}
        )

        self.assertEqual(context.operation, "test_operation")
        self.assertEqual(context.component, "test_component")
        self.assertEqual(context.parameters, {"key": "value"})
        self.assertGreaterEqual(context.start_time, start_time)
        self.assertGreaterEqual(context.end_time, context.start_time)
        self.assertGreaterEqual(context.duration, 0)

    def test_initialization_defaults(self):
        """测试默认初始化"""
        context = ExceptionContext()

        self.assertEqual(context.operation, "")
        self.assertEqual(context.component, "")
        self.assertEqual(context.parameters, {})
        self.assertIsInstance(context.start_time, float)

    def test_to_dict(self):
        """测试转换为字典"""
        context = ExceptionContext(
            operation="test_op",
            component="test_comp",
            parameters={"param": "value"}
        )

        data = context.to_dict()

        self.assertEqual(data['operation'], "test_op")
        self.assertEqual(data['component'], "test_comp")
        self.assertEqual(data['parameters'], {"param": "value"})
        self.assertIsInstance(data['start_time'], float)
        self.assertIsInstance(data['end_time'], float)
        self.assertIsInstance(data['duration'], float)


class TestHandleExceptions(unittest.TestCase):
    """测试异常处理装饰器"""

    def test_successful_operation(self):
        """测试成功操作"""
        @handle_exceptions()
        def successful_func():
            return "success"

        result = successful_func()
        self.assertEqual(result, "success")

    def test_log_and_return_default(self):
        """测试记录并返回默认值策略"""
        @handle_exceptions(
            strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
            default_return="default"
        )
        def failing_func():
            raise ValueError("Test error")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = failing_func()

        self.assertEqual(result, "default")
        mock_logger.error.assert_called_once()

    def test_log_and_reraise(self):
        """测试记录并重新抛出策略"""
        @handle_exceptions(strategy=ExceptionHandlingStrategy.LOG_AND_RERAISE)
        def failing_func():
            raise ValueError("Test error")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            with self.assertRaises(ValueError):
                failing_func()

        mock_logger.error.assert_called_once()

    def test_silent_return_default(self):
        """测试静默返回默认值策略"""
        @handle_exceptions(
            strategy=ExceptionHandlingStrategy.SILENT_RETURN_DEFAULT,
            default_return="silent_default"
        )
        def failing_func():
            raise ValueError("Test error")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = failing_func()

        self.assertEqual(result, "silent_default")
        # 静默策略不应该记录日志
        mock_logger.error.assert_not_called()

    def test_collect_and_return(self):
        """测试收集并返回策略"""
        @handle_exceptions(strategy=ExceptionHandlingStrategy.COLLECT_AND_RETURN)
        def failing_func():
            raise ValueError("Test error")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = failing_func()

        self.assertIsInstance(result, dict)
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], "Test error")
        self.assertIsInstance(result['exception'], ValueError)

    def test_retry_mechanism(self):
        """测试重试机制"""
        call_count = 0

        @handle_exceptions(
            strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
            max_retries=2,
            retry_delay=0.01,
            default_return="retried"
        )
        def failing_then_success_func():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError(f"Attempt {call_count} failed")
            return "success"

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = failing_then_success_func()

        self.assertEqual(result, "success")
        self.assertEqual(call_count, 3)  # 失败2次后成功

    def test_retry_exhaustion(self):
        """测试重试耗尽"""
        call_count = 0

        @handle_exceptions(
            strategy=ExceptionHandlingStrategy.LOG_AND_RETURN_DEFAULT,
            max_retries=2,
            default_return="exhausted"
        )
        def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise ValueError(f"Attempt {call_count} failed")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = always_failing_func()

        self.assertEqual(result, "exhausted")
        self.assertEqual(call_count, 3)  # 3次尝试都失败

    def test_context_inclusion(self):
        """测试上下文包含"""
        @handle_exceptions(include_context=True)
        def func_with_context(self):
            raise ValueError("Context test")

        # 创建一个模拟的对象
        mock_obj = Mock()
        mock_obj.__class__.__name__ = "TestClass"

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = func_with_context(mock_obj)

        # 验证日志包含上下文信息
        mock_logger.error.assert_called_once()
        call_args = mock_logger.error.call_args[0][0]
        self.assertIn("TestClass.func_with_context", call_args)

    def test_custom_log_level(self):
        """测试自定义日志级别"""
        @handle_exceptions(log_level=LogLevel.WARNING)
        def failing_func():
            raise ValueError("Test error")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = failing_func()

        # 应该使用warning级别记录日志
        mock_logger.warning.assert_called_once()
        mock_logger.error.assert_not_called()

    def test_multiple_exceptions_in_retry(self):
        """测试重试过程中的多种异常"""
        exceptions = [ValueError("Error 1"), RuntimeError("Error 2"), Exception("Error 3")]
        call_count = 0

        @handle_exceptions(
            max_retries=2,
            default_return="final_default"
        )
        def multiple_exceptions_func():
            nonlocal call_count
            call_count += 1
            if call_count <= len(exceptions):
                raise exceptions[call_count - 1]
            return "success"

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = multiple_exceptions_func()

        self.assertEqual(result, "final_default")
        self.assertEqual(call_count, 3)  # max_retries=2意味着3次尝试


class TestConvenienceDecorators(unittest.TestCase):
    """测试便捷装饰器"""

    def test_handle_config_exceptions(self):
        """测试配置异常处理装饰器"""
        @handle_config_exceptions(default_return="config_default")
        def config_func():
            raise ValueError("Config error")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = config_func()

        self.assertEqual(result, "config_default")
        mock_logger.warning.assert_called_once()

    def test_handle_cache_exceptions(self):
        """测试缓存异常处理装饰器"""
        @handle_cache_exceptions(default_return="cache_default")
        def cache_func():
            raise ValueError("Cache error")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = cache_func()

        self.assertEqual(result, "cache_default")
        mock_logger.error.assert_called_once()

    def test_handle_monitoring_exceptions(self):
        """测试监控异常处理装饰器"""
        @handle_monitoring_exceptions(default_return="monitoring_default")
        def monitoring_func():
            raise ValueError("Monitoring error")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = monitoring_func()

        self.assertEqual(result, "monitoring_default")
        # 监控异常处理是静默的，不记录日志
        mock_logger.warning.assert_not_called()

    def test_handle_validation_exceptions(self):
        """测试验证异常处理装饰器"""
        @handle_validation_exceptions(default_return="validation_default")
        def validation_func():
            raise ValueError("Validation error")

        with patch('src.infrastructure.config.core.common_exception_handler.logger') as mock_logger:
            result = validation_func()

        # 验证异常使用COLLECT_AND_RETURN策略，返回字典
        self.assertIsInstance(result, dict)
        self.assertFalse(result['success'])
        self.assertEqual(result['error'], "Validation error")
        mock_logger.error.assert_called_once()


class TestGlobalExceptionCollector(unittest.TestCase):
    """测试全局异常收集器"""

    def setUp(self):
        """测试前准备"""
        # 清空收集器
        global_exception_collector.clear()

    def test_add_exception(self):
        """测试添加异常"""
        exception = ValueError("Test exception")
        context = ExceptionContext("test_op", "test_comp")

        global_exception_collector.add_exception(exception, context)

        exceptions = global_exception_collector.get_exceptions()
        self.assertEqual(len(exceptions), 1)

        collected = exceptions[0]
        self.assertEqual(collected['exception_type'], "ValueError")
        self.assertEqual(collected['message'], "Test exception")
        self.assertIsNotNone(collected['context'])

    def test_add_exception_without_context(self):
        """测试不带上下文添加异常"""
        exception = RuntimeError("Runtime error")

        global_exception_collector.add_exception(exception)

        exceptions = global_exception_collector.get_exceptions()
        self.assertEqual(len(exceptions), 1)

        collected = exceptions[0]
        self.assertEqual(collected['exception_type'], "RuntimeError")
        self.assertIsNone(collected['context'])

    def test_get_exceptions(self):
        """测试获取异常"""
        exceptions = global_exception_collector.get_exceptions()
        self.assertEqual(exceptions, [])

        # 添加异常
        exception = RuntimeError("Runtime error")
        global_exception_collector.add_exception(exception)

        exceptions = global_exception_collector.get_exceptions()
        self.assertEqual(len(exceptions), 1)

    def test_clear(self):
        """测试清除异常"""
        global_exception_collector.add_exception(ValueError("Error"))
        self.assertTrue(global_exception_collector.has_exceptions())

        global_exception_collector.clear()
        self.assertFalse(global_exception_collector.has_exceptions())

    def test_has_exceptions(self):
        """测试检查是否有异常"""
        self.assertFalse(global_exception_collector.has_exceptions())

        global_exception_collector.add_exception(ValueError("Error"))
        self.assertTrue(global_exception_collector.has_exceptions())

    def test_get_summary(self):
        """测试获取摘要"""
        # 空收集器摘要
        summary = global_exception_collector.get_summary()
        self.assertEqual(summary['total_count'], 0)
        self.assertEqual(summary['max_capacity'], 1000)

        # 添加异常后的摘要
        global_exception_collector.add_exception(ValueError("Error 1"))
        global_exception_collector.add_exception(ValueError("Error 2"))
        global_exception_collector.add_exception(RuntimeError("Error 3"))

        summary = global_exception_collector.get_summary()
        self.assertEqual(summary['total_count'], 3)
        self.assertEqual(summary['by_type']['ValueError'], 2)
        self.assertEqual(summary['by_type']['RuntimeError'], 1)

    def test_max_exceptions_limit(self):
        """测试最大异常数量限制"""
        # 创建一个小的收集器用于测试
        small_collector = ExceptionCollector(max_exceptions=3)

        # 添加超过限制的异常
        for i in range(5):
            small_collector.add_exception(ValueError(f"Error {i}"))

        exceptions = small_collector.get_exceptions()
        self.assertEqual(len(exceptions), 3)  # 应该只保留3个

    def test_thread_safety(self):
        """测试线程安全性"""
        import concurrent.futures

        def add_exceptions(thread_id):
            for i in range(10):
                global_exception_collector.add_exception(
                    ValueError(f"Thread {thread_id} error {i}")
                )

        # 使用多线程添加异常
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(add_exceptions, i) for i in range(3)]
            concurrent.futures.wait(futures)

        # 验证所有异常都被正确收集
        all_exceptions = global_exception_collector.get_exceptions()
        self.assertEqual(len(all_exceptions), 30)  # 3线程 * 10异常


if __name__ == '__main__':
    unittest.main()
