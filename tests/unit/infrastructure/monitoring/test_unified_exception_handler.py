#!/usr/bin/env python3
"""
RQA2025 基础设施层统一异常处理器单元测试

测试统一异常处理框架的功能和正确性。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import patch, MagicMock

from src.infrastructure.monitoring.core.unified_exception_handler import (
    MonitoringException,
    ValidationError,
    ConfigurationError,
    ConnectionError,
    DataPersistenceError,
    AlertProcessingError,
    NotificationError,
    ExceptionHandlingStrategy,
    LogAndContinueStrategy,
    RetryStrategy,
    RaiseStrategy,
    ExceptionHandler,
    handle_monitoring_exception,
    with_exception_handling,
    global_exception_handler
)


class TestMonitoringExceptions(unittest.TestCase):
    """监控异常测试类"""

    def test_monitoring_exception_creation(self):
        """测试监控异常创建"""
        exception = MonitoringException(
            "Test error message",
            error_code="TEST_ERROR",
            context={"key": "value"},
            cause=ValueError("Original error")
        )

        self.assertEqual(exception.message, "Test error message")
        self.assertEqual(exception.error_code, "TEST_ERROR")
        self.assertEqual(exception.context, {"key": "value"})
        self.assertIsInstance(exception.cause, ValueError)
        self.assertIsNotNone(exception.timestamp)

    def test_monitoring_exception_to_dict(self):
        """测试异常转换为字典"""
        exception = MonitoringException("Test message")
        data = exception.to_dict()

        self.assertEqual(data['message'], "Test message")
        self.assertEqual(data['error_code'], "MONITORING_ERROR")
        self.assertEqual(data['context'], {})
        self.assertIsNone(data['cause'])
        self.assertIn('timestamp', data)
        self.assertIn('traceback', data)

    def test_validation_error(self):
        """测试验证异常"""
        exception = ValidationError("Invalid value", field="username", value="invalid")

        self.assertEqual(exception.error_code, "VALIDATION_ERROR")
        self.assertEqual(exception.context['field'], "username")
        self.assertEqual(exception.context['value'], "invalid")

    def test_configuration_error(self):
        """测试配置异常"""
        exception = ConfigurationError("Config missing", config_key="database.host")

        self.assertEqual(exception.error_code, "CONFIGURATION_ERROR")
        self.assertEqual(exception.context['config_key'], "database.host")

    def test_connection_error(self):
        """测试连接异常"""
        exception = ConnectionError("Connection failed", host="localhost", port=5432)

        self.assertEqual(exception.error_code, "CONNECTION_ERROR")
        self.assertEqual(exception.context['host'], "localhost")
        self.assertEqual(exception.context['port'], 5432)

    def test_data_persistence_error(self):
        """测试数据持久化异常"""
        exception = DataPersistenceError("Save failed", operation="insert", data_size=1024)

        self.assertEqual(exception.error_code, "DATA_PERSISTENCE_ERROR")
        self.assertEqual(exception.context['operation'], "insert")
        self.assertEqual(exception.context['data_size'], 1024)

    def test_alert_processing_error(self):
        """测试告警处理异常"""
        exception = AlertProcessingError("Alert failed", alert_id="alert_001", rule_id="rule_001")

        self.assertEqual(exception.error_code, "ALERT_PROCESSING_ERROR")
        self.assertEqual(exception.context['alert_id'], "alert_001")
        self.assertEqual(exception.context['rule_id'], "rule_001")

    def test_notification_error(self):
        """测试通知异常"""
        exception = NotificationError("Send failed", channel="email", recipient="admin@example.com")

        self.assertEqual(exception.error_code, "NOTIFICATION_ERROR")
        self.assertEqual(exception.context['channel'], "email")
        self.assertEqual(exception.context['recipient'], "admin@example.com")


class TestExceptionHandlingStrategies(unittest.TestCase):
    """异常处理策略测试类"""

    def test_log_and_continue_strategy(self):
        """测试记录并继续策略"""
        strategy = LogAndContinueStrategy()
        context = {
            'operation': 'test_operation',
            'default_value': 'default_result'
        }

        result = strategy.handle(ValueError("Test error"), context)
        self.assertEqual(result, 'default_result')

    def test_retry_strategy_success(self):
        """测试重试策略成功情况"""
        strategy = RetryStrategy(max_retries=2, delay_seconds=0.01)

        call_count = 0
        def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("Temporary error")
            return "success"

        context = {
            'operation': 'test_retry',
            'func': mock_func,
            'args': (),
            'kwargs': {}
        }

        result = strategy.handle(ValueError("Initial error"), context)
        self.assertEqual(result, "success")
        self.assertEqual(call_count, 2)

    def test_retry_strategy_failure(self):
        """测试重试策略失败情况"""
        strategy = RetryStrategy(max_retries=2, delay_seconds=0.01)

        call_count = 0
        def mock_func():
            nonlocal call_count
            call_count += 1
            raise ValueError("Persistent error")

        context = {
            'operation': 'test_retry_fail',
            'func': mock_func,
            'args': (),
            'kwargs': {}
        }

        with self.assertRaises(ValueError):
            strategy.handle(ValueError("Initial error"), context)

        self.assertEqual(call_count, 3)  # 初始调用 + 2次重试

    def test_raise_strategy(self):
        """测试直接抛出策略"""
        strategy = RaiseStrategy()
        context = {'operation': 'test_raise'}

        with self.assertRaises(ValueError):
            strategy.handle(ValueError("Test error"), context)


class TestExceptionHandler(unittest.TestCase):
    """异常处理器测试类"""

    def setUp(self):
        """测试设置"""
        self.handler = ExceptionHandler()

    def test_add_strategy(self):
        """测试添加策略"""
        custom_strategy = LogAndContinueStrategy()
        self.handler.add_strategy('custom', custom_strategy)

        self.assertIn('custom', self.handler.strategies)

    def test_default_strategy_mapping(self):
        """测试默认策略映射"""
        # ValidationError 应该使用 'raise' 策略
        self.assertEqual(self.handler._get_default_strategy(ValidationError("test")), 'raise')

        # ConnectionError 应该使用 'retry' 策略
        self.assertEqual(self.handler._get_default_strategy(ConnectionError("test")), 'retry')

        # 通用异常应该使用 'log_and_continue' 策略
        self.assertEqual(self.handler._get_default_strategy(ValueError("test")), 'log_and_continue')

    def test_handle_exception_with_custom_strategy(self):
        """测试使用自定义策略处理异常"""
        result = self.handler.handle_exception(
            ValueError("Test error"),
            operation="test_operation",
            strategy="log_and_continue",
            context={'default_value': 'success'}
        )

        self.assertEqual(result, 'success')

    def test_handle_exception_with_retry_strategy(self):
        """测试使用重试策略处理异常"""
        call_count = 0
        def mock_func():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection failed")
            return "connected"

        result = self.handler.handle_exception(
            ConnectionError("Initial connection error"),
            operation="database_connect",
            strategy="retry",
            context={
                'func': mock_func,
                'args': (),
                'kwargs': {}
            }
        )

        self.assertEqual(result, "connected")
        self.assertEqual(call_count, 2)

    @patch('src.infrastructure.monitoring.core.unified_exception_handler.logger')
    def test_handle_exception_with_raise_strategy(self, mock_logger):
        """测试使用抛出策略处理异常"""
        with self.assertRaises(ValidationError):
            self.handler.handle_exception(
                ValidationError("Validation failed"),
                operation="data_validation",
                strategy="raise"
            )

        # 验证日志被调用
        mock_logger.error.assert_called()


class TestExceptionDecorators(unittest.TestCase):
    """异常装饰器测试类"""

    def test_handle_monitoring_exception_decorator_success(self):
        """测试异常处理装饰器成功情况"""
        @handle_monitoring_exception("test_function")
        def test_func():
            return "success"

        result = test_func()
        self.assertEqual(result, "success")

    def test_handle_monitoring_exception_decorator_error(self):
        """测试异常处理装饰器错误情况"""
        @handle_monitoring_exception("test_function", strategy="log_and_continue")
        def test_func():
            raise ValueError("Test error")
            return "should not reach here"

        result = test_func()
        self.assertIsNone(result)  # 默认返回值

    def test_with_exception_handling_context_success(self):
        """测试异常处理上下文管理器成功情况"""
        with with_exception_handling("test_context") as ctx:
            result = "success"

        self.assertEqual(result, "success")

    def test_with_exception_handling_context_error(self):
        """测试异常处理上下文管理器错误情况"""
        try:
            with with_exception_handling("test_context", strategy="log_and_continue") as ctx:
                raise ValueError("Test error")
                result = "should not reach here"
        except:
            self.fail("异常应该被处理，不应该抛出")

        # 如果没有异常抛出，说明异常被正确处理了
        self.assertTrue(True)


class TestGlobalExceptionHandler(unittest.TestCase):
    """全局异常处理器测试类"""

    def test_global_exception_handler(self):
        """测试全局异常处理器"""
        result = global_exception_handler.handle_exception(
            ValueError("Global test"),
            operation="global_test",
            strategy="log_and_continue",
            context={'default_value': 'global_success'}
        )

        self.assertEqual(result, 'global_success')

    def test_convenience_function(self):
        """测试便捷函数"""
        from src.infrastructure.monitoring.core.unified_exception_handler import handle_exception

        result = handle_exception(
            ValueError("Convenience test"),
            operation="convenience_test",
            strategy="log_and_continue",
            context={'default_value': 'convenience_success'}
        )

        self.assertEqual(result, 'convenience_success')


if __name__ == '__main__':
    unittest.main()
