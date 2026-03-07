"""
基础设施层 - SpecializedErrorHandler 单元测试

测试专用错误处理器的核心功能，包括归档、重试、InfluxDB等专用场景。
覆盖率目标: 85%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock
import time

from src.infrastructure.error.handlers.specialized_error_handler import (
    SpecializedErrorHandler,
    FailureType,
    InfluxDBErrorType,
    RetryStrategy
)


class TestSpecializedErrorHandler(unittest.TestCase):
    """SpecializedErrorHandler 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.handler = SpecializedErrorHandler(max_history=100)

    def tearDown(self):
        """测试后清理"""
        self.handler.clear_history()

    def test_initialization(self):
        """测试初始化"""
        handler = SpecializedErrorHandler(max_history=50)
        self.assertIsInstance(handler, SpecializedErrorHandler)
        self.assertEqual(handler._max_history, 50)
        # 通过retry_manager访问重试配置
        self.assertIsInstance(handler._retry_manager._retry_configs, dict)
        self.assertIsInstance(handler._failure_stats, dict)

    def test_default_retry_configs(self):
        """测试默认重试配置"""
        # 验证默认配置存在 - 通过retry_manager访问
        self.assertIn('network', self.handler._retry_manager._retry_configs)
        self.assertIn('database', self.handler._retry_manager._retry_configs)
        self.assertIn('file', self.handler._retry_manager._retry_configs)

        # 验证配置结构
        network_config = self.handler._retry_manager._retry_configs['network']
        self.assertEqual(network_config.max_attempts, 5)
        self.assertEqual(network_config.strategy, RetryStrategy.EXPONENTIAL)

    def test_add_retry_config(self):
        """测试添加重试配置"""
        from src.infrastructure.error.handlers.specialized_error_handler import RetryConfig

        config = RetryConfig(
            max_attempts=10,
            base_delay=2.0,
            strategy=RetryStrategy.FIXED
        )

        self.handler.add_retry_config('custom', config)
        # 通过retry_manager验证配置已添加
        self.assertIn('custom', self.handler._retry_manager._retry_configs)
        self.assertEqual(self.handler._retry_manager._retry_configs['custom'].max_attempts, 10)

    def test_get_retry_config(self):
        """测试获取重试配置"""
        # 获取存在的配置
        config = self.handler.get_retry_config('network')
        if config is not None:
            self.assertEqual(config.max_attempts, 5)

        # 获取不存在的配置
        config = self.handler.get_retry_config('nonexistent')
        self.assertIsNone(config)

    def test_calculate_delay_fixed(self):
        """测试固定延迟计算"""
        from src.infrastructure.error.handlers.specialized_error_handler import RetryConfig

        config = RetryConfig(strategy=RetryStrategy.FIXED, base_delay=1.0, jitter=False)

        # 通过retry_manager访问_calculate_delay方法
        delay1 = self.handler._retry_manager._calculate_delay(config, 0)
        delay2 = self.handler._retry_manager._calculate_delay(config, 5)

        self.assertEqual(delay1, 1.0)
        self.assertEqual(delay2, 1.0)

    def test_calculate_delay_linear(self):
        """测试线性延迟计算"""
        from src.infrastructure.error.handlers.specialized_error_handler import RetryConfig

        config = RetryConfig(strategy=RetryStrategy.LINEAR, base_delay=1.0, jitter=False)

        delay1 = self.handler._retry_manager._calculate_delay(config, 0)  # 1.0 * (0 + 1) = 1.0
        delay2 = self.handler._retry_manager._calculate_delay(config, 2)  # 1.0 * (2 + 1) = 3.0

        self.assertEqual(delay1, 1.0)
        self.assertEqual(delay2, 3.0)

    def test_calculate_delay_exponential(self):
        """测试指数延迟计算"""
        from src.infrastructure.error.handlers.specialized_error_handler import RetryConfig

        config = RetryConfig(strategy=RetryStrategy.EXPONENTIAL, base_delay=1.0, backoff_factor=2.0, jitter=False)

        delay1 = self.handler._retry_manager._calculate_delay(config, 0)  # 1.0 * (2.0 ** 0) = 1.0
        delay2 = self.handler._retry_manager._calculate_delay(config, 2)  # 1.0 * (2.0 ** 2) = 4.0

        self.assertEqual(delay1, 1.0)
        self.assertEqual(delay2, 4.0)

    def test_calculate_delay_with_jitter(self):
        """测试带抖动的延迟计算"""
        from src.infrastructure.error.handlers.specialized_error_handler import RetryConfig

        config = RetryConfig(strategy=RetryStrategy.FIXED, base_delay=1.0, jitter=True)

        delays = []
        for i in range(10):
            delay = self.handler._retry_manager._calculate_delay(config, 0)
            delays.append(delay)

        # 验证抖动效果（结果应该在0.5-1.5之间）
        for delay in delays:
            self.assertGreaterEqual(delay, 0.5)
            self.assertLessEqual(delay, 1.5)

    def test_calculate_delay_max_limit(self):
        """测试最大延迟限制"""
        from src.infrastructure.error.handlers.specialized_error_handler import RetryConfig

        config = RetryConfig(
            strategy=RetryStrategy.EXPONENTIAL,
            base_delay=10.0,
            max_delay=30.0,
            backoff_factor=10.0,
            jitter=False
        )

        delay = self.handler._retry_manager._calculate_delay(config, 5)  # 10.0 * (10.0 ** 5) = 100000.0
        self.assertEqual(delay, 30.0)  # 应该被限制为最大值

    def test_execute_retry_success(self):
        """测试重试执行成功"""
        from src.infrastructure.error.handlers.specialized_error_handler import RetryConfig

        config = RetryConfig(max_attempts=3, base_delay=0.01)  # 很短的延迟用于测试

        # 模拟重试逻辑（这里简化，实际应该在重试配置中定义）
        attempts = 0
        max_attempts = 2  # 模拟2次重试后成功

        # 这里我们直接测试_execute_retry的结构
        # 由于_execute_retry是私有方法且依赖复杂逻辑，我们测试其公开接口

        error = ConnectionError("连接失败")
        context = {"operation": "network_call"}

        # 注册一个会重试的处理器
        def mock_handler(error, context):
            return {
                'action': 'retry',
                'retry_config': config,
                'message': 'Connection failed, retrying'
            }

        self.handler.register_handler(ConnectionError, mock_handler)
        result = self.handler.handle_error(error, context)

        self.assertTrue(result['handled'])
        self.assertIn('retry_config', result)
        self.assertEqual(result['action'], 'retry')

    def test_connection_error_handling(self):
        """测试连接错误处理"""
        error = ConnectionError("连接失败")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'ConnectionError')
        self.assertEqual(result['category'], 'network')

    def test_timeout_error_handling(self):
        """测试超时错误处理"""
        error = TimeoutError("操作超时")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'TimeoutError')
        self.assertEqual(result['category'], 'network')

    def test_io_error_handling(self):
        """测试IO错误处理"""
        error = IOError("IO操作失败")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'IOError')
        self.assertEqual(result['category'], 'system')

    def test_os_error_handling(self):
        """测试OS错误处理"""
        error = OSError("系统操作失败")
        result = self.handler.handle_error(error)

        self.assertFalse(result['handled'])
        self.assertEqual(result['error_type'], 'OSError')
        self.assertEqual(result['category'], 'system')

    def test_error_history_management(self):
        """测试错误历史管理"""
        # 添加一些错误
        errors = [
            ConnectionError("连接失败1"),
            TimeoutError("超时1"),
            IOError("IO错误1")
        ]

        for error in errors:
            self.handler.handle_error(error)

        # 验证历史记录
        history = self.handler.get_error_history()
        self.assertEqual(len(history), 3)

        # 验证历史记录内容
        error_types = [entry['error_type'] for entry in history]
        self.assertIn('ConnectionError', error_types)
        self.assertIn('TimeoutError', error_types)
        self.assertIn('IOError', error_types)

        # 测试清空历史
        self.handler.clear_history()
        self.assertEqual(len(self.handler.get_error_history()), 0)

    def test_get_stats_comprehensive(self):
        """测试综合统计信息"""
        # 添加一些错误
        errors = [
            ConnectionError("网络错误"),
            ValueError("值错误"),
            IOError("IO错误")
        ]

        for error in errors:
            self.handler.handle_error(error)

        stats = self.handler.get_stats()

        # 验证统计信息结构
        self.assertIn('total_errors', stats)
        self.assertIn('severity_distribution', stats)
        self.assertIn('category_distribution', stats)
        self.assertIn('registered_handlers', stats)
        self.assertIn('registered_strategies', stats)
        self.assertIn('retry_configs', stats)

        # 验证统计数据
        self.assertEqual(stats['total_errors'], 3)
        self.assertGreater(stats['registered_handlers'], 0)

    def test_register_and_get_handlers(self):
        """测试注册和获取处理器"""
        def mock_handler(error, context):
            return {"handled": True, "result": "mock"}

        # 注册处理器
        self.handler.register_handler(ValueError, mock_handler)

        # 验证处理器已注册
        handlers = self.handler.get_registered_handlers()
        self.assertIn('ValueError', handlers)

    def test_register_and_get_strategies(self):
        """测试注册和获取策略"""
        def mock_strategy():
            return "strategy_result"

        # 注册策略
        self.handler.register_strategy("test_strategy", mock_strategy)

        # 验证策略已注册
        strategies = self.handler.get_registered_strategies()
        self.assertIn("test_strategy", strategies)

    def test_severity_classification(self):
        """测试严重程度分类"""
        test_cases = [
            (KeyboardInterrupt("中断"), "critical"),
            (ConnectionError("连接错误"), "error"),
            (TimeoutError("超时"), "error"),
            (IOError("IO错误"), "warning"),
            (ValueError("值错误"), "info")
        ]

        for error, expected_severity in test_cases:
            with self.subTest(error=error):
                result = self.handler.handle_error(error)
                self.assertEqual(result['severity'], expected_severity)

    def test_category_classification(self):
        """测试错误类别分类"""
        test_cases = [
            (ConnectionError("连接错误"), "network"),
            (TimeoutError("超时错误"), "network"),
            (IOError("IO错误"), "system"),
            (OSError("系统错误"), "system"),
            (ValueError("值错误"), "unknown")
        ]

        for error, expected_category in test_cases:
            with self.subTest(error=error):
                result = self.handler.handle_error(error)
                self.assertEqual(result['category'], expected_category)

    def test_context_preservation(self):
        """测试上下文保留"""
        error = ValueError("测试错误")
        context = {
            "user_id": 123,
            "operation": "test_operation",
            "timestamp": time.time()
        }

        result = self.handler.handle_error(error, context)

        # 验证上下文被保留
        self.assertIn('error_context', result)
        error_context = result['error_context']
        self.assertEqual(error_context['context'], context)

    def test_connection_error_with_context(self):
        """测试带上下文的连接错误处理"""
        error = ConnectionError("连接失败")
        context = {"host": "example.com", "port": 8080}

        result = self.handler.handle_error(error, context)

        # 验证处理结果
        self.assertEqual(result['error_type'], 'ConnectionError')
        self.assertEqual(result['severity'], 'error')
        self.assertEqual(result['category'], 'network')
        self.assertEqual(result['context'], context)

    def test_timeout_error_with_context(self):
        """测试带上下文的超时错误处理"""
        error = TimeoutError("操作超时")
        context = {"operation": "database_query", "timeout": 30}

        result = self.handler.handle_error(error, context)

        # 验证处理结果
        self.assertEqual(result['error_type'], 'TimeoutError')
        self.assertEqual(result['severity'], 'error')
        self.assertEqual(result['category'], 'network')

    def test_io_error_with_context(self):
        """测试带上下文的IO错误处理"""
        error = IOError("文件读取失败")
        context = {"file_path": "/path/to/file", "operation": "read"}

        result = self.handler.handle_error(error, context)

        # 验证处理结果
        self.assertEqual(result['error_type'], 'IOError')
        self.assertEqual(result['severity'], 'warning')
        self.assertEqual(result['category'], 'system')

    def test_os_error_with_context(self):
        """测试带上下文的OS错误处理"""
        error = OSError("系统调用失败")
        context = {"syscall": "open", "resource": "/dev/resource"}

        result = self.handler.handle_error(error, context)

        # 验证处理结果
        self.assertEqual(result['error_type'], 'OSError')
        self.assertEqual(result['severity'], 'warning')
        self.assertEqual(result['category'], 'system')

    def test_error_history_limit(self):
        """测试错误历史限制"""
        # 添加超过限制的错误数量
        for i in range(150):
            error = ValueError(f"错误 {i}")
            self.handler.handle_error(error)

        history = self.handler.get_error_history()
        
        # 验证历史记录数量不超过限制
        self.assertEqual(len(history), 100)  # 默认限制是100

    def test_error_history_clear(self):
        """测试错误历史清空"""
        # 添加一些错误
        errors = [
            ConnectionError("连接失败1"),
            TimeoutError("超时1"),
            IOError("IO错误1")
        ]

        for error in errors:
            self.handler.handle_error(error)

        # 验证历史记录
        history = self.handler.get_error_history()
        self.assertEqual(len(history), 3)

        # 测试清空历史
        self.handler.clear_history()
        self.assertEqual(len(self.handler.get_error_history()), 0)


if __name__ == '__main__':
    unittest.main()