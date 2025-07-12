import unittest
from unittest.mock import patch, MagicMock
from src.infrastructure.error import ErrorHandler, ErrorLevel

class TestErrorHandler(unittest.TestCase):
    """错误处理框架单元测试"""

    def setUp(self):
        self.handler = ErrorHandler()
        self.mock_logger = MagicMock()
        self.handler.logger = self.mock_logger

    def test_error_classification(self):
        """测试错误分类与日志级别"""
        # 业务错误 - 应记录为WARNING
        class BusinessError(Exception): pass
        self.handler.handle(BusinessError(), log_level=ErrorLevel.WARNING)
        self.mock_logger.warning.assert_called()

        # 系统错误 - 应记录为ERROR
        class SystemError(Exception): pass
        self.handler.handle(SystemError())
        self.mock_logger.error.assert_called()

        # 关键故障 - 应记录为CRITICAL
        class CriticalError(Exception): pass
        self.handler.handle(CriticalError(), log_level=ErrorLevel.CRITICAL)
        self.mock_logger.critical.assert_called()

    def test_custom_handlers(self):
        """测试自定义错误处理器"""
        mock_handler = MagicMock()
        self.handler.add_handler(mock_handler)

        test_error = ValueError("Test error")
        self.handler.handle(test_error)

        mock_handler.assert_called_once_with(test_error, {})

    def test_error_context(self):
        """测试错误上下文传递"""
        test_context = {
            'module': 'trade_engine',
            'operation': 'order_execution',
            'order_id': 12345
        }

        self.handler.handle(Exception(), context=test_context)

        # 验证日志记录包含上下文
        args, kwargs = self.mock_logger.error.call_args
        self.assertIn('module', kwargs.get('extra', {}))
        self.assertEqual(kwargs['extra']['module'], 'trade_engine')

    @patch('src.infrastructure.error.AlertManager.send_alert')
    def test_critical_alert(self, mock_send_alert):
        """测试关键错误告警触发"""
        self.handler.handle(Exception(), log_level=ErrorLevel.CRITICAL)
        mock_send_alert.assert_called()

    def test_log_filters(self):
        """测试日志过滤器"""
        # 添加过滤器 - 只记录包含特定标记的错误
        self.handler.add_log_filter(
            lambda ctx: ctx.get('important', False)
        )

        # 不重要的错误 - 不应记录
        self.handler.handle(Exception())
        self.mock_logger.error.assert_not_called()

        # 重要的错误 - 应记录
        self.handler.handle(Exception(), extra_log_data={'important': True})
        self.mock_logger.error.assert_called()

    def test_retry_mechanism(self):
        """测试自动重试机制"""
        mock_operation = MagicMock()
        mock_operation.side_effect = [Exception(), True]

        result = self.handler.with_retry(
            mock_operation,
            max_retries=3,
            retry_delay=0.1
        )

        self.assertTrue(result)
        self.assertEqual(mock_operation.call_count, 2)

if __name__ == '__main__':
    unittest.main()
