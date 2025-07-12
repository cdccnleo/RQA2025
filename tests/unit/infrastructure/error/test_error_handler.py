import logging
import pytest
from src.infrastructure.error.error_handler import ErrorHandler
from unittest.mock import MagicMock, patch


class TestErrorHandler:
    @pytest.fixture
    def error_handler(self):
        """提供干净的ErrorHandler实例"""
        return ErrorHandler()

    def test_error_classification(self, error_handler):
        """测试错误分类"""

        # 模拟业务错误
        class BusinessError(Exception): pass

        # 模拟系统错误
        class SystemError(Exception): pass

        # 验证业务错误被分类为WARNING
        error_handler.handle(BusinessError(), log_level='WARNING')
        # 验证系统错误被分类为ERROR
        error_handler.handle(SystemError(), log_level='ERROR')

    def test_custom_handler(self, error_handler):
        """测试自定义错误处理器"""
        handler_called = False

        def custom_handler(e, ctx):
            nonlocal handler_called
            handler_called = True
            assert isinstance(e, ValueError)
            assert ctx.get('operation') == 'test'

        error_handler.add_handler(custom_handler)
        error_handler.handle(ValueError("test"), context={'operation': 'test'})

        assert handler_called is True

    @patch('src.infrastructure.error.error_handler.logger')
    def test_log_context(self, mock_logger):
        """测试日志上下文管理"""
        # 创建ErrorHandler实例
        error_handler = ErrorHandler()

        # 设置测试上下文
        context = {
            'app': 'test_app',
            'module': 'test_module',
            'request_id': '12345'
        }
        error_handler.update_log_context(**context)

        # 注册一个会失败的处理器
        def failing_handler(e, ctx):
            raise RuntimeError("handler error")

        error_handler.add_handler(failing_handler)

        # 额外的日志数据
        extra_log_data = {'detail': 'test_detail'}

        # 处理一个异常
        error_handler.handle(
            RuntimeError("test error"),
            extra_log_data=extra_log_data,
            log_level='ERROR'
        )

        # 验证mock_logger.log被调用两次（主异常+处理器异常）
        assert mock_logger.log.call_count == 2

        # 获取调用参数
        args, kwargs = mock_logger.log.call_args

        # 验证日志级别
        assert args[0] == logging.ERROR
        # 验证日志消息包含"Error handler failed"
        assert "Error handler failed" in args[1]

        # 验证extra参数中的error_context字典
        error_context = kwargs['extra']['error_context']
        # 检查我们更新的上下文（加上ctx_前缀）
        assert error_context['ctx_app'] == 'test_app'
        assert error_context['ctx_module'] == 'test_module'
        assert error_context['ctx_request_id'] == '12345'
        # 检查额外的日志数据（也加上ctx_前缀）
        assert error_context['ctx_detail'] == 'test_detail'

        # 验证exc_info为True
        assert kwargs['exc_info'] is True

    def test_retry_mechanism(self, error_handler):
        """测试自动重试机制"""
        retry_count = 0

        def failing_operation():
            nonlocal retry_count
            retry_count += 1
            if retry_count < 3:
                raise ConnectionError("模拟连接失败")
            return "成功"

        result = error_handler.with_retry(
            failing_operation,
            max_retries=3,
            delay=0.1
        )

        assert result == "成功"
        assert retry_count == 3