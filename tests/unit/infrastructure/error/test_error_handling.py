import pytest
import logging
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from src.infrastructure.error.error_handler import ErrorHandler, ErrorRecord
from src.infrastructure.error.retry_handler import RetryHandler

# 测试用异常类
class TestException(Exception):
    pass

class RetryableException(Exception):
    pass

class NonRetryableException(Exception):
    pass

# Fixtures
@pytest.fixture
def error_handler():
    """基础错误处理器fixture"""
    return ErrorHandler()

@pytest.fixture
def retry_handler():
    """基础重试处理器fixture"""
    return RetryHandler(
        max_attempts=3,
        initial_delay=0.1,
        retry_exceptions=[RetryableException]
    )

# 测试用例
class TestErrorHandler:
    """测试错误处理器功能"""

    def test_basic_error_handling(self, error_handler):
        """测试基础错误处理"""

        # 测试处理器返回非None值
        def mock_handler_func(exception, context=None):
            return "handled"

        mock_handler = MagicMock(side_effect=mock_handler_func)
        error_handler.register_handler(TestException, mock_handler)

        exception = TestException("test error")
        result = error_handler.handle(exception)

        assert result == "handled"
        mock_handler.assert_called_once()

        # 验证错误记录（总记录数应为1）
        if hasattr(error_handler, 'get_records'):
            records = error_handler.get_records()
            assert len(records) == 1  # 第一次调用创建1条记录

        # 测试处理器返回None（注册新处理器替换旧的）
        def none_handler(exception, context=None):
            return None

        error_handler.register_handler(TestException, none_handler)
        new_exception = TestException("new error")  # 使用新异常对象
        result = error_handler.handle(new_exception)  # 处理新异常
        assert result is None

        # 验证记录总数（2条记录）
        records = error_handler.get_records()
        assert len(records) == 2

        # 验证未处理记录（只有第二次调用是未处理的）
        unhandled_records = error_handler.get_records(handled=False)
        assert len(unhandled_records) == 1  # 只有第二次调用

        # 测试处理器抛出异常
        def failing_handler(exception, context=None):
            raise RuntimeError("handler error")

        error_handler.register_handler(TestException, failing_handler)
        another_exception = TestException("third error")  # 第三个异常
        result = error_handler.handle(another_exception)
        assert result is None

        # 总记录数应为3
        records = error_handler.get_records()
        assert len(records) == 3

        # 未处理记录应有2条（第二次和第三次）
        unhandled_records = error_handler.get_records(handled=False)
        assert len(unhandled_records) == 2

    def test_unhandled_exception(self, error_handler):
        """测试未处理异常"""
        exception = TestException("test error")
        result = error_handler.handle(exception)

        assert result is None
        records = error_handler.get_records(handled=False)
        assert len(records) == 1

    def test_error_records_filtering(self, error_handler):
        """测试错误记录过滤"""
        # 注册两个处理器
        error_handler.register_handler(TestException, lambda e, context=None: "handled")
        error_handler.register_handler(Exception, lambda e, context=None: None)

        # 生成不同时间的错误
        past_time = datetime.now() - timedelta(minutes=5)

        with patch('src.infrastructure.error.error_handler.time') as mock_time:
            mock_time.time.return_value = past_time.timestamp()
            error_handler.handle(Exception("old error"))

        error_handler.handle(TestException("new error"))

        # 测试时间过滤 - 放宽时间范围到10分钟以确保包含所有记录
        start_time = (datetime.now() - timedelta(minutes=10)).timestamp()
        records = error_handler.get_records(start_time=start_time)
        assert len(records) == 2  # 应该包含两条记录
        
        # 测试处理状态过滤
        assert len(error_handler.get_records(handled=True)) == 1
        assert len(error_handler.get_records(handled=False)) == 1

    def test_alert_hooks(self, error_handler):
        """测试告警钩子"""
        hook_mock = MagicMock()
        error_handler.add_alert_hook(hook_mock)

        exception = TestException("hook test")
        error_handler.handle(exception)

        hook_mock.assert_called_once()
        record = hook_mock.call_args[0][0]
        assert isinstance(record, ErrorRecord)
        assert record.exception == exception

class TestRetryHandler:
    """测试重试处理器功能"""
    
    def test_successful_retry(self, retry_handler):
        """测试成功重试"""
        mock_func = MagicMock()
        mock_func.__name__ = "mock_func"
        mock_func.__qualname__ = "mock_func"
        mock_func.__module__ = "test_module"
        mock_func.side_effect = [RetryableException("retry"), "success"]

        decorated = retry_handler(mock_func)
        result = decorated()

        assert result == "success"
        assert mock_func.call_count == 2

        stats = retry_handler.get_retry_stats("mock_func")
        assert stats['total_attempts'] == 2
        assert stats['successful'] is True

    @pytest.mark.parametrize("exception,should_retry", [
        (RetryableException("test"), True),
        (NonRetryableException("test"), False)
    ])
    def test_retry_condition(self, retry_handler, exception, should_retry):
        """测试重试条件判断"""
        mock_func = MagicMock()
        mock_func.__name__ = "mock_func"
        mock_func.__qualname__ = "mock_func"
        mock_func.__module__ = "test_module"
        mock_func.side_effect = exception
        
        decorated = retry_handler(mock_func)

        if should_retry:
            with pytest.raises(Exception) as excinfo:
                decorated()
            # 应抛出RetryError，且__cause__为原始异常类型
            from src.infrastructure.error.exceptions import RetryError
            assert isinstance(excinfo.value, RetryError)
            assert isinstance(excinfo.value.__cause__, type(exception))
            assert mock_func.call_count == retry_handler.max_attempts  # 初始尝试+重试
        else:
            with pytest.raises(type(exception)):
                decorated()
            assert mock_func.call_count == 1

    def test_backoff_calculation(self, retry_handler):
        """测试退避时间计算"""
        # 测试抖动范围
        delays = set()
        for _ in range(100):
            delay = retry_handler._calculate_delay(attempt=2)
            delays.add(delay)

        base_delay = retry_handler.initial_delay * retry_handler.backoff_factor
        min_delay = base_delay * (1 - retry_handler.jitter)
        max_delay = base_delay * (1 + retry_handler.jitter)

        assert all(min_delay <= d <= max_delay for d in delays)
        assert len(delays) > 1  # 确保有随机性

@pytest.mark.integration
class TestIntegration:
    def test_handler_with_retry(self, error_handler, retry_handler):
        """测试错误处理器与重试的集成"""
        # 注册可重试异常处理器
        error_handler.register_handler(
            RetryableException,
            lambda e, context=None: f"handled: {str(e)}",
            retryable=True
        )

        # 创建测试函数
        call_count = 0
        def flaky_function():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise RetryableException(f"attempt {call_count}")
            return "success"

        # 应用重试装饰器
        decorated = retry_handler(flaky_function)

        # 测试处理
        result = error_handler.handle(RetryableException("direct"))
        assert "handled" in result

        # 测试重试流程
        assert decorated() == "success"
        assert call_count == 3
