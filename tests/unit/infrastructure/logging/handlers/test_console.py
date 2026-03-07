"""
测试控制台日志处理器

覆盖 console.py 中的 ConsoleHandler 类
"""

import logging
import sys
import builtins
from io import StringIO
from unittest.mock import patch
from src.infrastructure.logging.handlers.console import ConsoleHandler


class TestConsoleHandler:
    """ConsoleHandler 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        handler = ConsoleHandler()

        assert handler.stream == sys.stdout
        assert handler.colorize == False
        assert handler._formatter is None

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'stream': sys.stderr,
            'colorize': True
        }
        handler = ConsoleHandler(config)

        assert handler.stream == sys.stderr
        assert handler.colorize == True

    def test_emit_basic(self):
        """测试基本发出"""
        # 使用StringIO作为stream来捕获输出
        mock_stdout = StringIO()
        config = {'stream': mock_stdout}
        handler = ConsoleHandler(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test message",
            args=(),
            exc_info=None
        )

        handler._emit(record)

        output = mock_stdout.getvalue()
        assert "INFO: Test message" in output

    def test_emit_with_custom_stream(self):
        """测试自定义流发出"""
        custom_stream = StringIO()
        config = {'stream': custom_stream}
        handler = ConsoleHandler(config)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None
        )

        handler._emit(record)

        output = custom_stream.getvalue()
        assert "ERROR: Error message" in output

    def test_emit_with_formatter(self):
        """测试带格式化器的发出"""
        mock_stdout = StringIO()
        config = {'stream': mock_stdout}
        handler = ConsoleHandler(config)
        formatter = logging.Formatter('%(levelname)s - %(name)s - %(message)s')
        handler.set_formatter(formatter)

        record = logging.LogRecord(
            name="my_logger",
            level=logging.WARNING,
            pathname="",
            lineno=0,
            msg="Warning message",
            args=(),
            exc_info=None
        )

        handler._emit(record)

        output = mock_stdout.getvalue()
        assert "WARNING - my_logger - Warning message" in output

    def test_emit_with_colorize(self):
        """测试带颜色化的发出"""
        mock_stdout = StringIO()
        config = {'colorize': True, 'stream': mock_stdout}
        handler = ConsoleHandler(config)

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Error message",
            args=(),
            exc_info=None
        )

        handler._emit(record)

        output = mock_stdout.getvalue()
        # 应该包含ANSI颜色代码
        assert '\033[31m' in output  # 红色
        assert '\033[0m' in output   # 重置
        assert "ERROR: Error message" in output

    def test_colorize_different_levels(self):
        """测试不同级别的颜色化"""
        handler = ConsoleHandler()

        # 测试不同级别
        test_cases = [
            (logging.DEBUG, '\033[36m'),    # 青色
            (logging.INFO, '\033[32m'),     # 绿色
            (logging.WARNING, '\033[33m'),  # 黄色
            (logging.ERROR, '\033[31m'),    # 红色
            (logging.CRITICAL, '\033[35m'), # 紫色
        ]

        for level, expected_color in test_cases:
            message = handler._colorize_message("Test", level)
            assert expected_color in message
            assert '\033[0m' in message

    def test_colorize_unknown_level(self):
        """测试未知级别的颜色化"""
        handler = ConsoleHandler()

        message = handler._colorize_message("Test", 999)
        # 未知级别不应该添加颜色
        assert '\033[' not in message

    def test_emit_error_handling(self):
        """测试发出错误处理"""
        handler = ConsoleHandler()

        # 模拟print失败
        record = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)

        # 只mock到stream的print失败，让stderr的print正常工作
        def mock_print(*args, **kwargs):
            if kwargs.get('file') == handler.stream:
                raise Exception("Print failed")
            # 对于stderr的print，正常执行
            return original_print(*args, **kwargs)

        original_print = builtins.print
        mock_stderr = StringIO()
        with patch('builtins.print', side_effect=mock_print):
            with patch('sys.stderr', mock_stderr):
                handler._emit(record)

                error_output = mock_stderr.getvalue()
                assert "[ERROR] Failed to write to console" in error_output

    def test_set_formatter(self):
        """测试设置格式化器"""
        handler = ConsoleHandler()
        formatter = logging.Formatter()

        handler.set_formatter(formatter)
        assert handler._formatter == formatter

    def test_close(self):
        """测试关闭处理器"""
        handler = ConsoleHandler()

        # 控制台处理器关闭应该是静默的
        handler._close()

    def test_get_status(self):
        """测试获取状态"""
        config = {'stream': sys.stderr, 'colorize': True}
        handler = ConsoleHandler(config)
        formatter = logging.Formatter()
        handler.set_formatter(formatter)

        status = handler.get_status()

        assert status['stream'] == 'stderr'
        assert status['colorize'] == True
        assert status['has_formatter'] == True
        assert status['name'] == 'ConsoleHandler'
        assert status['enabled'] == True
        assert status['closed'] == False

    def test_get_status_stdout(self):
        """测试获取stdout状态"""
        handler = ConsoleHandler()  # 默认是stdout

        status = handler.get_status()
        assert status['stream'] == 'stdout'

    def test_format_record_default(self):
        """测试默认记录格式化"""
        handler = ConsoleHandler()

        record = logging.LogRecord(
            name="test",
            level=logging.DEBUG,
            pathname="",
            lineno=0,
            msg="Debug message",
            args=(),
            exc_info=None
        )

        result = handler._format_record(record)
        assert result == "DEBUG: Debug message"

    def test_format_record_with_formatter(self):
        """测试带格式化器的记录格式化"""
        handler = ConsoleHandler()
        formatter = logging.Formatter('[%(levelname)s] %(message)s')
        handler.set_formatter(formatter)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Info message",
            args=(),
            exc_info=None
        )

        result = handler._format_record(record)
        assert result == "[INFO] Info message"
