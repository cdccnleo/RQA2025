"""
测试文本日志格式化器

覆盖 text.py 中的 TextFormatter 类
"""

import logging
from unittest.mock import Mock
from src.infrastructure.logging.formatters.text import TextFormatter


class TestTextFormatter:
    """TextFormatter 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        formatter = TextFormatter()

        assert formatter.template == "{timestamp} {level} {logger}: {message}"
        assert formatter.name == "TextFormatter"

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {'template': '{level}: {message}'}
        formatter = TextFormatter(config)

        assert formatter.template == "{level}: {message}"

    def test_format_full_record(self):
        """测试完整记录格式化"""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        result = formatter._format(record)

        # 检查包含所有部分
        assert "INFO" in result
        assert "test_logger" in result
        assert "Test message" in result
        assert ":" in result  # 分隔符

    def test_format_minimal_record(self):
        """测试最小记录格式化"""
        config = {
            'include_timestamp': False,
            'include_level': False,
            'include_logger_name': False,
            'template': '{message}'
        }
        formatter = TextFormatter(config)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )

        result = formatter._format(record)

        # 应该只包含消息
        assert result.strip() == "Test message"

    def test_format_with_custom_template(self):
        """测试自定义模板格式化"""
        config = {'template': '[{level}] {message}'}
        formatter = TextFormatter(config)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=10,
            msg="Error occurred",
            args=(),
            exc_info=None
        )

        result = formatter._format(record)

        assert result == "[ERROR] Error occurred"

    def test_format_timestamp(self):
        """测试时间戳格式化"""
        formatter = TextFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.created = 1609459200  # 2021-01-01 00:00:00 UTC

        timestamp = formatter._format_timestamp(record)
        assert isinstance(timestamp, str)
        assert len(timestamp) > 0

    def test_format_level(self):
        """测试级别格式化"""
        formatter = TextFormatter()

        # 测试不同级别
        record_info = logging.LogRecord("test", logging.INFO, "", 0, "", (), None)
        record_error = logging.LogRecord("test", logging.ERROR, "", 0, "", (), None)
        record_debug = logging.LogRecord("test", logging.DEBUG, "", 0, "", (), None)

        assert formatter._format_level(record_info) == "INFO"
        assert formatter._format_level(record_error) == "ERROR"
        assert formatter._format_level(record_debug) == "DEBUG"

    def test_format_logger_name(self):
        """测试日志器名称格式化"""
        formatter = TextFormatter()
        record = logging.LogRecord("my.test.logger", logging.INFO, "", 0, "", (), None)

        assert formatter._format_logger_name(record) == "my.test.logger"

    def test_format_message_with_args(self):
        """测试带参数的消息格式化"""
        formatter = TextFormatter()
        record = logging.LogRecord("test", logging.INFO, "", 0, "User %s logged in", ("john",), None)

        # 测试record.getMessage()能正确格式化带参数的消息
        message = record.getMessage()
        assert "User john logged in" == message

        # 测试formatter能处理带参数的消息
        result = formatter._format(record)
        assert "User john logged in" in result
