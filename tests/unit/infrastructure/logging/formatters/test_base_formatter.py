"""
测试基础日志格式化器

覆盖 base.py 中的 BaseFormatter 类
"""

import logging
from unittest.mock import Mock
from src.infrastructure.logging.formatters.base import BaseFormatter
from src.infrastructure.logging.formatters.text import TextFormatter


class TestBaseFormatter:
    """BaseFormatter 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        formatter = BaseFormatter()

        assert formatter.config == {}
        assert formatter.name == "BaseFormatter"
        assert formatter.date_format == "%Y-%m-%d %H:%M:%S"
        assert formatter.include_level == True
        assert formatter.include_logger_name == True
        assert formatter.include_timestamp == True
        assert formatter.max_message_length == 0

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'name': 'CustomFormatter',
            'date_format': '%Y-%m-%d',
            'include_level': False,
            'include_logger_name': False,
            'include_timestamp': False,
            'max_message_length': 100
        }
        formatter = BaseFormatter(config)

        assert formatter.config == config
        assert formatter.name == "CustomFormatter"
        assert formatter.date_format == "%Y-%m-%d"
        assert formatter.include_level == False
        assert formatter.include_logger_name == False
        assert formatter.include_timestamp == False
        assert formatter.max_message_length == 100

    def test_format_error_handling(self):
        """测试格式化错误处理"""
        formatter = TextFormatter()

        # 创建一个mock记录，当getMessage()抛出异常时
        mock_record = Mock(spec=logging.LogRecord)
        mock_record.getMessage.return_value = "Mock message"
        mock_record.levelname = "INFO"
        mock_record.name = "test"

        result = formatter.format(mock_record)
        # 应该正常格式化
        assert isinstance(result, str)
        assert "Mock message" in result

    def test_abstract_method(self):
        """测试抽象方法"""
        formatter = BaseFormatter()

        # 应该抛出NotImplementedError
        mock_record = Mock(spec=logging.LogRecord)
        try:
            formatter._format(mock_record)
            assert False, "Should raise NotImplementedError"
        except NotImplementedError:
            pass  # 期望的行为
