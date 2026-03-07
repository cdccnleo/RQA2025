#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 日志格式化器

测试各种日志格式化器的功能，包括JSON、结构化文本、纯文本格式化。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import logging
import sys
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.infrastructure.logging.formatters import JSONFormatter, StructuredFormatter, TextFormatter
from src.infrastructure.logging.core import LogLevel


class TestJSONFormatter:
    """JSON格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = JSONFormatter()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None
        assert self.formatter.pretty_print is False
        assert self.formatter.include_extra is True
        assert self.formatter.include_exc_info is True
        assert isinstance(self.formatter.custom_fields, dict)

    def test_initialization_with_config(self):
        """测试带配置的初始化"""
        config = {
            'pretty_print': True,
            'include_extra': False,
            'include_exc_info': False,
            'custom_fields': {'service': 'test'}
        }
        formatter = JSONFormatter(config)
        assert formatter.pretty_print is True
        assert formatter.include_extra is False
        assert formatter.include_exc_info is False
        assert formatter.custom_fields['service'] == 'test'

    def test_format_basic_record(self):
        """测试格式化基本日志记录"""
        # 创建测试日志记录
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="测试消息",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0  # 2022-01-01 00:00:00

        result = self.formatter.format(record)

        # 解析JSON结果
        data = json.loads(result)

        # 验证基本字段
        assert '2022-01-01' in data['timestamp']
        assert data['level'] == 'INFO'
        assert data['logger'] == 'test.logger'
        assert data['message'] == '测试消息'

    def test_format_with_extra_fields(self):
        """测试包含额外字段的格式化"""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="警告消息: %s",
            args=("参数",),
            exc_info=None
        )
        record.created = datetime.now().timestamp()

        # 添加额外字段
        record.custom_field = "custom_value"
        record.user_id = 12345

        result = self.formatter.format(record)
        data = json.loads(result)

        # 验证额外字段
        assert 'extra_custom_field' in data
        assert data['extra_custom_field'] == 'custom_value'
        assert 'extra_user_id' in data
        assert data['extra_user_id'] == 12345

    def test_format_with_exception(self):
        """测试包含异常信息的格式化"""
        try:
            raise ValueError("测试异常")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="发生异常",
            args=(),
            exc_info=exc_info
        )
        record.created = datetime.now().timestamp()

        result = self.formatter.format(record)
        data = json.loads(result)

        # 验证异常信息
        assert 'exception' in data
        assert data['exception']['type'] == 'ValueError'
        assert '测试异常' in data['exception']['message']

    def test_pretty_print(self):
        """测试美化打印"""
        config = {'pretty_print': True}
        formatter = JSONFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="测试",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # 美化打印应该有多行
        assert '\n' in result
        assert '  ' in result  # 缩进

        # 验证仍然是有效JSON
        data = json.loads(result)
        assert data['message'] == '测试'

    def test_custom_fields(self):
        """测试自定义字段"""
        config = {
            'custom_fields': {
                'service': 'test_service',
                'version': '1.0.0',
                'environment': 'testing'
            }
        }
        formatter = JSONFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="测试",
            args=(),
            exc_info=None
        )

        result = formatter.format(record)
        data = json.loads(result)

        # 验证自定义字段
        assert data['service'] == 'test_service'
        assert data['version'] == '1.0.0'
        assert data['environment'] == 'testing'

    def test_format_error_handling(self):
        """测试格式化错误处理"""
        # 创建有问题的记录
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="测试",
            args=(),
            exc_info=None
        )

        # 模拟格式化过程中的异常
        with patch.object(self.formatter, '_build_base_log_data', side_effect=Exception("测试异常")):
            result = self.formatter.format(record)
            # 应该返回错误格式
            assert "[FORMAT_ERROR]" in result

    def test_set_format_method(self):
        """测试设置格式类型方法"""
        from src.infrastructure.logging.core import LogFormat

        # 不应该抛出异常
        self.formatter.set_format(LogFormat.JSON)


class TestStructuredFormatter:
    """结构化文本格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = StructuredFormatter()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None

    def test_format_basic_record(self):
        """测试格式化基本日志记录"""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/test.py",
            lineno=42,
            msg="测试消息",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0

        result = self.formatter.format(record)

        # 验证结构化格式
        assert '2022-01-01' in result
        assert 'INFO' in result
        assert 'test.logger' in result
        assert '测试消息' in result

    def test_format_with_extra_fields(self):
        """测试包含额外字段的格式化"""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=10,
            msg="警告消息",
            args=(),
            exc_info=None
        )
        record.created = datetime.now().timestamp()
        record.user_id = 12345
        record.action = "login"

        result = self.formatter.format(record)

        # 验证额外字段
        assert 'user_id=12345' in result
        assert 'action=login' in result

    def test_set_format_method(self):
        """测试设置格式类型方法"""
        from src.infrastructure.logging.core import LogFormat

        # 不应该抛出异常
        self.formatter.set_format(LogFormat.STRUCTURED)


class TestTextFormatter:
    """纯文本格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = TextFormatter()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None

    def test_format_basic_record(self):
        """测试格式化基本日志记录"""
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="测试消息",
            args=(),
            exc_info=None
        )
        record.created = 1640995200.0

        result = self.formatter.format(record)

        # 验证文本格式
        assert '2022-01-01' in result
        assert 'INFO' in result
        assert 'test.logger' in result
        assert '测试消息' in result

    def test_format_with_args(self):
        """测试包含参数的消息格式化"""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="用户 %s 执行操作 %s",
            args=("alice", "login"),
            exc_info=None
        )

        result = self.formatter.format(record)

        # 验证参数替换
        assert '用户 alice 执行操作 login' in result

    def test_set_format_method(self):
        """测试设置格式类型方法"""
        from src.infrastructure.logging.core import LogFormat

        # 不应该抛出异常
        self.formatter.set_format(LogFormat.TEXT)

    def test_custom_template_formatting(self):
        """测试自定义模板格式化"""
        custom_template = '{level} - {message}'
        self.formatter.set_template(custom_template)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=1,
            msg="Custom template test",
            args=(),
            exc_info=None
        )
        
        result = self.formatter.format(record)
        assert 'WARNING - Custom template test' in result

    def test_template_with_different_combinations(self):
        """测试不同字段组合的模板格式化"""
        # 测试只包含message字段的模板
        simple_template = '{message}'
        self.formatter.set_template(simple_template)
        
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Simple template test",
            args=(),
            exc_info=None
        )
        
        result = self.formatter.format(record)
        assert result.strip() == 'Simple template test'

    def test_get_config_method(self):
        """测试get_config方法"""
        config = self.formatter.get_config()
        
        assert isinstance(config, dict)
        assert 'template' in config
        assert isinstance(config['template'], str)

    def test_formatter_with_different_config_options(self):
        """测试不同配置选项的格式化器"""
        config = {
            'template': '{logger}: {message}',
            'include_timestamp': False,
            'include_level': True,
            'include_logger_name': True
        }
        
        formatter = TextFormatter(config)
        
        record = logging.LogRecord(
            name="config_test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=1,
            msg="Config test message",
            args=(),
            exc_info=None
        )
        
        result = formatter.format(record)
        assert 'config_test: Config test message' in result

    def test_set_template_method(self):
        """测试set_template方法"""
        original_template = self.formatter.template
        
        new_template = '{message} [{level}]'
        self.formatter.set_template(new_template)
        
        assert self.formatter.template == new_template
        assert self.formatter.template != original_template


class TestFormatterIntegration:
    """格式化器集成测试"""

    def test_all_formatters_implement_interface(self):
        """测试所有格式化器都正确实现了接口"""
        from src.infrastructure.logging.core.interfaces import ILogFormatter

        formatters = [JSONFormatter(), StructuredFormatter(), TextFormatter()]

        for formatter in formatters:
            assert isinstance(formatter, ILogFormatter)
            assert hasattr(formatter, 'format')
            assert hasattr(formatter, 'set_format')

    def test_formatters_handle_different_levels(self):
        """测试格式化器处理不同日志级别"""
        levels = [logging.DEBUG, logging.INFO, logging.WARNING, logging.ERROR, logging.CRITICAL]

        record_template = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="测试消息",
            args=(),
            exc_info=None
        )

        formatters = [JSONFormatter(), StructuredFormatter(), TextFormatter()]

        for formatter in formatters:
            for level in levels:
                record = logging.LogRecord(
                    name="test",
                    level=level,
                    pathname="test.py",
                    lineno=1,
                    msg=f"Level {level} message",
                    args=(),
                    exc_info=None
                )

                result = formatter.format(record)
                assert result is not None
                assert len(result) > 0

                # 对于JSON格式化器，验证是有效JSON
                if isinstance(formatter, JSONFormatter):
                    data = json.loads(result)
                    assert 'level' in data

    def test_formatters_handle_unicode(self):
        """测试格式化器处理Unicode字符"""
        unicode_message = "测试消息 🚀 with emoji 中文"

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=unicode_message,
            args=(),
            exc_info=None
        )

        formatters = [JSONFormatter(), StructuredFormatter(), TextFormatter()]

        for formatter in formatters:
            result = formatter.format(record)
            assert unicode_message in result

            # 对于JSON格式化器，确保Unicode正确编码
            if isinstance(formatter, JSONFormatter):
                data = json.loads(result)
                assert data['message'] == unicode_message


if __name__ == "__main__":
    pytest.main([__file__])
