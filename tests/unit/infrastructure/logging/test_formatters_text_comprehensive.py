#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 文本日志格式化器深度测试
测试TextFormatter的核心格式化功能、边界条件和错误处理
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import logging
import time
from datetime import datetime
from unittest.mock import patch

from infrastructure.logging.formatters.text import TextFormatter
from infrastructure.logging.core.interfaces import LogFormat


class TestTextFormatterInitialization:
    """TextFormatter初始化测试"""

    def test_initialization_default(self):
        """测试默认初始化"""
        formatter = TextFormatter()
        assert formatter.name == "TextFormatter"
        assert formatter.date_format == '%Y-%m-%d %H:%M:%S'
        assert formatter.include_level is True
        assert formatter.include_logger_name is True
        assert formatter.include_timestamp is True
        assert formatter.max_message_length == 0
        assert formatter.template == '{timestamp} {level} {logger}: {message}'

    def test_initialization_custom_config(self):
        """测试自定义配置初始化"""
        config = {
            'name': 'CustomTextFormatter',
            'date_format': '%Y/%m/%d %H-%M-%S',
            'include_level': False,
            'include_logger_name': False,
            'include_timestamp': False,
            'max_message_length': 100,
            'template': '{level} - {message}'
        }
        formatter = TextFormatter(config)
        assert formatter.name == 'CustomTextFormatter'
        assert formatter.date_format == '%Y/%m/%d %H-%M-%S'
        assert formatter.include_level is False
        assert formatter.include_logger_name is False
        assert formatter.include_timestamp is False
        assert formatter.max_message_length == 100
        assert formatter.template == '{level} - {message}'

    def test_initialization_empty_config(self):
        """测试空配置初始化"""
        formatter = TextFormatter({})
        assert formatter.name == "TextFormatter"
        assert formatter.template == '{timestamp} {level} {logger}: {message}'


class TestTextFormatterBasicFormatting:
    """TextFormatter基本格式化测试"""

    @pytest.fixture
    def formatter(self):
        """TextFormatter fixture"""
        return TextFormatter()

    @pytest.fixture
    def sample_record(self):
        """创建示例日志记录"""
        record = logging.LogRecord(
            name='TestLogger',
            level=logging.INFO,
            pathname='test.py',
            lineno=10,
            msg='Test message',
            args=(),
            exc_info=None
        )
        record.created = time.time()
        return record

    def test_format_basic_record(self, formatter, sample_record):
        """测试基本记录格式化"""
        result = formatter.format(sample_record)

        # 验证包含所有必需的部分
        assert 'INFO' in result  # 日志级别
        assert 'TestLogger' in result  # 日志器名称
        assert 'Test message' in result  # 消息
        assert ':' in result  # 分隔符

    def test_format_with_timestamp(self, formatter, sample_record):
        """测试带时间戳的格式化"""
        result = formatter.format(sample_record)

        # 应该包含时间戳
        parts = result.split()
        timestamp_part = parts[0] + ' ' + parts[1]

        # 验证时间戳格式
        try:
            datetime.strptime(timestamp_part, '%Y-%m-%d %H:%M:%S')
        except ValueError:
            pytest.fail(f"Invalid timestamp format: {timestamp_part}")

    def test_format_without_level(self, sample_record):
        """测试不包含级别的格式化"""
        config = {'include_level': False}
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        # 不应该包含INFO
        assert 'INFO' not in result
        assert 'Test message' in result

    def test_format_without_logger_name(self, sample_record):
        """测试不包含日志器名称的格式化"""
        config = {'include_logger_name': False}
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        # 不应该包含TestLogger
        assert 'TestLogger' not in result
        assert 'Test message' in result

    def test_format_without_timestamp(self, sample_record):
        """测试不包含时间戳的格式化"""
        config = {'include_timestamp': False}
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        # 时间戳部分应该是空的，格式应该是" INFO TestLogger: Test message"
        assert result.startswith(' ')  # 以空格开头（时间戳为空）
        assert 'INFO' in result
        assert 'TestLogger' in result
        assert 'Test message' in result


class TestTextFormatterTemplateCustomization:
    """TextFormatter模板自定义测试"""

    @pytest.fixture
    def sample_record(self):
        """创建示例日志记录"""
        record = logging.LogRecord(
            name='TestLogger',
            level=logging.WARNING,
            pathname='test.py',
            lineno=20,
            msg='Warning message',
            args=(),
            exc_info=None
        )
        record.created = time.time()
        return record

    def test_custom_template_simple(self, sample_record):
        """测试简单自定义模板"""
        config = {'template': '{level} | {message}'}
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        assert result == 'WARNING | Warning message'

    def test_custom_template_partial(self, sample_record):
        """测试部分字段的自定义模板"""
        config = {
            'template': '{logger} -> {level}: {message}',
            'include_timestamp': False
        }
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        assert result == 'TestLogger -> WARNING: Warning message'

    def test_custom_template_all_fields(self, sample_record):
        """测试包含所有字段的自定义模板"""
        config = {'template': '[{timestamp}] {level} in {logger}: {message}'}
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        assert '[202' in result  # 时间戳
        assert 'WARNING' in result
        assert 'TestLogger' in result
        assert 'Warning message' in result

    def test_template_missing_field(self, sample_record):
        """测试模板缺少字段"""
        config = {'template': '{level} - {message}'}  # 缺少timestamp和logger
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        assert result == 'WARNING - Warning message'

    def test_template_invalid_placeholder(self, sample_record):
        """测试无效的占位符"""
        config = {'template': '{invalid_field} {level} {message}'}
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        # 应该使用错误后备格式
        assert '[FORMAT_ERROR]' in result
        assert 'Warning message' in result


class TestTextFormatterMessageTruncation:
    """TextFormatter消息截断测试"""

    @pytest.fixture
    def formatter(self):
        """TextFormatter fixture"""
        return TextFormatter()

    @pytest.fixture
    def sample_record(self):
        """创建示例日志记录"""
        return logging.LogRecord(
            name='TestLogger',
            level=logging.ERROR,
            pathname='test.py',
            lineno=30,
            msg='This is a very long message that should be truncated',
            args=(),
            exc_info=None
        )

    def test_no_truncation_by_default(self, formatter, sample_record):
        """测试默认不截断"""
        result = formatter.format(sample_record)

        assert 'This is a very long message that should be truncated' in result

    def test_truncation_enabled(self, sample_record):
        """测试启用截断"""
        config = {'max_message_length': 20}
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        # 查找消息部分
        message_part = result.split(': ')[-1] if ': ' in result else result

        assert len(message_part) <= 20
        assert message_part.endswith('...')  # 应该以省略号结束

    def test_truncation_exact_length(self, sample_record):
        """测试精确长度截断"""
        config = {'max_message_length': 10}
        formatter = TextFormatter(config)

        result = formatter.format(sample_record)

        message_part = result.split(': ')[-1] if ': ' in result else result

        assert len(message_part) == 10
        assert message_part.endswith('...')

    def test_truncation_short_message(self, sample_record):
        """测试短消息不截断"""
        short_record = logging.LogRecord(
            name='TestLogger',
            level=logging.INFO,
            pathname='test.py',
            lineno=40,
            msg='Short',
            args=(),
            exc_info=None
        )

        config = {'max_message_length': 20}
        formatter = TextFormatter(config)

        result = formatter.format(short_record)

        assert 'Short' in result
        assert '...' not in result  # 不应该截断


class TestTextFormatterErrorHandling:
    """TextFormatter错误处理测试"""

    @pytest.fixture
    def formatter(self):
        """TextFormatter fixture"""
        return TextFormatter()

    def test_format_error_fallback(self, formatter):
        """测试格式化错误时的后备格式"""
        # 创建一个会导致格式化的记录
        record = logging.LogRecord(
            name='TestLogger',
            level=logging.CRITICAL,
            pathname='test.py',
            lineno=50,
            msg='Error message',
            args=(),
            exc_info=None
        )

        # Mock内部方法使其抛出异常
        with patch.object(formatter, '_format', side_effect=ValueError("Format error")):
            result = formatter.format(record)

            assert '[FORMAT_ERROR]' in result
            assert 'Error message' in result
            assert 'Format error' in result

    def test_template_format_error_fallback(self):
        """测试模板格式化错误的后备处理"""
        config = {'template': '{invalid_field} {level} {message}'}
        formatter = TextFormatter(config)

        record = logging.LogRecord(
            name='TestLogger',
            level=logging.ERROR,
            pathname='test.py',
            lineno=60,
            msg='Template error test',
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # 应该使用后备格式
        assert '[FORMAT_ERROR]' in result
        assert 'Template error test' in result

    def test_timestamp_format_error(self):
        """测试时间戳格式化错误"""
        config = {'date_format': '%invalid_format'}
        formatter = TextFormatter(config)

        record = logging.LogRecord(
            name='TestLogger',
            level=logging.WARNING,
            pathname='test.py',
            lineno=70,
            msg='Timestamp error test',
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # 应该使用后备格式
        assert '[FORMAT_ERROR]' in result
        assert 'Timestamp error test' in result


class TestTextFormatterSpecialCases:
    """TextFormatter特殊情况测试"""

    def test_unicode_message_handling(self):
        """测试Unicode消息处理"""
        formatter = TextFormatter()

        record = logging.LogRecord(
            name='UnicodeLogger',
            level=logging.INFO,
            pathname='test.py',
            lineno=80,
            msg='测试消息 🚀 Unicode: ∑∆∞',
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert '测试消息' in result
        assert '🚀' in result
        assert 'Unicode:' in result
        assert '∑∆∞' in result

    def test_empty_message_handling(self):
        """测试空消息处理"""
        formatter = TextFormatter()

        record = logging.LogRecord(
            name='EmptyLogger',
            level=logging.DEBUG,
            pathname='test.py',
            lineno=90,
            msg='',
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert 'DEBUG' in result
        assert 'EmptyLogger' in result
        # 空消息也应该被正确处理

    def test_none_message_handling(self):
        """测试None消息处理"""
        formatter = TextFormatter()

        record = logging.LogRecord(
            name='NoneLogger',
            level=logging.INFO,
            pathname='test.py',
            lineno=100,
            msg=None,
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        # None消息应该被正确处理，转换为字符串'None'
        assert 'INFO' in result
        assert 'NoneLogger' in result
        assert 'None' in result  # msg=None 转换为字符串'None'

    def test_long_logger_name_handling(self):
        """测试长日志器名称处理"""
        formatter = TextFormatter()

        long_name = 'a' * 200  # 很长的名称
        record = logging.LogRecord(
            name=long_name,
            level=logging.WARNING,
            pathname='test.py',
            lineno=110,
            msg='Long name test',
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert long_name in result
        assert 'WARNING' in result
        assert 'Long name test' in result

    def test_special_characters_in_logger_name(self):
        """测试日志器名称包含特殊字符"""
        formatter = TextFormatter()

        special_name = 'logger-with.dots.and-dashes_underscores'
        record = logging.LogRecord(
            name=special_name,
            level=logging.ERROR,
            pathname='test.py',
            lineno=120,
            msg='Special chars test',
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert special_name in result
        assert 'ERROR' in result
        assert 'Special chars test' in result


class TestTextFormatterConfigurationEdgeCases:
    """TextFormatter配置边界条件测试"""

    def test_minimal_config(self):
        """测试最小配置"""
        config = {
            'include_level': False,
            'include_logger_name': False,
            'include_timestamp': False,
            'template': '{message}'
        }
        formatter = TextFormatter(config)

        record = logging.LogRecord(
            name='MinimalLogger',
            level=logging.CRITICAL,
            pathname='test.py',
            lineno=130,
            msg='Minimal config test',
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert result == 'Minimal config test'

    def test_maximal_config(self):
        """测试最大化配置"""
        config = {
            'include_level': True,
            'include_logger_name': True,
            'include_timestamp': True,
            'max_message_length': 50,
            'template': '[{timestamp}] [{level}] [{logger}] {message}'
        }
        formatter = TextFormatter(config)

        record = logging.LogRecord(
            name='MaximalLogger',
            level=logging.DEBUG,
            pathname='test.py',
            lineno=140,
            msg='This is a very long message that should be truncated to fit the limit',
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert '[202' in result  # 时间戳
        assert '[DEBUG]' in result
        assert '[MaximalLogger]' in result
        assert '...' in result  # 截断标记

    def test_zero_max_length(self):
        """测试最大长度为0（不限制）"""
        config = {'max_message_length': 0}
        formatter = TextFormatter(config)

        long_message = 'x' * 1000
        record = logging.LogRecord(
            name='ZeroLengthLogger',
            level=logging.INFO,
            pathname='test.py',
            lineno=150,
            msg=long_message,
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert long_message in result  # 不应该截断

    def test_negative_max_length(self):
        """测试负的最大长度"""
        config = {'max_message_length': -1}
        formatter = TextFormatter(config)

        record = logging.LogRecord(
            name='NegativeLengthLogger',
            level=logging.WARNING,
            pathname='test.py',
            lineno=160,
            msg='Negative length test',
            args=(),
            exc_info=None
        )

        result = formatter.format(record)

        assert 'Negative length test' in result  # 不应该截断


class TestTextFormatterPerformance:
    """TextFormatter性能测试"""

    @pytest.fixture
    def formatter(self):
        """TextFormatter fixture"""
        return TextFormatter()

    def test_bulk_formatting_performance(self, formatter):
        """测试批量格式化性能"""
        import time

        # 创建大量记录
        records = []
        for i in range(1000):
            record = logging.LogRecord(
                name=f'BulkLogger{i}',
                level=logging.INFO,
                pathname='test.py',
                lineno=i,
                msg=f'Bulk message {i}',
                args=(),
                exc_info=None
            )
            record.created = time.time()
            records.append(record)

        # 批量格式化
        start_time = time.time()
        results = [formatter.format(record) for record in records]
        end_time = time.time()

        duration = end_time - start_time

        # 验证结果
        assert len(results) == 1000
        for i, result in enumerate(results):
            assert f'Bulk message {i}' in result

        # 性能检查：1000个记录应该在合理时间内完成
        assert duration < 2.0, f"Bulk formatting too slow: {duration:.2f}s for 1000 records"

        messages_per_second = len(records) / duration
        print(f"TextFormatter performance: {messages_per_second:.0f} messages/second")

    def test_complex_template_performance(self):
        """测试复杂模板性能"""
        config = {
            'template': '[{timestamp}] [{level}] [{logger}] {message}',
            'include_timestamp': True,
            'include_level': True,
            'include_logger_name': True
        }
        formatter = TextFormatter(config)

        record = logging.LogRecord(
            name='ComplexTemplateLogger',
            level=logging.ERROR,
            pathname='test.py',
            lineno=170,
            msg='Complex template performance test',
            args=(),
            exc_info=None
        )

        # 多次格式化测试性能
        num_iterations = 1000
        start_time = time.time()

        for _ in range(num_iterations):
            result = formatter.format(record)

        end_time = time.time()

        duration = end_time - start_time

        # 验证结果格式正确
        assert '[202' in result
        assert '[ERROR]' in result
        assert '[ComplexTemplateLogger]' in result
        assert 'Complex template performance test' in result

        # 性能检查
        assert duration < 1.0, f"Complex template too slow: {duration:.2f}s for {num_iterations} iterations"
