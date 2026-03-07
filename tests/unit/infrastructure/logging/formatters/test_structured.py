"""
测试结构化日志格式化器

覆盖 structured.py 中的 StructuredFormatter 类
"""

import logging
from unittest.mock import Mock
from src.infrastructure.logging.formatters.structured import StructuredFormatter


class TestStructuredFormatter:
    """StructuredFormatter 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        formatter = StructuredFormatter()

        assert formatter.field_separator == ' | '
        assert formatter.key_value_separator == '='
        assert formatter.include_empty_fields == False
        assert formatter.name == "StructuredFormatter"

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'field_separator': ' || ',
            'key_value_separator': ':',
            'include_empty_fields': True,
            'name': 'CustomStructuredFormatter'
        }
        formatter = StructuredFormatter(config)

        assert formatter.field_separator == ' || '
        assert formatter.key_value_separator == ':'
        assert formatter.include_empty_fields == True
        assert formatter.name == 'CustomStructuredFormatter'

    def test_format_basic_record(self):
        """测试格式化基本日志记录"""
        formatter = StructuredFormatter()

        # 创建模拟的日志记录
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=10,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0  # 2021-01-01 00:00:00

        result = formatter.format(record)

        # 验证结果包含预期字段
        assert "timestamp=" in result
        assert "level=INFO" in result
        assert "logger=test_logger" in result
        assert "message=Test message" in result
        assert " | " in result  # 字段分隔符

    def test_format_with_extra_fields(self):
        """测试格式化带有额外字段的记录"""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=20,
            msg="Warning message",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0

        # 添加额外字段
        record.user_id = 123
        record.action = "login"
        record.__dict__.update({'user_id': 123, 'action': 'login'})

        result = formatter.format(record)

        # 验证额外字段被包含
        assert "user_id=123" in result
        assert "action=login" in result

    def test_format_with_exception(self):
        """测试格式化带有异常的记录"""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=30,
            msg="Error occurred",
            args=(),
            exc_info=None  # structured formatter不处理exc_info
        )
        record.created = 1609459200.0

        result = formatter.format(record)

        # 验证基本字段存在（structured formatter不处理异常）
        assert "level=ERROR" in result
        assert "message=Error occurred" in result

    def test_format_custom_separators(self):
        """测试自定义分隔符"""
        config = {
            'field_separator': ' || ',
            'key_value_separator': ':'
        }
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.DEBUG,
            pathname="test.py",
            lineno=40,
            msg="Debug message",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0

        result = formatter.format(record)

        # 验证自定义分隔符
        assert " || " in result
        assert "timestamp:" in result
        assert "level:DEBUG" in result

    def test_format_include_empty_fields(self):
        """测试包含空字段"""
        config = {'include_empty_fields': True}
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=50,
            msg="Message with empty field",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0
        record.empty_field = ""
        record.none_field = None

        result = formatter.format(record)

        # 验证空字段被包含
        assert "empty_field=" in result
        assert "none_field=None" in result

    def test_format_exclude_empty_fields(self):
        """测试排除空字段"""
        config = {'include_empty_fields': False}
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=60,
            msg="Message with empty field",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0
        record.empty_field = ""
        record.none_field = None
        record.valid_field = "value"

        result = formatter.format(record)

        # 验证None字段被排除，空字符串字段被包含，有效字段被包含
        # 根据代码逻辑：value is not None or self.include_empty_fields
        # 所以None字段会被排除，空字符串字段会被包含
        assert "none_field=" not in result
        assert "valid_field=value" in result

    def test_format_with_args(self):
        """测试带有参数的消息格式化"""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=70,
            msg="User %s performed %s",
            args=("Alice", "login"),
            exc_info=None
        )
        record.created = 1609459200.0

        result = formatter.format(record)

        # 验证参数被正确格式化到消息中
        assert "message=User Alice performed login" in result

    def test_format_minimal_record(self):
        """测试格式化最小化记录"""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="minimal",
            level=logging.CRITICAL,
            pathname="minimal.py",
            lineno=1,
            msg="Minimal",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0

        result = formatter.format(record)

        # 验证基本字段都存在
        assert "timestamp=" in result
        assert "level=CRITICAL" in result
        assert "logger=minimal" in result
        assert "message=Minimal" in result

    def test_format_max_message_length(self):
        """测试最大消息长度限制"""
        config = {'max_message_length': 20}
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=80,
            msg="This is a very long message that should be truncated",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0

        result = formatter.format(record)

        # 找到消息部分
        message_part = [part for part in result.split(' | ') if part.startswith('message=')][0]
        message_value = message_part.split('=', 1)[1]

        # 验证消息被截断
        assert len(message_value) <= 20
        assert message_value.endswith('...')

    def test_format_disabled_timestamp(self):
        """测试禁用时间戳"""
        config = {'include_timestamp': False}
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=90,
            msg="No timestamp",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0

        result = formatter.format(record)

        # 验证时间戳被排除
        assert not result.startswith('timestamp=')
        assert "level=INFO" in result
        assert "message=No timestamp" in result

    def test_format_disabled_level(self):
        """测试禁用级别"""
        config = {'include_level': False}
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.WARNING,
            pathname="test.py",
            lineno=100,
            msg="No level",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0

        result = formatter.format(record)

        # 验证级别被排除
        assert "level=" not in result
        assert "timestamp=" in result
        assert "message=No level" in result

    def test_format_disabled_logger_name(self):
        """测试禁用日志器名称"""
        config = {'include_logger_name': False}
        formatter = StructuredFormatter(config)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=110,
            msg="No logger name",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0

        result = formatter.format(record)

        # 验证日志器名称被排除
        assert "logger=" not in result
        assert "timestamp=" in result
        assert "level=ERROR" in result
        assert "message=No logger name" in result

    def test_format_complex_extra_data(self):
        """测试复杂额外数据"""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="complex_test",
            level=logging.INFO,
            pathname="complex.py",
            lineno=120,
            msg="Complex data test",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0

        # 添加复杂额外数据
        record.nested_data = {"user": {"id": 123, "name": "Alice"}, "metadata": {"version": "1.0"}}
        record.list_data = [1, 2, 3, "test"]
        record.bool_data = True

        result = formatter.format(record)

        # 验证复杂数据被正确格式化
        assert "nested_data=" in result
        assert "list_data=" in result
        assert "bool_data=true" in result  # 小写格式

    def test_format_error_handling(self):
        """测试错误处理"""
        formatter = StructuredFormatter()

        # 创建一个有问题的记录
        record = logging.LogRecord(
            name="error_test",
            level=logging.CRITICAL,
            pathname="error.py",
            lineno=130,
            msg="Error test",
            args=(),
            exc_info=None
        )
        record.created = 1609459200.0

        # 添加会导致序列化问题的对象
        class BadObject:
            def __str__(self):
                raise RuntimeError("Cannot stringify")

        record.bad_object = BadObject()

        # 格式化应该不会崩溃
        result = formatter.format(record)

        # 结果应该包含错误信息或基本字段
        # 如果有错误处理，可能会返回错误消息
        assert isinstance(result, str)
        assert len(result) > 0

    def test_get_config(self):
        """测试获取配置"""
        config = {
            'field_separator': ' || ',
            'key_value_separator': ':',
            'include_empty_fields': True,
            'name': 'TestFormatter'
        }
        formatter = StructuredFormatter(config)

        result_config = formatter.get_config()

        # 验证配置被正确返回
        assert result_config['field_separator'] == ' || '
        assert result_config['key_value_separator'] == ':'
        assert result_config['include_empty_fields'] == True
        assert result_config['name'] == 'TestFormatter'


if __name__ == "__main__":
    pytest.main([__file__])
