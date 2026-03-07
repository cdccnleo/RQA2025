"""
测试JSON日志格式化器

覆盖 json.py 中的 JSONFormatter 类
"""

import json
import logging
from src.infrastructure.logging.formatters.json import JSONFormatter


class TestJSONFormatter:
    """JSONFormatter 测试"""

    def test_init_default(self):
        """测试默认初始化"""
        formatter = JSONFormatter()

        assert formatter.pretty_print == False
        assert formatter.include_extra == True
        assert formatter.include_exc_info == True
        assert formatter.custom_fields == {}

    def test_init_with_config(self):
        """测试带配置初始化"""
        config = {
            'pretty_print': True,
            'include_extra': False,
            'include_exc_info': False,
            'custom_fields': {'service': 'test'}
        }
        formatter = JSONFormatter(config)

        assert formatter.pretty_print == True
        assert formatter.include_extra == False
        assert formatter.include_exc_info == False
        assert formatter.custom_fields == {'service': 'test'}

    def test_format_basic_record(self):
        """测试基本记录格式化"""
        formatter = JSONFormatter()
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
        data = json.loads(result)

        # 检查基础字段
        assert data['level'] == 'INFO'
        assert data['logger'] == 'test_logger'
        assert data['message'] == 'Test message'
        assert 'timestamp' in data
        assert data['file'] == 'test.py'
        assert data['line'] == 10

    def test_format_with_extra(self):
        """测试带额外字段的记录格式化"""
        formatter = JSONFormatter()
        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=20,
            msg="Error occurred",
            args=(),
            exc_info=None
        )
        # 添加额外字段
        record.user_id = 123
        record.request_id = "abc-123"

        result = formatter._format(record)
        data = json.loads(result)

        assert data['extra_user_id'] == 123
        assert data['extra_request_id'] == "abc-123"

    def test_format_pretty_print(self):
        """测试美化打印格式化"""
        config = {'pretty_print': True}
        formatter = JSONFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None
        )

        result = formatter._format(record)
        # 美化打印应该有多行
        assert '\n' in result

        # 验证仍然是有效JSON
        data = json.loads(result)
        assert data['message'] == 'Test'

    def test_format_without_extra(self):
        """测试不包含额外字段的格式化"""
        config = {'include_extra': False}
        formatter = JSONFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None
        )
        record.extra_field = "should_not_appear"

        result = formatter._format(record)
        data = json.loads(result)

        # 额外字段不应该出现
        assert 'extra_field' not in data

    def test_format_with_exception(self):
        """测试带异常信息的格式化"""
        formatter = JSONFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = logging.sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Exception occurred",
            args=(),
            exc_info=exc_info
        )

        result = formatter._format(record)
        data = json.loads(result)

        assert 'exception' in data
        assert 'traceback' in data['exception']
        assert 'ValueError' in data['exception']['type']

    def test_format_without_exception_info(self):
        """测试不包含异常信息的格式化"""
        config = {'include_exc_info': False}
        formatter = JSONFormatter(config)

        try:
            raise RuntimeError("Test")
        except RuntimeError:
            exc_info = logging.sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="",
            lineno=0,
            msg="Exception occurred",
            args=(),
            exc_info=exc_info
        )

        result = formatter._format(record)
        data = json.loads(result)

        # 不应该包含异常信息
        assert 'exception' not in data
        assert 'traceback' not in data

    def test_format_with_custom_fields(self):
        """测试自定义字段格式化"""
        config = {'custom_fields': {'service': 'my-service', 'version': '1.0'}}
        formatter = JSONFormatter(config)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="",
            lineno=0,
            msg="Test",
            args=(),
            exc_info=None
        )

        result = formatter._format(record)
        data = json.loads(result)

        assert data['service'] == 'my-service'
        assert data['version'] == '1.0'
