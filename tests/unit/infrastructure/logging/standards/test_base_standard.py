"""
测试目标：提升standards/base_standard.py的真实覆盖率
实际导入和使用src.infrastructure.logging.standards.base_standard模块
"""

from datetime import datetime
from unittest.mock import Mock
import pytest

from src.infrastructure.logging.standards.base_standard import (
    BaseStandardFormat, StandardFormatType, StandardLogEntry
)
from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory


class TestBaseStandardFormat:
    """测试BaseStandardFormat类"""

    @pytest.fixture
    def base_formatter(self):
        """创建基础格式化器实例（使用具体实现）"""
        class ConcreteStandardFormat(BaseStandardFormat):
            def format_log_entry(self, entry):
                return {"message": "formatted"}

            def supports_batch(self):
                return False

        return ConcreteStandardFormat(StandardFormatType.ELK)

    def test_initialization(self, base_formatter):
        """测试初始化"""
        assert base_formatter.format_type == StandardFormatType.ELK

    def test_format_log_entry_implemented(self, base_formatter):
        """测试format_log_entry方法已实现"""
        entry = Mock()

        # ConcreteStandardFormat实现了这个方法
        result = base_formatter.format_log_entry(entry)
        assert result == {"message": "formatted"}

    def test_get_format_name(self, base_formatter):
        """测试获取格式名称"""
        assert base_formatter.format_type.value == "elk"

    def test_convert_log_level(self, base_formatter):
        """测试日志级别转换"""
        assert base_formatter.convert_log_level(LogLevel.INFO) == "INFO"
        assert base_formatter.convert_log_level(LogLevel.DEBUG) == "DEBUG"
        assert base_formatter.convert_log_level(LogLevel.ERROR) == "ERROR"

    def test_get_content_type(self, base_formatter):
        """测试获取内容类型"""
        content_type = base_formatter.get_content_type()
        assert content_type == "application/json"

    def test_validate_log_entry_valid(self, base_formatter):
        """测试验证有效日志条目"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message="Test message"
        )

        result = base_formatter.validate_entry(entry)
        assert result is True

    def test_validate_log_entry_invalid(self, base_formatter):
        """测试验证无效日志条目"""
        # 测试缺少必要字段的情况
        invalid_entries = [
            StandardLogEntry(timestamp=datetime(2024, 1, 1, 12, 0, 0), level=LogLevel.INFO, message=""),  # 空消息
            StandardLogEntry(timestamp=datetime(2024, 1, 1, 12, 0, 0), level=None, message="test"),  # 无效level
            StandardLogEntry(timestamp=datetime(2024, 1, 1, 12, 0, 0), level=LogLevel.INFO, message="test", category=None),  # 无效category
        ]

        for invalid_entry in invalid_entries:
            result = base_formatter.validate_entry(invalid_entry)
            assert result is False

    def test_create_standard_entry_from_dict(self, base_formatter):
        """测试从字典创建标准条目"""
        entry_dict = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "level": LogLevel.ERROR,
            "message": "Error message",
            "category": LogCategory.BUSINESS,
            "source": "test_source",
            "host": "test_host",
            "service": "test_service",
            "environment": "test_env",
            "trace_id": "trace123",
            "span_id": "span456",
            "user_id": "user789"
        }

        result = base_formatter._dict_to_standard_entry(entry_dict)

        assert isinstance(result, StandardLogEntry)
        assert result.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert result.level == LogLevel.ERROR
        assert result.message == "Error message"
        assert result.category == LogCategory.BUSINESS
        assert result.source == "test_source"
        assert result.host == "test_host"
        assert result.service == "test_service"
        assert result.environment == "test_env"
        assert result.trace_id == "trace123"
        assert result.span_id == "span456"
        assert result.user_id == "user789"

    def test_create_standard_entry_from_dict_minimal(self, base_formatter):
        """测试从字典创建标准条目（最小字段）"""
        entry_dict = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "level": LogLevel.INFO,
            "message": "Info message"
        }

        result = base_formatter._dict_to_standard_entry(entry_dict)

        assert isinstance(result, StandardLogEntry)
        assert result.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert result.level == LogLevel.INFO
        assert result.message == "Info message"
        assert result.category == LogCategory.SYSTEM  # 默认值
        assert result.environment == "production"  # 默认值

    def test_create_standard_entry_from_dict_missing_required(self, base_formatter):
        """测试从字典创建标准条目（缺少必要字段）"""
        incomplete_dict = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "level": LogLevel.INFO
            # 缺少message
        }

        # _dict_to_standard_entry提供默认值，不抛出异常
        result = base_formatter._dict_to_standard_entry(incomplete_dict)
        assert isinstance(result, StandardLogEntry)
        assert result.message == ""  # 默认空消息
        assert result.level == LogLevel.INFO


class TestStandardLogEntry:
    """测试StandardLogEntry类"""

    def test_initialization_full(self):
        """测试完整初始化"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.WARNING,
            message="Warning message",
            category=LogCategory.SECURITY,
            source="test_source",
            host="test_host",
            service="test_service",
            environment="staging",
            trace_id="trace123",
            span_id="span456",
            user_id="user789"
        )

        assert entry.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert entry.level == LogLevel.WARNING
        assert entry.message == "Warning message"
        assert entry.category == LogCategory.SECURITY
        assert entry.source == "test_source"
        assert entry.host == "test_host"
        assert entry.service == "test_service"
        assert entry.environment == "staging"
        assert entry.trace_id == "trace123"
        assert entry.span_id == "span456"
        assert entry.user_id == "user789"

    def test_initialization_defaults(self):
        """测试默认值初始化"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message="Info message"
        )

        assert entry.category == LogCategory.SYSTEM
        assert entry.source == ""
        assert entry.host == ""
        assert entry.service == ""
        assert entry.environment == "production"
        assert entry.trace_id is not None  # 自动生成
        assert entry.span_id is None
        assert entry.user_id is None

    def test_timestamp_conversions(self):
        """测试时间戳转换方法"""
        from src.infrastructure.logging.standards.base_standard import BaseStandardFormat

        dt = datetime(2024, 1, 1, 12, 0, 0)

        # 测试ISO格式转换
        iso = BaseStandardFormat.timestamp_to_iso(dt)
        assert iso == "2024-01-01T12:00:00"

        # 测试Unix毫秒时间戳转换
        unix_ms = BaseStandardFormat.timestamp_to_unix_ms(dt)
        expected_ms = int(dt.timestamp() * 1000)
        assert unix_ms == expected_ms

        # 测试Unix纳秒时间戳转换
        unix_ns = BaseStandardFormat.timestamp_to_unix_ns(dt)
        expected_ns = int(dt.timestamp() * 1_000_000_000)
        assert unix_ns == expected_ns


