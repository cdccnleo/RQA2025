"""
测试目标：提升standards/graylog_standard.py的真实覆盖率
实际导入和使用src.infrastructure.logging.standards.graylog_standard模块
"""

from datetime import datetime
from unittest.mock import Mock
import pytest

from src.infrastructure.logging.standards.graylog_standard import GraylogStandardFormat
from src.infrastructure.logging.standards.base_standard import StandardLogEntry, StandardFormatType
from src.infrastructure.logging.core.interfaces import LogLevel


class TestGraylogStandardFormat:
    """测试GraylogStandardFormat类"""

    @pytest.fixture
    def formatter(self):
        """创建格式化器实例"""
        return GraylogStandardFormat()

    @pytest.fixture
    def formatter_with_custom_facility(self):
        """创建带有自定义facility的格式化器实例"""
        return GraylogStandardFormat(facility="test-facility")

    def test_initialization_default(self, formatter):
        """测试默认参数初始化"""
        assert formatter.format_type == StandardFormatType.GRAYLOG
        assert formatter.facility == "rqa2025"

    def test_initialization_custom(self, formatter_with_custom_facility):
        """测试自定义facility初始化"""
        assert formatter_with_custom_facility.facility == "test-facility"

    def test_format_log_entry_with_standard_entry(self, formatter):
        """测试使用StandardLogEntry格式化"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message="Test message",
            metadata={"key": "value"}
        )

        result = formatter.format_log_entry(entry)

        # Result is JSON string, parse it
        import json
        parsed_result = json.loads(result)

        assert isinstance(parsed_result, dict)
        assert parsed_result["version"] == "1.1"
        assert parsed_result["host"] == "rqa2025-host"
        assert parsed_result["short_message"] == "Test message"
        assert parsed_result["timestamp"] == entry.timestamp.timestamp()
        assert parsed_result["level"] == 6  # INFO level
        assert parsed_result["_service"] == "infrastructure"
        assert parsed_result["_category"] == "system"
        assert parsed_result["_key"] == "value"

    def test_format_log_entry_with_dict(self, formatter):
        """测试使用字典格式化"""
        entry_dict = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "level": "ERROR",
            "message": "Error message",
            "metadata": {"error_code": 500}
        }

        result = formatter.format_log_entry(entry_dict)

        # 结果是JSON字符串，需要解析
        import json
        parsed_result = json.loads(result)

        assert isinstance(parsed_result, dict)
        assert parsed_result["short_message"] == "Error message"
        assert parsed_result["level"] == 3  # ERROR level
        assert parsed_result["_error_code"] == 500

    def test_format_log_entry_with_long_message(self, formatter):
        """测试长消息格式化"""
        long_message = "A" * 500  # 超过GELF短消息限制
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message=long_message,
            extra_fields={"logger_name": "test_logger"}
        )

        result = formatter.format_log_entry(entry)

        # Result is JSON string, parse it
        import json
        parsed_result = json.loads(result)

        # Check short message is truncated (source code uses json.dumps[:200])
        assert len(parsed_result["short_message"]) <= 200
        assert parsed_result["full_message"] == long_message  # 完整消息

    def test_create_base_gelf_entry(self, formatter):
        """测试创建基础GELF条目"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.WARNING,
            message="Warning message",
            extra_fields={"logger_name": "warn_logger"}
        )

        result = formatter._create_base_gelf_entry(entry)

        assert result["version"] == "1.1"
        assert result["host"] == "rqa2025-host"
        assert result["short_message"] == "Warning message"
        assert result["timestamp"] == entry.timestamp.timestamp()
        assert result["level"] == 4  # WARNING level
        assert result["facility"] == "rqa2025"

    def test_add_message_fields(self, formatter):
        """测试添加消息字段"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message="A" * 300,  # Long message to trigger full_message
            extra_fields={
                "module": "test_module",
                "function": "test_function",
                "line": 42,
                "formatted_message": "Formatted test message"
            }
        )

        gelf_entry = {}
        formatter._add_message_fields(gelf_entry, entry)

        # For long messages, full_message should be added
        assert "full_message" in gelf_entry
        assert gelf_entry["full_message"] == "A" * 300

    def test_add_tracing_fields_with_trace_id(self, formatter):
        """测试添加跟踪字段（有trace_id）"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message="Test message",
            trace_id="12345",
            span_id="67890"
        )

        gelf_entry = {}
        formatter._add_tracing_fields(gelf_entry, entry)

        assert gelf_entry["_trace_id"] == "12345"
        assert gelf_entry["_span_id"] == "67890"

    def test_add_tracing_fields_without_trace_id(self, formatter):
        """测试添加跟踪字段（无trace_id）"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            message="Test message",
            trace_id=None,
            span_id=None,
            correlation_id=None
        )

        gelf_entry = {}
        formatter._add_tracing_fields(gelf_entry, entry)

        # StandardLogEntry会自动生成trace_id和correlation_id
        assert "_trace_id" in gelf_entry
        assert "_correlation_id" in gelf_entry
        assert "_span_id" not in gelf_entry

    def test_add_user_session_fields(self, formatter):
        """测试添加用户会话字段"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message="Test message",
            user_id="user123",
            session_id="session456",
            request_id="req789"
        )

        gelf_entry = {}
        formatter._add_user_session_fields(gelf_entry, entry)

        assert gelf_entry["_user_id"] == "user123"
        assert gelf_entry["_session_id"] == "session456"
        assert gelf_entry["_request_id"] == "req789"

    def test_add_metadata_fields(self, formatter):
        """测试添加元数据字段"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            message="Test message",
            metadata={"custom_field": "custom_value", "nested": {"key": "value"}}
        )

        gelf_entry = {}
        formatter._add_metadata_fields(gelf_entry, entry)

        assert gelf_entry["_custom_field"] == "custom_value"
        assert gelf_entry["_nested"] == {"key": "value"}

    def test_add_tags_fields(self, formatter):
        """测试添加标签字段"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            message="Test message",
            tags=["tag1", "tag2", "tag3"]
        )

        gelf_entry = {}
        formatter._add_tags_fields(gelf_entry, entry)

        assert gelf_entry["_tags"] == "tag1,tag2,tag3"

    def test_add_extra_fields(self, formatter):
        """测试添加额外字段"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level="INFO",
            message="Test message",
            extra_fields={"extra1": "value1", "extra2": "value2"}
        )

        gelf_entry = {"existing": "value"}
        formatter._add_extra_fields(gelf_entry, entry)

        assert gelf_entry["existing"] == "value"
        assert gelf_entry["_extra1"] == "value1"
        assert gelf_entry["_extra2"] == "value2"

    def test_dict_to_standard_entry(self, formatter):
        """测试字典转换为标准条目"""
        entry_dict = {
            "timestamp": datetime(2024, 1, 1, 12, 0, 0),
            "level": "CRITICAL",
            "message": "Critical message",
            "logger_name": "critical_logger",
            "module": "critical_module",
            "function": "critical_function",
            "line": 999,
            "thread_id": 999,
            "process_id": 9999,
            "metadata": {"critical": True},
            "formatted_message": "Critical message"
        }

        result = formatter._dict_to_standard_entry(entry_dict)

        assert isinstance(result, StandardLogEntry)
        assert result.timestamp == datetime(2024, 1, 1, 12, 0, 0)
        assert result.level == LogLevel.CRITICAL
        assert result.message == "Critical message"

    def test_map_level_to_gelf_level(self, formatter):
        """测试映射日志级别到GELF级别"""
        # 测试标准级别映射
        assert formatter._map_level_to_gelf_level(LogLevel.DEBUG) == 7
        assert formatter._map_level_to_gelf_level(LogLevel.INFO) == 6
        assert formatter._map_level_to_gelf_level(LogLevel.WARNING) == 4
        assert formatter._map_level_to_gelf_level(LogLevel.ERROR) == 3
        assert formatter._map_level_to_gelf_level(LogLevel.CRITICAL) == 2

    def test_format_as_json_string(self, formatter):
        """测试格式化为JSON字符串"""
        entry = StandardLogEntry(
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            level=LogLevel.INFO,
            message="Test message",
            extra_fields={"logger_name": "test_logger"}
        )

        result = formatter.format_log_entry(entry)

        # 结果应该是一个JSON字符串
        assert isinstance(result, str)
        assert "Test message" in result
        assert '"version": "1.1"' in result

    def test_format_multiple_entries(self, formatter):
        """测试格式化多个条目"""
        entries = [
            StandardLogEntry(
                timestamp=datetime(2024, 1, 1, 12, 0, 0),
                level=LogLevel.INFO,
                message="Message 1"
            ),
            StandardLogEntry(
                timestamp=datetime(2024, 1, 1, 12, 1, 0),
                level=LogLevel.ERROR,
                message="Message 2"
            )
        ]

        results = []
        for entry in entries:
            result = formatter.format_log_entry(entry)
            results.append(result)

        assert len(results) == 2
        # Results are already JSON strings from format_log_entry
        import json
        result0 = json.loads(results[0])
        result1 = json.loads(results[1])

        assert result0["short_message"] == "Message 1"
        assert result0["level"] == 6  # INFO
        assert result1["short_message"] == "Message 2"
        assert result1["level"] == 3  # ERROR
