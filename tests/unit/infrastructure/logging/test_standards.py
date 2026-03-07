#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单元测试 - 日志标准格式化器

测试各种标准日志格式化器的功能，包括ELK、Datadog、Loki等。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
import json
import logging
from datetime import datetime
from unittest.mock import patch, MagicMock

from src.infrastructure.logging.standards import (
    ELKStandardFormat,
    DatadogStandardFormat,
    LokiStandardFormat,
    GraylogStandardFormat,
    NewRelicStandardFormat,
    SplunkStandardFormat,
    FluentdStandardFormat,
    StandardLogEntry
)
from src.infrastructure.logging.standards.standard_formatter import StandardFormatter
from src.infrastructure.logging.standards.standard_manager import StandardFormatManager, StandardOutputConfig
from src.infrastructure.logging.standards.base_standard import StandardFormatType
from src.infrastructure.logging.core import LogLevel, LogCategory


class TestELKStandardFormat:
    """ELK标准格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = ELKStandardFormat()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None
        assert self.formatter.get_content_type() == "application/json"

    def test_format_log_entry_basic(self):
        """测试基本日志条目格式化"""
        log_entry = StandardLogEntry(
            timestamp=datetime(2022, 1, 1, 0, 0, 0),
            level=LogLevel.INFO,
            message='测试消息',
            source='test.logger'
        )

        result = self.formatter.format_log_entry(log_entry)

        # 验证结果是字符串或字典
        assert result is not None
        if isinstance(result, str):
            # 如果是字符串，应该是JSON
            data = json.loads(result)
            assert data['message'] == '测试消息'
        else:
            # 如果是字典，直接验证
            assert result['message'] == '测试消息'

    def test_format_log_entry_with_metadata(self):
        """测试包含元数据的日志条目格式化"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'ERROR',
            'logger_name': 'app.error',
            'message': '数据库连接失败',
            'correlation_id': 'abc-123',
            'user_id': 'user_456',
            'request_id': 'req_789',
            'extra_service': 'auth_service',
            'extra_version': '1.2.3'
        }

        result = self.formatter.format_log_entry(log_entry)
        data = json.loads(result)

        # 验证元数据字段
        assert data['correlation_id'] == 'abc-123'
        assert data['user_id'] == 'user_456'
        assert data['request_id'] == 'req_789'
        assert data['service'] == 'infrastructure'  # 默认服务名
        assert data['extra_version'] == '1.2.3'  # 额外字段

    def test_create_base_elk_entry(self):
        """测试创建ELK条目的基础字段"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            message="基础字段测试",
            category=LogCategory.BUSINESS,
            service="test-elk-service",
            host="test-host",
            trace_id="trace-elk-123",
            span_id="span-elk-456",
            user_id="user-elk-789",
            session_id="session-elk-abc",
            request_id="request-elk-def",
            correlation_id="correlation-elk-ghi"
        )
        
        result = self.formatter._create_base_elk_entry(entry)
        
        assert isinstance(result, dict)
        assert '@timestamp' in result
        assert result['level'] == 'WARNING'
        assert result['message'] == "基础字段测试"
        assert result['category'] == "business"
        assert result['service'] == "test-elk-service"
        assert result['host'] == "test-host"
        assert result['trace.id'] == "trace-elk-123"
        assert result['span.id'] == "span-elk-456"
        assert result['user.id'] == "user-elk-789"
        assert result['session.id'] == "session-elk-abc"
        assert result['request.id'] == "request-elk-def"
        assert result['correlation.id'] == "correlation-elk-ghi"

    def test_add_metadata_fields(self):
        """测试添加元数据字段"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            message="元数据字段测试",
            metadata={"security": "high", "priority": 1, "module": "auth"}
        )
        
        elk_entry = {"base": "data"}
        self.formatter._add_metadata_fields(elk_entry, entry)
        
        assert elk_entry["base"] == "data"
        assert "metadata.security" in elk_entry
        assert elk_entry["metadata.security"] == "high"
        assert "metadata.priority" in elk_entry
        assert elk_entry["metadata.priority"] == 1
        assert "metadata.module" in elk_entry
        assert elk_entry["metadata.module"] == "auth"

    def test_add_metadata_fields_no_metadata(self):
        """测试添加元数据字段（无元数据）"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="无元数据测试"
        )
        
        elk_entry = {"base": "data"}
        self.formatter._add_metadata_fields(elk_entry, entry)
        
        # 不应该添加任何metadata.*字段
        assert elk_entry == {"base": "data"}

    def test_add_optional_fields(self):
        """测试添加可选字段"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="可选字段测试",
            tags=["tag1", "tag2", "error"],
            extra_fields={"custom_field": "custom_value", "numeric": 42}
        )
        
        elk_entry = {"base": "data"}
        self.formatter._add_optional_fields(elk_entry, entry)
        
        assert elk_entry["base"] == "data"
        assert "tags" in elk_entry
        assert elk_entry["tags"] == ["tag1", "tag2", "error"]
        assert "custom_field" in elk_entry
        assert elk_entry["custom_field"] == "custom_value"
        assert elk_entry["numeric"] == 42

    def test_add_optional_fields_no_tags_extra(self):
        """测试添加可选字段（无标签和额外字段）"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="无可选字段测试"
        )
        
        elk_entry = {"base": "data"}
        self.formatter._add_optional_fields(elk_entry, entry)
        
        assert elk_entry == {"base": "data"}

    def test_add_elk_specific_fields(self):
        """测试添加ELK特定字段"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.CRITICAL,
            message="ELK特定字段测试",
            category=LogCategory.SECURITY,
            service="security-service",
            host="security-host",
            environment="production"
        )
        
        elk_entry = {"base": "data"}
        self.formatter._add_elk_specific_fields(elk_entry, entry)
        
        assert elk_entry["base"] == "data"
        assert elk_entry["log.level"] == "CRITICAL"
        assert elk_entry["log.logger"] == f"security.security-service"
        assert elk_entry["event.dataset"] == f"security.security-service"
        assert elk_entry["ecs.version"] == "1.12.0"
        
        # 验证agent字段
        assert "agent" in elk_entry
        agent = elk_entry["agent"]
        assert agent["name"] == "rqa2025-logging"
        assert agent["type"] == "infrastructure"
        assert agent["version"] == "1.0.0"
        
        assert elk_entry["host.name"] == "security-host"
        assert elk_entry["service.name"] == "security-service"
        assert elk_entry["service.environment"] == "production"

    def test_serialize_elk_entry(self):
        """测试序列化ELK条目"""
        elk_entry = {
            "test": "value",
            "number": 123,
            "nested": {"key": "value"}
        }
        
        result = self.formatter._serialize_elk_entry(elk_entry)
        
        assert isinstance(result, str)
        
        import json
        parsed = json.loads(result)
        assert parsed == elk_entry

    def test_supports_batch(self):
        """测试支持批量"""
        assert self.formatter.supports_batch() is True

    def test_format_batch(self):
        """测试批量格式化"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="批量消息1",
                category=LogCategory.SYSTEM
            ),
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.WARNING,
                message="批量消息2",
                category=LogCategory.BUSINESS
            )
        ]
        
        result = self.formatter.format_batch(entries)
        
        assert isinstance(result, str)
        # 应该以换行符结尾
        assert result.endswith('\n')
        
        # 验证包含两个JSON行
        lines = result.strip().split('\n')
        assert len(lines) == 2
        
        # 验证每行都是有效的JSON
        import json
        for line in lines:
            data = json.loads(line)
            assert 'message' in data

    def test_format_batch_empty(self):
        """测试批量格式化（空列表）"""
        result = self.formatter.format_batch([])
        assert result == ""

    def test_create_bulk_index_request(self):
        """测试创建Elasticsearch批量索引请求"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entries = [
            StandardLogEntry(
                timestamp=datetime(2023, 1, 15, 10, 30, 0),
                level=LogLevel.INFO,
                message="批量索引测试1",
                category=LogCategory.SYSTEM,
                trace_id="trace-123"
            ),
            StandardLogEntry(
                timestamp=datetime(2023, 1, 15, 10, 31, 0),
                level=LogLevel.ERROR,
                message="批量索引测试2",
                category=LogCategory.BUSINESS,
                trace_id="trace-456"
            )
        ]
        
        result = self.formatter.create_bulk_index_request(entries, "test-index")
        
        assert isinstance(result, str)
        assert result.endswith('\n')
        
        lines = result.strip().split('\n')
        # 每个条目应该有2行：一行元数据，一行文档
        assert len(lines) == 4
        
        # 验证元数据行
        import json
        meta_line_1 = json.loads(lines[0])
        assert "index" in meta_line_1
        assert "_index" in meta_line_1["index"]
        assert "_id" in meta_line_1["index"]
        
        # 验证文档行
        doc_line_1 = json.loads(lines[1])
        assert doc_line_1["message"] == "批量索引测试1"


class TestDatadogStandardFormat:
    """Datadog标准格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = DatadogStandardFormat()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None
        assert self.formatter.get_content_type() == "application/json"

    def test_format_log_entry_basic(self):
        """测试基本日志条目格式化"""
        log_entry = {
            'timestamp': '2022-01-01T00:00:00.000000',
            'level': 'WARNING',
            'logger_name': 'test.logger',
            'message': '警告消息',
            'ddtags': 'env:prod,service:myapp',
            'ddsource': 'python'
        }

        result = self.formatter.format_log_entry(log_entry)
        data = json.loads(result)

        # 验证Datadog格式
        assert data['level'] == 'WARNING'
        assert data['logger_name'] == 'test.logger'
        assert data['message'] == '警告消息'
        assert 'ddtags' in data
        assert 'ddsource' in data

    def test_format_log_entry_with_attributes(self):
        """测试包含属性的日志条目格式化"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'logger_name': 'app.request',
            'message': '处理用户请求',
            'attributes': {
                'user_id': '12345',
                'endpoint': '/api/users',
                'method': 'GET',
                'response_time': 150
            }
        }

        result = self.formatter.format_log_entry(log_entry)
        data = json.loads(result)

        # 验证属性字段 (在attributes子对象中)
        assert data['attributes']['user_id'] == '12345'
        assert data['attributes']['endpoint'] == '/api/users'
        assert data['attributes']['method'] == 'GET'
        assert data['attributes']['response_time'] == 150


class TestLokiStandardFormat:
    """Loki标准格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = LokiStandardFormat()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None
        assert self.formatter.get_content_type() == "application/json"

    def test_format_log_entry_basic(self):
        """测试基本日志条目格式化"""
        log_entry = {
            'timestamp': '2022-01-01T00:00:00.000000',
            'level': 'ERROR',
            'logger_name': 'test.logger',
            'message': '错误信息',
            'labels': {
                'app': 'myapp',
                'version': '1.0.0'
            }
        }

        result = self.formatter.format_log_entry(log_entry)
        data = json.loads(result)

        # 验证Loki格式 (实际使用entries结构)
        assert 'entries' in data
        assert len(data['entries']) > 0

        # 检查entries结构
        entry = data['entries'][0]
        assert 'line' in entry
        assert 'ts' in entry
        assert 'logger_name' in entry

        # 验证消息内容在line中
        line_content = entry['line']
        assert '错误信息' in line_content
        assert 'test.logger' in entry['logger_name']

    def test_format_log_entry_with_metadata(self):
        """测试包含元数据的日志条目格式化"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'logger_name': 'app.business',
            'message': '业务操作成功',
            'labels': {
                'service': 'order_service',
                'operation': 'create_order'
            },
            'correlation_id': 'corr-123',
            'user_id': 'user-456'
        }

        result = self.formatter.format_log_entry(log_entry)
        data = json.loads(result)

        # Loki格式化器返回entries结构
        assert 'entries' in data
        assert len(data['entries']) > 0
        entry = data['entries'][0]

        # 验证日志行包含必要信息
        line_content = entry['line']
        assert '业务操作成功' in line_content
        assert 'corr-123' in line_content
        assert 'user-456' in line_content

    def test_build_loki_labels(self):
        """测试构建Loki标签"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            category=LogCategory.SYSTEM,
            service="test_service",
            host="test_host",
            environment="test",
            trace_id="trace-123",
            correlation_id="corr-456"
        )
        
        labels = self.formatter._build_loki_labels(entry)
        labels_dict = json.loads(labels)
        
        assert "job" in labels_dict
        assert "service" in labels_dict
        assert "category" in labels_dict
        assert "level" in labels_dict
        assert "host" in labels_dict
        assert "environment" in labels_dict
        assert labels_dict["service"] == "test_service"
        assert labels_dict["host"] == "test_host"

    def test_add_optional_labels(self):
        """测试添加可选标签"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        labels = {}
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            message="Test message",
            category=LogCategory.SYSTEM,
            source="test_source",
            trace_id="trace-789",
            correlation_id="corr-789"
        )
        
        self.formatter._add_optional_labels(labels, entry)
        
        assert labels["source"] == "test_source"
        assert labels["trace_id"] == "trace-789"
        assert labels["correlation_id"] == "corr-789"

    def test_create_loki_log_line(self):
        """测试创建Loki日志行"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="Error occurred",
            category=LogCategory.ERROR,
            metadata={"error_code": "E001"},
            tags=["error", "critical"]
        )
        
        log_line = self.formatter._create_loki_log_line(entry)
        log_line_parsed = json.loads(log_line)
        
        assert isinstance(log_line_parsed, str)
        assert "Error occurred" in log_line_parsed
        assert "ERROR" in log_line_parsed

    def test_build_structured_data_for_loki(self):
        """测试为Loki构建结构化数据"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test message",
            category=LogCategory.BUSINESS,
            service="business_service",
            host="business_host",
            environment="production",
            trace_id="trace-123",
            span_id="span-456",
            user_id="user-789",
            session_id="session-abc",
            request_id="req-def",
            correlation_id="corr-ghi",
            metadata={"action": "create"},
            tags=["business", "api"],
            extra_fields={"request_time": 100}
        )
        
        structured_data = self.formatter._build_structured_data_for_loki(entry)
        data_parsed = json.loads(structured_data)
        
        assert "timestamp" in data_parsed
        assert "level" in data_parsed
        assert "category" in data_parsed
        assert "service" in data_parsed
        assert "trace_id" in data_parsed
        assert "user_id" in data_parsed
        assert "session_id" in data_parsed
        assert "metadata" in data_parsed
        assert "tags" in data_parsed
        assert "request_time" in data_parsed

    def test_build_base_structured_data(self):
        """测试构建基础结构化数据"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            message="Debug message",
            category=LogCategory.DEBUG,
            service="debug_service",
            host="debug_host",
            environment="development"
        )
        
        base_data = self.formatter._build_base_structured_data(entry)
        
        assert "timestamp" in base_data
        assert "level" in base_data
        assert "category" in base_data
        assert "service" in base_data
        assert base_data["service"] == "debug_service"
        assert base_data["host"] == "debug_host"
        assert base_data["environment"] == "development"

    def test_add_metadata_and_tags(self):
        """测试添加元数据和标签"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        data = {"existing": "value"}
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test",
            category=LogCategory.SYSTEM,
            metadata={"key1": "value1", "key2": "value2"},
            tags=["tag1", "tag2", "tag3"]
        )
        
        self.formatter._add_metadata_and_tags(data, entry)
        
        assert "existing" in data  # 原有数据应该保留
        assert "metadata" in data
        assert "tags" in data
        assert data["metadata"]["key1"] == "value1"
        assert "tag1" in data["tags"]

    def test_add_extra_fields_to_structured_data(self):
        """测试添加额外字段到结构化数据"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        data = {"base": "data"}
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            message="Warning",
            category=LogCategory.SYSTEM,
            extra_fields={"extra1": "value1", "extra2": 42, "extra3": True}
        )
        
        self.formatter._add_extra_fields_to_structured_data(data, entry)
        
        assert "base" in data  # 原有数据应该保留
        assert "extra1" in data
        assert "extra2" in data
        assert "extra3" in data
        assert data["extra1"] == "value1"
        assert data["extra2"] == 42
        assert data["extra3"] is True

    def test_build_loki_response(self):
        """测试构建Loki响应"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.CRITICAL,
            message="Critical error",
            category=LogCategory.SYSTEM,
            service="critical_service",
            extra_fields={"logger_name": "critical.logger"}
        )
        
        labels = json.dumps({"job": "rqa2025", "level": "critical"})
        log_line = json.dumps("Critical error occurred")
        
        response = self.formatter._build_loki_response(labels, entry, log_line)
        
        assert "labels" in response
        assert "entries" in response
        assert len(response["entries"]) == 1
        assert "ts" in response["entries"][0]
        assert "line" in response["entries"][0]
        assert "logger_name" in response["entries"][0]

    def test_supports_batch(self):
        """测试批量支持"""
        assert self.formatter.supports_batch() is True

    def test_format_batch(self):
        """测试批量格式化"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="Message 1",
                category=LogCategory.SYSTEM,
                service="batch_service"
            ),
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.WARNING,
                message="Message 2",
                category=LogCategory.SYSTEM,
                service="batch_service"
            )
        ]
        
        result = self.formatter.format_batch(entries)
        
        # 结果应该是Loki流格式
        assert isinstance(result, str) or isinstance(result, list)
        
        if isinstance(result, str):
            data = json.loads(result)
            assert "streams" in data

    def test_create_stream_key(self):
        """测试创建流键"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test",
            category=LogCategory.SYSTEM,
            service="stream_service",
            host="stream_host"
        )
        
        stream_key = self.formatter._create_stream_key(entry)
        
        assert isinstance(stream_key, str)
        assert len(stream_key) > 0

    def test_create_stream_labels(self):
        """测试创建流标签"""
        from src.infrastructure.logging.core import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="Test",
            category=LogCategory.SYSTEM,
            service="stream_service",
            environment="production"
        )
        
        stream_labels = self.formatter._create_stream_labels(entry)
        
        assert isinstance(stream_labels, str)
        labels_dict = json.loads(stream_labels)
        assert "job" in labels_dict
        assert "service" in labels_dict


class TestGraylogStandardFormat:
    """Graylog标准格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = GraylogStandardFormat()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None
        assert self.formatter.get_content_type() == "application/json"

    def test_format_log_entry_basic(self):
        """测试基本日志条目格式化"""
        log_entry = {
            'timestamp': '2022-01-01T00:00:00.000000',
            'level': 'CRITICAL',
            'logger_name': 'test.critical',
            'message': '严重错误',
            'facility': 'app',
            'version': '1.1'
        }

        result = self.formatter.format_log_entry(log_entry)
        data = json.loads(result)

        # 验证Graylog GELF格式
        assert data['version'] == '1.1'
        assert data['host'] == 'rqa2025-host'  # 默认主机名
        assert data['short_message'] == '严重错误'
        assert data['level'] == 2  # CRITICAL级别
        assert data['facility'] == 'rqa2025'  # 默认facility

    def test_format_log_entry_with_tracing(self):
        """测试包含追踪信息的日志条目格式化"""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'WARNING',
            'logger_name': 'app.tracing',
            'message': '追踪警告',
            'correlation_id': 'trace-123',
            'session_id': 'sess-456',
            'user_id': 'user-789',
            'request_id': 'req-101'
        }

        result = self.formatter.format_log_entry(log_entry)
        data = json.loads(result)

        # 验证追踪字段
        assert data['_correlation_id'] == 'trace-123'
        assert data['_session_id'] == 'sess-456'
        assert data['_user_id'] == 'user-789'
        assert data['_request_id'] == 'req-101'


class TestNewRelicStandardFormat:
    """New Relic标准格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = NewRelicStandardFormat()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None
        assert self.formatter.get_content_type() == "application/json"

    def test_format_log_entry_basic(self):
        """测试基本日志条目格式化"""
        log_entry = {
            'timestamp': '2022-01-01T00:00:00.000000',
            'level': 'INFO',
            'logger_name': 'test.logger',
            'message': 'New Relic日志',
            'attributes': {
                'service.name': 'my-service',
                'service.version': '2.0.0'
            }
        }

        result = self.formatter.format_log_entry(log_entry)
        data = json.loads(result)

        # 验证New Relic格式
        assert data['message'] == 'New Relic日志'
        assert data['level'] == 'INFO'
        assert data['logger_name'] == 'test.logger'
        assert data['timestamp'] is not None

        # 验证attributes (NewRelic格式化器使用默认值)
        assert data['service.name'] == 'rqa2025'  # 默认服务名
        assert data['service.version'] == '1.0.0'  # 默认版本


class TestSplunkStandardFormat:
    """Splunk标准格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = SplunkStandardFormat()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None
        assert self.formatter.get_content_type() == "application/json"

    def test_format_log_entry_basic(self):
        """测试基本日志条目格式化"""
        log_entry = {
            'timestamp': '2022-01-01T00:00:00.000000',
            'level': 'ERROR',
            'logger_name': 'test.splunk',
            'message': 'Splunk HEC事件',
            'event': {
                'action': 'failed_login',
                'user': 'testuser',
                'ip': '192.168.1.1'
            }
        }

        result = self.formatter.format_log_entry(log_entry)
        data = json.loads(result)

        # 验证Splunk HEC格式
        assert isinstance(data['event'], dict)
        assert data['event']['message'] == 'Splunk HEC事件'
        assert data['event']['level'] == 'ERROR'
        assert data['logger_name'] == 'test.splunk'
        assert 'time' in data

        # 验证额外事件数据 (在event字典的子event字段中)
        assert data['event']['event']['action'] == 'failed_login'
        assert data['event']['event']['user'] == 'testuser'
        assert data['event']['event']['ip'] == '192.168.1.1'


class TestFluentdStandardFormat:
    """Fluentd标准格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = FluentdStandardFormat()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None
        assert self.formatter.get_content_type() == "application/x-msgpack"

    def test_format_log_entry_basic(self):
        """测试基本日志条目格式化"""
        log_entry = {
            'timestamp': '2022-01-01T00:00:00.000000',
            'level': 'DEBUG',
            'logger_name': 'test.fluentd',
            'message': 'Fluentd消息',
            'tag': 'app.debug'
        }

        result = self.formatter.format_log_entry(log_entry)

        # Fluentd格式化器返回字典而不是JSON字符串
        assert isinstance(result, dict)
        assert 'tag' in result
        assert 'timestamp' in result
        assert 'record' in result

        # 解析record字段中的JSON字符串
        import json
        record_data = json.loads(result['record'])

        # 验证Fluentd格式
        assert record_data['message'] == 'Fluentd消息'
        assert record_data['level'] == 'DEBUG'
        assert record_data['logger_name'] == 'test.fluentd'
        assert record_data['tag'] == 'app.debug'

    def test_format_log_entry_with_standard_entry(self):
        """测试使用StandardLogEntry对象格式化日志条目"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="测试StandardLogEntry",
            category=LogCategory.SYSTEM,
            service="test-service",
            host="test-host",
            trace_id="trace-123",
            metadata={"key": "value"},
            tags=["tag1"],
            extra_fields={"extra": "data"}
        )
        
        result = self.formatter.format_log_entry(entry)
        
        assert isinstance(result, dict)
        assert 'tag' in result
        assert 'timestamp' in result
        assert 'record' in result
        assert result['tag'] == f"rqa2025.system.test-service"

    def test_create_fluentd_tag(self):
        """测试创建Fluentd标签"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            message="标签测试",
            category=LogCategory.BUSINESS,
            service="business-service"
        )
        
        tag = self.formatter._create_fluentd_tag(entry)
        
        assert tag == "rqa2025.business.business-service"
        assert isinstance(tag, str)

    def test_create_fluentd_tag_no_service(self):
        """测试创建Fluentd标签（无service）"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.ERROR,
            message="标签测试无service",
            category=LogCategory.SECURITY
        )
        
        tag = self.formatter._create_fluentd_tag(entry)
        
        assert tag == "rqa2025.security.infrastructure"

    def test_create_fluentd_record(self):
        """测试创建Fluentd记录"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.CRITICAL,
            message="记录测试",
            category=LogCategory.SECURITY,
            service="security-service",
            trace_id="trace-456",
            metadata={"security": "high"},
            tags=["security", "critical"]
        )
        
        result = self.formatter._create_fluentd_record(entry)
        
        # 结果应该是JSON字符串
        import json
        assert isinstance(result, str)
        
        record_data = json.loads(result)
        assert record_data['message'] == "记录测试"
        assert record_data['level'] == "CRITICAL"
        assert record_data['category'] == "security"
        assert record_data['service'] == "security-service"
        assert record_data['trace_id'] == "trace-456"
        assert record_data['metadata'] == {"security": "high"}
        assert record_data['tags'] == ["security", "critical"]

    def test_create_base_record(self):
        """测试创建基础记录"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="基础记录测试",
            category=LogCategory.BUSINESS,
            service="test-base-service",
            host="test-host",
            environment="test-env",
            trace_id="trace-base",
            span_id="span-base",
            user_id="user-base",
            session_id="session-base",
            request_id="request-base",
            correlation_id="correlation-base",
            source="test-source"
        )
        
        record = self.formatter._create_base_record(entry)
        
        assert isinstance(record, dict)
        assert record['message'] == "基础记录测试"
        assert record['level'] == "INFO"
        assert record['category'] == "business"
        assert record['service'] == "test-base-service"
        assert record['host'] == "test-host"
        assert record['environment'] == "test-env"
        assert record['trace_id'] == "trace-base"
        assert record['span_id'] == "span-base"
        assert record['user_id'] == "user-base"
        assert record['session_id'] == "session-base"
        assert record['request_id'] == "request-base"
        assert record['correlation_id'] == "correlation-base"
        assert record['source'] == "test-source"

    def test_add_metadata_to_record(self):
        """测试添加元数据到记录"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.WARNING,
            message="元数据测试",
            metadata={"test_key": "test_value", "number": 123}
        )
        
        record = {}
        self.formatter._add_metadata_to_record(record, entry)
        
        assert "metadata" in record
        assert record["metadata"] == {"test_key": "test_value", "number": 123}

    def test_add_metadata_to_record_no_metadata(self):
        """测试添加元数据到记录（无元数据）"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="无元数据测试"
        )
        
        record = {"existing": "data"}
        self.formatter._add_metadata_to_record(record, entry)
        
        # 不应该添加metadata字段
        assert "metadata" not in record
        assert record["existing"] == "data"

    def test_add_optional_fields_to_record(self):
        """测试添加可选字段到记录"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.DEBUG,
            message="可选字段测试",
            tags=["tag1", "tag2"],
            extra_fields={"extra_key": "extra_value", "number": 456}
        )
        
        record = {"base": "data"}
        self.formatter._add_optional_fields_to_record(record, entry)
        
        assert record["base"] == "data"
        assert "tags" in record
        assert record["tags"] == ["tag1", "tag2"]
        assert "extra_key" in record
        assert record["extra_key"] == "extra_value"
        assert record["number"] == 456

    def test_add_optional_fields_to_record_no_extra(self):
        """测试添加可选字段到记录（无额外字段）"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="无额外字段测试",
            tags=["single_tag"]
        )
        
        record = {"base": "data"}
        self.formatter._add_optional_fields_to_record(record, entry)
        
        assert record["base"] == "data"
        assert "tags" in record
        assert record["tags"] == ["single_tag"]
        # extra_fields为None，不应该添加

    def test_supports_batch(self):
        """测试支持批量"""
        assert self.formatter.supports_batch() is True

    def test_format_batch(self):
        """测试批量格式化"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="批量消息1",
                category=LogCategory.SYSTEM
            ),
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.WARNING,
                message="批量消息2",
                category=LogCategory.BUSINESS
            )
        ]
        
        result = self.formatter.format_batch(entries)
        
        assert isinstance(result, list)
        assert len(result) == 2
        
        for i, item in enumerate(result):
            assert isinstance(item, dict)
            assert 'tag' in item
            assert 'timestamp' in item
            assert 'record' in item

    def test_create_json_payload(self):
        """测试创建JSON负载"""
        from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
        
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="JSON负载测试",
                category=LogCategory.SYSTEM
            )
        ]
        
        result = self.formatter.create_json_payload(entries)
        
        assert isinstance(result, str)
        
        import json
        data = json.loads(result)
        assert isinstance(data, list)
        assert len(data) == 1
        
        payload_item = data[0]
        assert 'tag' in payload_item
        assert 'timestamp' in payload_item
        assert 'record' in payload_item

    def test_create_forward_payload_without_msgpack(self):
        """测试创建Forward协议负载（无msgpack）"""
        from unittest.mock import patch
        
        # Mock HAS_MSGPACK为False
        with patch('src.infrastructure.logging.standards.fluentd_standard.HAS_MSGPACK', False):
            from src.infrastructure.logging.core.interfaces import LogLevel, LogCategory
            
            entries = [
                StandardLogEntry(
                    timestamp=datetime.now(),
                    level=LogLevel.INFO,
                    message="No msgpack test",
                    category=LogCategory.SYSTEM
                )
            ]
            
            with pytest.raises(ImportError, match="msgpack is required"):
                self.formatter.create_forward_payload(entries)


class TestStandardsIntegration:
    """标准格式化器集成测试"""

    def test_all_formatters_implement_interface(self):
        """测试所有标准格式化器都正确实现了接口"""
        from src.infrastructure.logging.standards.base_standard import BaseStandardFormat

        formatters = [
            ELKStandardFormat(),
            DatadogStandardFormat(),
            LokiStandardFormat(),
            GraylogStandardFormat(),
            NewRelicStandardFormat(),
            SplunkStandardFormat(),
            FluentdStandardFormat()
        ]

        for formatter in formatters:
            assert isinstance(formatter, BaseStandardFormat)
            assert hasattr(formatter, 'format_log_entry')
            assert hasattr(formatter, 'get_content_type')
            assert callable(formatter.get_content_type)

    def test_formatters_handle_different_log_levels(self):
        """测试格式化器处理不同日志级别"""
        levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

        test_entry = {
            'timestamp': datetime.now().isoformat(),
            'logger_name': 'test.levels',
            'message': '测试级别',
            'level': 'INFO'  # 将被覆盖
        }

        formatters = [
            ELKStandardFormat(),
            DatadogStandardFormat(),
            GraylogStandardFormat(),
            NewRelicStandardFormat(),
            SplunkStandardFormat(),
            FluentdStandardFormat()
        ]

        for formatter in formatters:
            for level in levels:
                entry = test_entry.copy()
                entry['level'] = level

                result = formatter.format_log_entry(entry)
                assert result is not None
                assert len(result) > 0

                # 对于JSON格式化器，验证是有效JSON且包含级别
                if formatter.get_content_type() == "application/json":
                    data = json.loads(result)
                    assert 'level' in data or level in str(data)

    def test_formatters_handle_unicode_and_special_chars(self):
        """测试格式化器处理Unicode字符和特殊字符"""
        special_message = "测试消息 🚀 with emoji 中文字符 & < > \" '"

        test_entry = {
            'timestamp': datetime.now().isoformat(),
            'level': 'INFO',
            'logger_name': 'test.special',
            'message': special_message,
            'special_field': 'value with "quotes" and \'apostrophes\''
        }

        formatters = [
            ELKStandardFormat(),
            DatadogStandardFormat(),
            LokiStandardFormat(),
            GraylogStandardFormat(),
            NewRelicStandardFormat(),
            SplunkStandardFormat(),
            FluentdStandardFormat()
        ]

        for formatter in formatters:
            result = formatter.format_log_entry(test_entry)
            assert result is not None

            # 对于JSON格式化器，确保Unicode正确处理
            if formatter.get_content_type() == "application/json":
                data = json.loads(result)
                # 将数据转换为字符串，检查是否包含原始特殊字符（Unicode和特殊符号）
                data_str = json.dumps(data, ensure_ascii=False)
                assert '测试消息' in data_str  # 检查中文字符
                assert '🚀' in data_str  # 检查emoji
                assert '&' in data_str  # 检查HTML实体
                assert '<' in data_str  # 检查特殊字符

    def test_formatters_produce_valid_output(self):
        """测试格式化器产生有效输出"""
        test_entry = {
            'timestamp': '2022-01-01T00:00:00.000000',
            'level': 'INFO',
            'logger_name': 'test.valid',
            'message': '验证输出有效性',
            'correlation_id': 'test-123',
            'user_id': 'user-456'
        }

        formatters = [
            ELKStandardFormat(),
            DatadogStandardFormat(),
            LokiStandardFormat(),
            GraylogStandardFormat(),
            NewRelicStandardFormat(),
            SplunkStandardFormat(),
            FluentdStandardFormat()
        ]

        for formatter in formatters:
            result = formatter.format_log_entry(test_entry)

            # 验证输出不为空
            assert result is not None

            # 处理不同类型的返回值
            if isinstance(result, str):
                assert len(result.strip()) > 0

                # 对于JSON格式化器，验证是有效JSON
                if formatter.get_content_type() == "application/json":
                    try:
                        data = json.loads(result)
                        assert isinstance(data, dict)
                        # 验证包含原始消息
                        assert '验证输出有效性' in str(data)
                    except json.JSONDecodeError:
                        pytest.fail(f"Invalid JSON produced by {formatter.__class__.__name__}: {result}")

            elif isinstance(result, dict):
                # 对于返回dict的格式化器，验证dict不为空
                assert len(result) > 0

                # 如果是JSON格式化器，序列化为字符串验证
                if formatter.get_content_type() == "application/json":
                    json_str = json.dumps(result)
                    assert len(json_str.strip()) > 0
                    # 验证包含原始消息
                    assert '验证输出有效性' in str(result)

            else:
                pytest.fail(f"Unexpected result type from {formatter.__class__.__name__}: {type(result)}")


class TestStandardFormatter:
    """StandardFormatter统一格式化器测试"""

    def setup_method(self):
        """测试前准备"""
        self.formatter = StandardFormatter()

    def test_initialization(self):
        """测试初始化"""
        assert self.formatter is not None
        assert hasattr(self.formatter, '_formatters')
        assert isinstance(self.formatter._formatters, dict)

    def test_get_supported_formats(self):
        """测试获取支持的格式类型"""
        supported_formats = self.formatter.get_supported_formats()
        assert isinstance(supported_formats, list)
        assert len(supported_formats) > 0
        # 验证包含主要的格式类型
        assert StandardFormatType.ELK in supported_formats
        assert StandardFormatType.SPLUNK in supported_formats

    def test_get_content_type(self):
        """测试获取内容类型"""
        # 测试已知格式
        content_type = self.formatter.get_content_type(StandardFormatType.ELK)
        assert isinstance(content_type, str)
        assert content_type == "application/json"
        
        # 测试不支持的格式
        content_type_unknown = self.formatter.get_content_type("unknown_format")
        assert content_type_unknown == "application/json"  # 默认类型

    def test_supports_batch(self):
        """测试批量操作支持"""
        # 测试已知格式的批量支持
        elk_supports_batch = self.formatter.supports_batch(StandardFormatType.ELK)
        assert isinstance(elk_supports_batch, bool)
        
        # 测试不支持的格式
        unknown_supports_batch = self.formatter.supports_batch("unknown_format")
        assert unknown_supports_batch == False

    def test_format_log_entry(self):
        """测试格式化单个日志条目"""
        # 创建标准日志条目
        log_entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="测试消息",
            category=LogCategory.SYSTEM,
            source="test.logger"
        )
        
        # 测试ELK格式化
        result = self.formatter.format_log_entry(log_entry, StandardFormatType.ELK)
        assert result is not None
        assert isinstance(result, (str, dict))

    def test_format_log_entry_invalid_format(self):
        """测试无效格式类型"""
        log_entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="测试消息"
        )
        
        # 测试不支持的格式类型
        with pytest.raises(ValueError, match="不支持的格式类型"):
            self.formatter.format_log_entry(log_entry, "invalid_format")

    def test_format_batch(self):
        """测试批量格式化"""
        # 创建多个日志条目
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="批量测试消息1"
            ),
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.WARNING,
                message="批量测试消息2"
            )
        ]
        
        # 测试批量格式化
        result = self.formatter.format_batch(entries, StandardFormatType.ELK)
        assert result is not None

    def test_format_batch_invalid_format(self):
        """测试批量格式化无效格式"""
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="测试消息"
            )
        ]
        
        # 测试不支持的格式类型
        with pytest.raises(ValueError, match="不支持的格式类型"):
            self.formatter.format_batch(entries, "invalid_format")

    def test_create_standard_entry(self):
        """测试创建标准日志条目静态方法"""
        timestamp = datetime.now()
        
        entry = StandardFormatter.create_standard_entry(
            timestamp=timestamp,
            level=LogLevel.ERROR,
            message="静态方法测试",
            category=LogCategory.BUSINESS,
            source="static_test",
            host="test-host",
            service="test-service",
            environment="test",
            trace_id="trace-123",
            span_id="span-456",
            user_id="user-789",
            metadata={"key": "value"},
            tags=["tag1", "tag2"],
            extra_fields={"extra": "data"}
        )
        
        assert isinstance(entry, StandardLogEntry)
        assert entry.timestamp == timestamp
        assert entry.level == LogLevel.ERROR
        assert entry.message == "静态方法测试"
        assert entry.category == LogCategory.BUSINESS
        assert entry.source == "static_test"
        assert entry.host == "test-host"
        assert entry.service == "test-service"
        assert entry.environment == "test"
        assert entry.trace_id == "trace-123"
        assert entry.span_id == "span-456"
        assert entry.user_id == "user-789"
        assert entry.metadata == {"key": "value"}
        assert entry.tags == ["tag1", "tag2"]
        assert entry.extra_fields == {"extra": "data"}

    def test_create_standard_entry_defaults(self):
        """测试创建标准日志条目的默认值"""
        timestamp = datetime.now()
        
        # 使用最少的参数
        entry = StandardFormatter.create_standard_entry(
            timestamp=timestamp,
            level=LogLevel.INFO,
            message="默认值测试"
        )
        
        assert isinstance(entry, StandardLogEntry)
        assert entry.timestamp == timestamp
        assert entry.level == LogLevel.INFO
        assert entry.message == "默认值测试"
        assert entry.category == LogCategory.SYSTEM  # 默认值
        assert entry.source == ""  # 默认值
        assert entry.host == ""  # 默认值
        assert entry.service == ""  # 默认值
        assert entry.environment == "production"  # 默认值
        assert entry.metadata == {}  # 默认值
        assert entry.tags == []  # 默认值
        assert entry.extra_fields == {}  # 默认值

    def test_convert_from_internal_format(self):
        """测试从内部格式转换"""
        # 创建内部格式的日志记录
        internal_record = {
            "timestamp": datetime.now(),
            "level": LogLevel.WARNING,
            "message": "内部格式测试",
            "category": LogCategory.SYSTEM,
            "source": "internal.source",
            "host": "internal-host",
            "service": "internal-service",
            "environment": "staging",
            "trace_id": "internal-trace",
            "user_id": "internal-user",
            "metadata": {"internal": "data"},
            "tags": ["internal-tag"],
            "extra_fields": {"extra": "internal"}
        }
        
        # 转换
        entry = self.formatter.convert_from_internal_format(internal_record)
        
        assert isinstance(entry, StandardLogEntry)
        assert entry.timestamp == internal_record["timestamp"]
        assert entry.level == LogLevel.WARNING
        assert entry.message == "内部格式测试"
        assert entry.category == LogCategory.SYSTEM
        assert entry.source == "internal.source"
        assert entry.host == "internal-host"
        assert entry.service == "internal-service"
        assert entry.environment == "staging"
        assert entry.trace_id == "internal-trace"
        assert entry.user_id == "internal-user"
        assert entry.metadata == {"internal": "data"}
        assert entry.tags == ["internal-tag"]
        assert entry.extra_fields == {"extra": "internal"}

    def test_convert_from_internal_format_with_defaults(self):
        """测试从内部格式转换使用默认值"""
        # 创建最小内部格式记录
        internal_record = {
            "message": "最小内部格式"
        }
        
        # 转换
        entry = self.formatter.convert_from_internal_format(internal_record)
        
        assert isinstance(entry, StandardLogEntry)
        assert entry.message == "最小内部格式"
        assert entry.level == LogLevel.INFO  # 默认值
        assert entry.category == LogCategory.GENERAL  # 默认值
        assert entry.source == ""  # 默认值
        assert entry.host == ""  # 默认值
        assert entry.service == ""  # 默认值
        assert entry.environment == "production"  # 默认值
        assert entry.metadata == {}  # 默认值
        assert entry.tags == []  # 默认值
        assert entry.extra_fields == {}  # 默认值

    @patch('src.infrastructure.logging.standards.standard_formatter.ELKStandardFormat')
    def test_format_log_entry_validation_error(self, mock_elk_format):
        """测试格式化器验证错误"""
        # 模拟验证失败
        mock_formatter = MagicMock()
        mock_formatter.validate_entry.return_value = False
        mock_elk_format.return_value = mock_formatter
        
        # 重新创建formatter以使用mock
        formatter = StandardFormatter()
        formatter._formatters = {StandardFormatType.ELK: mock_formatter}
        
        log_entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="验证失败测试"
        )
        
        # 测试验证失败
        with pytest.raises(ValueError, match="无效的日志条目"):
            formatter.format_log_entry(log_entry, StandardFormatType.ELK)


class TestStandardFormatManager:
    """StandardFormatManager标准格式管理器测试"""

    def setup_method(self):
        """测试前准备"""
        self.manager = StandardFormatManager()

    def test_initialization(self):
        """测试初始化"""
        assert self.manager is not None
        assert hasattr(self.manager, 'formatter')
        assert hasattr(self.manager, 'configs')
        assert hasattr(self.manager, 'executor')
        assert isinstance(self.manager.formatter, StandardFormatter)
        assert isinstance(self.manager.configs, dict)

    def test_register_config(self):
        """测试注册配置"""
        config = StandardOutputConfig(
            format_type=StandardFormatType.ELK,
            endpoint="http://test.example.com",
            batch_size=100
        )
        
        self.manager.register_config("test-target", config)
        assert "test-target" in self.manager.configs
        assert self.manager.configs["test-target"] == config

    def test_unregister_config(self):
        """测试注销配置"""
        config = StandardOutputConfig(format_type=StandardFormatType.ELK)
        self.manager.register_config("temp-target", config)
        
        assert "temp-target" in self.manager.configs
        self.manager.unregister_config("temp-target")
        assert "temp-target" not in self.manager.configs

    def test_unregister_nonexistent_config(self):
        """测试注销不存在的配置"""
        # 不应该抛出异常
        self.manager.unregister_config("nonexistent")

    def test_get_config(self):
        """测试获取配置"""
        config = StandardOutputConfig(
            format_type=StandardFormatType.SPLUNK,
            endpoint="http://splunk.example.com"
        )
        self.manager.register_config("splunk-target", config)
        
        retrieved_config = self.manager.get_config("splunk-target")
        assert retrieved_config == config
        
        # 测试不存在的配置
        nonexistent = self.manager.get_config("nonexistent")
        assert nonexistent is None

    def test_format_for_target(self):
        """测试为目标格式化日志条目"""
        config = StandardOutputConfig(format_type=StandardFormatType.ELK)
        self.manager.register_config("elk-target", config)
        
        log_entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="测试格式化"
        )
        
        result = self.manager.format_for_target(log_entry, "elk-target")
        assert result is not None

    def test_format_for_target_invalid_target(self):
        """测试无效目标的格式化"""
        log_entry = StandardLogEntry(
            timestamp=datetime.now(),
            level=LogLevel.INFO,
            message="测试"
        )
        
        with pytest.raises(ValueError, match="未找到目标配置"):
            self.manager.format_for_target(log_entry, "invalid-target")

    def test_format_batch_for_target(self):
        """测试批量格式化"""
        config = StandardOutputConfig(format_type=StandardFormatType.ELK)
        self.manager.register_config("batch-target", config)
        
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="批量测试1"
            ),
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.WARNING,
                message="批量测试2"
            )
        ]
        
        result = self.manager.format_batch_for_target(entries, "batch-target")
        assert result is not None

    def test_format_batch_for_target_invalid(self):
        """测试无效目标的批量格式化"""
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="测试"
            )
        ]
        
        with pytest.raises(ValueError, match="未找到目标配置"):
            self.manager.format_batch_for_target(entries, "invalid-target")

    def test_send_batch_sync(self):
        """测试同步发送批次"""
        config = StandardOutputConfig(
            format_type=StandardFormatType.ELK,
            endpoint="http://test.example.com"
        )
        self.manager.register_config("sync-target", config)
        
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="同步测试"
            )
        ]
        
        result = self.manager.send_batch_sync(entries, "sync-target")
        assert result is not None
        assert result["status"] == "success"

    def test_get_supported_targets(self):
        """测试获取支持的目标"""
        # 初始为空
        targets = self.manager.get_supported_targets()
        assert isinstance(targets, list)
        
        # 添加配置后验证
        config = StandardOutputConfig(format_type=StandardFormatType.ELK)
        self.manager.register_config("test-target", config)
        
        targets = self.manager.get_supported_targets()
        assert "test-target" in targets

    def test_get_target_info(self):
        """测试获取目标信息"""
        config = StandardOutputConfig(
            format_type=StandardFormatType.ELK,
            endpoint="http://test.example.com",
            batch_size=200,
            timeout=60.0,
            compression=True,
            async_mode=True
        )
        self.manager.register_config("info-target", config)
        
        info = self.manager.get_target_info("info-target")
        assert isinstance(info, dict)
        assert info["format_type"] == "elk"
        assert info["endpoint"] == "http://test.example.com"
        assert info["batch_size"] == 200
        assert info["timeout"] == 60.0
        assert info["compression"] == True
        assert info["async_mode"] == True

    def test_get_target_info_invalid(self):
        """测试获取无效目标信息"""
        with pytest.raises(ValueError, match="未找到目标配置"):
            self.manager.get_target_info("invalid-target")

    def test_create_batch_processor(self):
        """测试创建批量处理器"""
        config = StandardOutputConfig(format_type=StandardFormatType.ELK)
        self.manager.register_config("processor-target", config)
        
        processor = self.manager.create_batch_processor("processor-target", batch_size=50)
        assert callable(processor)
        
        # 测试处理器功能
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="处理器测试"
            )
        ]
        
        result = processor(entries)
        assert isinstance(result, dict)

    def test_create_batch_processor_invalid_target(self):
        """测试为无效目标创建批量处理器"""
        with pytest.raises(ValueError, match="未找到目标配置"):
            self.manager.create_batch_processor("invalid-target")

    def test_create_sample_configs(self):
        """测试创建示例配置"""
        configs = self.manager.create_sample_configs()
        assert isinstance(configs, dict)
        assert len(configs) > 0
        
        # 验证配置结构
        for name, config in configs.items():
            assert isinstance(config, StandardOutputConfig)
            assert config.format_type is not None

    @pytest.mark.asyncio
    async def test_send_to_target_with_endpoint(self):
        """测试发送到有端点的目标"""
        config = StandardOutputConfig(
            format_type=StandardFormatType.ELK,
            endpoint="http://test.example.com"
        )
        self.manager.register_config("async-target", config)
        
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="异步发送测试"
            )
        ]
        
        result = await self.manager.send_to_target(entries, "async-target")
        assert result is not None
        assert result["status"] == "success"
        assert result["target"] == "elk"
        assert result["endpoint"] == "http://test.example.com"

    @pytest.mark.asyncio
    async def test_send_to_target_no_endpoint(self):
        """测试发送到没有配置端点的目标"""
        config = StandardOutputConfig(format_type=StandardFormatType.ELK)
        self.manager.register_config("no-endpoint-target", config)
        
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="测试"
            )
        ]
        
        with pytest.raises(ValueError, match="没有配置端点"):
            await self.manager.send_to_target(entries, "no-endpoint-target")

    @pytest.mark.asyncio
    async def test_send_to_target_invalid_target(self):
        """测试发送到无效目标"""
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="测试"
            )
        ]
        
        with pytest.raises(ValueError, match="未找到目标配置"):
            await self.manager.send_to_target(entries, "invalid-target")

    @pytest.mark.asyncio
    async def test_send_batch_async(self):
        """测试异步发送批次"""
        config = StandardOutputConfig(format_type=StandardFormatType.ELK)
        self.manager.register_config("async-batch-target", config)
        
        entries = [
            StandardLogEntry(
                timestamp=datetime.now(),
                level=LogLevel.INFO,
                message="异步批次测试"
            )
        ]
        
        result = await self.manager.send_batch_async(entries, "async-batch-target")
        assert result is not None
        assert result["status"] == "success"

    def test_mock_send_sync(self):
        """测试模拟同步发送"""
        config = StandardOutputConfig(
            format_type=StandardFormatType.ELK,
            endpoint="http://test.example.com"
        )
        
        test_data = {"message": "测试数据"}
        result = self.manager._mock_send_sync(config, test_data)
        
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["target"] == "elk"
        assert result["endpoint"] == "http://test.example.com"
        assert "timestamp" in result

    @pytest.mark.asyncio
    async def test_mock_send_to_endpoint(self):
        """测试模拟异步发送到端点"""
        config = StandardOutputConfig(
            format_type=StandardFormatType.SPLUNK,
            endpoint="http://splunk.example.com"
        )
        
        test_data = {"message": "异步测试数据"}
        result = await self.manager._mock_send_to_endpoint(config, test_data)
        
        assert isinstance(result, dict)
        assert result["status"] == "success"
        assert result["target"] == "splunk"
        assert result["endpoint"] == "http://splunk.example.com"
        assert "timestamp" in result


if __name__ == "__main__":
    pytest.main([__file__])
