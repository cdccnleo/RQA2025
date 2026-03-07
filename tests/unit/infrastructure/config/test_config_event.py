#!/usr/bin/env python3
"""
配置事件管理测试
测试配置相关的事件类和功能
"""

import pytest
import time
import uuid
from unittest.mock import patch, MagicMock
from datetime import datetime
from src.infrastructure.config.config_event import (
    ConfigEvent,
    ConfigChangeEvent,
    ConfigLoadEvent,
    ConfigValidationEvent,
    ConfigReloadEvent,
    ConfigBackupEvent,
    ConfigErrorEvent
)


class TestConfigEvent:
    """测试基础配置事件类"""

    def test_init_default(self):
        """测试默认初始化"""
        event = ConfigEvent()

        assert event.event_type == "generic"
        assert event.data == {}
        assert event.source == "config_system"
        assert isinstance(event.timestamp, float)
        assert isinstance(event.event_id, str)
        assert len(event.event_id) == 36  # UUID4长度

    def test_init_with_params(self):
        """测试带参数初始化"""
        data = {"key": "value"}
        event = ConfigEvent("test_event", data, "test_source")

        assert event.event_type == "test_event"
        assert event.data == data
        assert event.source == "test_source"
        assert event.data is not data  # 应该深拷贝

    def test_data_deep_copy(self):
        """测试数据深拷贝"""
        original_data = {"nested": {"key": "value"}}
        event = ConfigEvent(data=original_data)

        # 修改原始数据，不应该影响事件数据
        original_data["nested"]["key"] = "modified"
        assert event.data["nested"]["key"] == "value"

    def test_to_dict(self):
        """测试转换为字典"""
        event = ConfigEvent("test_event", {"key": "value"}, "test_source")
        event_dict = event.to_dict()

        assert event_dict["event_id"] == event.event_id
        assert event_dict["event_type"] == "test_event"
        assert event_dict["data"] == {"key": "value"}
        assert event_dict["source"] == "test_source"
        assert event_dict["timestamp"] == event.timestamp
        assert "datetime" in event_dict

        # 验证datetime格式
        datetime_str = event_dict["datetime"]
        parsed_datetime = datetime.fromisoformat(datetime_str)
        assert isinstance(parsed_datetime, datetime)


class TestConfigChangeEvent:
    """测试配置变更事件"""

    def test_init_basic(self):
        """测试基本初始化"""
        event = ConfigChangeEvent("test.key", "old", "new")

        assert event.event_type == "config_changed"
        assert event.key == "test.key"
        assert event.old_value == "old"
        assert event.new_value == "new"
        assert event.change_type == "modified"
        assert event.data["key"] == "test.key"
        assert event.data["old_value"] == "old"
        assert event.data["new_value"] == "new"
        assert event.data["change_type"] == "modified"

    def test_change_type_added(self):
        """测试添加类型变更"""
        event = ConfigChangeEvent("test.key", None, "new")
        assert event.change_type == "added"
        assert event.data["change_type"] == "added"

    def test_change_type_deleted(self):
        """测试删除类型变更"""
        event = ConfigChangeEvent("test.key", "old", None)
        assert event.change_type == "deleted"
        assert event.data["change_type"] == "deleted"

    def test_change_type_unchanged(self):
        """测试未变更类型"""
        event = ConfigChangeEvent("test.key", "value", "value")
        assert event.change_type == "unchanged"
        assert event.data["change_type"] == "unchanged"

    def test_init_with_timestamp(self):
        """测试带时间戳初始化"""
        custom_timestamp = 1234567890.0
        event = ConfigChangeEvent("test.key", "old", "new", custom_timestamp)

        assert event.timestamp == custom_timestamp

    def test_init_with_source(self):
        """测试带源初始化"""
        event = ConfigChangeEvent("test.key", "old", "new", "custom_source")

        assert event.source == "custom_source"


class TestConfigLoadEvent:
    """测试配置加载事件"""

    def test_init_minimal(self):
        """测试最小化初始化"""
        event = ConfigLoadEvent()

        assert event.event_type == "config_loaded"
        assert event.success is True
        assert event.error_message == ""
        assert event.file_path == ""
        assert event.format == "unknown"

    def test_init_with_file_path(self):
        """测试带文件路径初始化"""
        event = ConfigLoadEvent(file_path="/path/to/config.json")

        assert event.file_path == "/path/to/config.json"
        assert event.format == "json"
        assert event.data["source_path"] == "/path/to/config.json"
        assert event.data["file_path"] == "/path/to/config.json"

    def test_determine_format_json(self):
        """测试JSON格式识别"""
        assert ConfigLoadEvent._determine_format("config.json") == "json"
        assert ConfigLoadEvent._determine_format("config.JSON") == "json"

    def test_determine_format_yaml(self):
        """测试YAML格式识别"""
        assert ConfigLoadEvent._determine_format("config.yaml") == "yaml"
        assert ConfigLoadEvent._determine_format("config.yml") == "yaml"
        assert ConfigLoadEvent._determine_format("config.YAML") == "yaml"

    def test_determine_format_toml(self):
        """测试TOML格式识别"""
        assert ConfigLoadEvent._determine_format("config.toml") == "toml"

    def test_determine_format_ini(self):
        """测试INI格式识别"""
        assert ConfigLoadEvent._determine_format("config.ini") == "ini"
        assert ConfigLoadEvent._determine_format("config.cfg") == "ini"
        assert ConfigLoadEvent._determine_format("config.conf") == "ini"

    def test_determine_format_unknown(self):
        """测试未知格式"""
        assert ConfigLoadEvent._determine_format("config.unknown") == "unknown"
        assert ConfigLoadEvent._determine_format("config") == "unknown"

    def test_init_with_success_failure(self):
        """测试成功/失败状态"""
        # 成功加载
        event = ConfigLoadEvent(success=True, file_path="config.json")
        assert event.success is True
        assert event.data["success"] is True

        # 失败加载
        event = ConfigLoadEvent(success=False, error_message="Load failed", file_path="config.json")
        assert event.success is False
        assert event.error_message == "Load failed"
        assert event.data["success"] is False
        assert event.data["error_message"] == "Load failed"

    def test_legacy_constructor_support(self):
        """测试向后兼容的构造函数"""
        # 旧签名：ConfigLoadEvent(source_type, file_path, success, error_message, data)
        event = ConfigLoadEvent("json", "config.json", True, "no error", {"extra": "data"})

        assert event.format == "json"
        assert event.file_path == "config.json"
        assert event.success is True
        assert event.error_message == "no error"
        assert event.data["data"]["extra"] == "data"  # 数据在event_payload的data字段中

    def test_init_with_kwargs(self):
        """测试关键字参数初始化"""
        event = ConfigLoadEvent(
            source="json",
            file_path="config.json",
            success=True,
            error_message="test",
            data={"custom": "data"}
        )

        assert event.format == "json"
        assert event.file_path == "config.json"
        assert event.success is True
        assert event.error_message == "test"
        assert event.data["data"]["custom"] == "data"  # 数据在event_payload的data字段中


class TestConfigValidationEvent:
    """测试配置验证事件"""

    def test_init_passed(self):
        """测试验证通过"""
        event = ConfigValidationEvent("test.key", True)

        assert event.event_type == "config_validated"
        assert event.config_key == "test.key"
        assert event.passed is True
        assert event.errors == []
        assert event.data["config_key"] == "test.key"
        assert event.data["is_valid"] is True  # 兼容性字段
        assert event.data["passed"] is True
        assert event.data["validation_errors"] == []
        assert event.data["errors"] == []

    def test_init_failed_with_errors(self):
        """测试验证失败带错误"""
        errors = ["Invalid format", "Missing required field"]
        event = ConfigValidationEvent("test.key", False, errors)

        assert event.passed is False
        assert event.errors == errors
        assert event.data["is_valid"] is False
        assert event.data["validation_errors"] == errors
        assert event.data["errors"] == errors


class TestConfigReloadEvent:
    """测试配置重载事件"""

    def test_init_success(self):
        """测试重载成功"""
        changed_keys = ["key1", "key2"]
        event = ConfigReloadEvent("manual", True, changed_keys)

        assert event.event_type == "config_reloaded"
        assert event.data["trigger"] == "manual"
        assert event.data["success"] is True
        assert event.data["changed_keys"] == changed_keys

    def test_init_failure(self):
        """测试重载失败"""
        event = ConfigReloadEvent("auto", False)

        assert event.data["trigger"] == "auto"
        assert event.data["success"] is False
        assert event.data["changed_keys"] == []


class TestConfigBackupEvent:
    """测试配置备份事件"""

    def test_init_success_with_size(self):
        """测试备份成功带大小"""
        event = ConfigBackupEvent("/path/backup.json", True, 1024)

        assert event.event_type == "config_backed_up"
        assert event.data["backup_path"] == "/path/backup.json"
        assert event.data["success"] is True
        assert event.data["backup_size"] == 1024

    def test_init_failure(self):
        """测试备份失败"""
        event = ConfigBackupEvent("/path/backup.json", False)

        assert event.data["success"] is False
        assert event.data["backup_size"] is None


class TestConfigErrorEvent:
    """测试配置错误事件"""

    def test_init_with_error_type_and_message(self):
        """测试错误类型和消息初始化"""
        context = {"operation": "save", "file": "config.json"}
        event = ConfigErrorEvent("validation_error", "Invalid host", context)

        assert event.event_type == "config_error"
        assert event.operation == "save"  # 从context获取
        assert event.error_type == "validation_error"
        assert event.error_message == "Invalid host"
        assert event.context == context
        assert event.data["operation"] == "save"
        assert event.data["error_type"] == "validation_error"
        assert event.data["error_message"] == "Invalid host"

    def test_init_with_operation_and_exception(self):
        """测试操作和异常初始化"""
        exception = ValueError("Invalid value")
        context = {"file": "config.json"}
        event = ConfigErrorEvent("load", exception, context)

        assert event.operation == "load"
        assert event.error_type == "ValueError"
        assert event.error_message == "ValueError: Invalid value"
        assert event.context == context

    def test_init_with_none_exception(self):
        """测试None异常情况"""
        event = ConfigErrorEvent("save", None)

        assert event.error_type == "Unknown"
        assert event.error_message == "Unknown error"
        assert event.operation == "save"
