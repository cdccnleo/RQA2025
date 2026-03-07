"""
datadog_standard 模块

提供 datadog_standard 相关功能和接口。
"""

import json

from ..core.interfaces import LogLevel
from .base_standard import BaseStandardFormat, StandardFormatType, StandardLogEntry
from typing import Any, Dict, List, Union
"""
RQA2025 基础设施层 - Datadog 标准格式

实现 Datadog 的日志格式。
支持 Datadog Log Management API 格式。
"""


class DatadogStandardFormat(BaseStandardFormat):
    """Datadog 标准格式实现"""

    def __init__(self, service: str = "rqa2025", source: str = "python"):
        super().__init__(StandardFormatType.DATADOG)
        self.default_service = service
        self.default_source = source

    def format_log_entry(self, entry: Union[StandardLogEntry, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        格式化日志条目为Datadog格式

        Args:
            entry: 标准日志条目或字典

        Returns:
            Datadog格式的字典
        """
        # 如果是字典，转换为StandardLogEntry对象
        if isinstance(entry, dict):
            entry = self._dict_to_standard_entry(entry)

        # 创建基础Datadog条目
        dd_entry = self._create_base_datadog_entry(entry)

        # 添加可选字段
        self._add_metadata_fields(dd_entry, entry)
        self._add_extra_fields(dd_entry, entry)
        self._add_tracing_fields(dd_entry, entry)
        self._add_user_fields(dd_entry, entry)

        import json
        return json.dumps(dd_entry, ensure_ascii=False)

    def _create_base_datadog_entry(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """创建基础Datadog条目"""
        # 从extra_fields获取原始logger_name，如果没有则构造一个
        logger_name = entry.extra_fields.get('logger_name') if hasattr(entry, 'extra_fields') and entry.extra_fields else f"{entry.category.value}.{entry.service or self.default_service}"

        return {
            "timestamp": self.timestamp_to_unix_ns(entry.timestamp),
            "level": self.convert_log_level(entry.level),
            "logger_name": logger_name,
            "message": entry.message,
            "service": entry.service or self.default_service,
            "source": self.default_source,
            "host": entry.host,
            "ddsource": entry.source or "rqa2025-logging",
            "ddtags": self._create_datadog_tags(entry),
            "logger": {
                "name": f"{entry.category.value}.{entry.service or self.default_service}",
                "thread_name": entry.trace_id,
                "method_name": entry.request_id
            },
            "status": self._map_level_to_status(entry.level),
            "env": entry.environment,
            "version": "1.0.0"
        }

    def _add_metadata_fields(self, dd_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加元数据字段"""
        if entry.metadata:
            for key, value in entry.metadata.items():
                dd_entry[f"metadata.{key}"] = value

    def _add_extra_fields(self, dd_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加额外字段"""
        if entry.extra_fields:
            dd_entry.update(entry.extra_fields)

    def _add_tracing_fields(self, dd_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加分布式追踪字段"""
        if entry.trace_id:
            dd_entry["dd.trace_id"] = entry.trace_id
        if entry.span_id:
            dd_entry["dd.span_id"] = entry.span_id

    def _add_user_fields(self, dd_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加用户和会话字段"""
        if entry.user_id:
            dd_entry["usr.id"] = entry.user_id
        if entry.session_id:
            dd_entry["usr.session_id"] = entry.session_id

    def supports_batch(self) -> bool:
        """Datadog支持批量"""
        return True

    def format_batch(self, entries: List[StandardLogEntry]) -> Union[str, List[Dict[str, Any]]]:
        """
        批量格式化Datadog日志条目

        Args:
            entries: 日志条目列表

        Returns:
            批量事件列表
        """
        return [self.format_log_entry(entry) for entry in entries]

    def create_batch_payload(self, entries: List[StandardLogEntry]) -> str:
        """
        创建Datadog批量负载

        Args:
            entries: 日志条目列表

        Returns:
            JSON数组字符串
        """
        events = self.format_batch(entries)
        return json.dumps(events, ensure_ascii=False, separators=(',', ':'))

    def _create_datadog_tags(self, entry: StandardLogEntry) -> str:
        """创建Datadog标签字符串"""
        tags = [
            f"env:{entry.environment}",
            f"service:{entry.service or self.default_service}",
            f"category:{entry.category.value}",
            f"source:{entry.source or self.default_source}"
        ]

        if entry.tags:
            tags.extend(entry.tags)

        if entry.correlation_id:
            tags.append(f"correlation_id:{entry.correlation_id}")

        return ",".join(tags)

    def _map_level_to_status(self, level: LogLevel) -> str:
        """映射日志级别到Datadog状态"""
        mapping = {
            LogLevel.DEBUG: "debug",
            LogLevel.INFO: "info",
            LogLevel.WARNING: "warn",
            LogLevel.ERROR: "error",
            LogLevel.CRITICAL: "critical"
        }
        return mapping.get(level, "info")


class DatadogStandard(DatadogStandardFormat):
    """兼容旧测试命名的别名类"""
    pass