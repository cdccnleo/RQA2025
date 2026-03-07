"""
splunk_standard 模块

提供 splunk_standard 相关功能和接口。
"""

import json

from ..core.interfaces import LogLevel, LogCategory
from .base_standard import BaseStandardFormat, StandardFormatType, StandardLogEntry
from typing import Any, Dict, List, Union
"""
RQA2025 基础设施层 - Splunk 标准格式

实现 Splunk 的日志格式。
支持 Splunk HTTP Event Collector (HEC) 格式。
"""


class SplunkStandardFormat(BaseStandardFormat):
    """Splunk 标准格式实现"""

    def __init__(self, sourcetype: str = "rqa2025:json"):
        super().__init__(StandardFormatType.SPLUNK)
        self.sourcetype = sourcetype

    def format_log_entry(self, entry: Union[StandardLogEntry, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        格式化日志条目为Splunk HEC格式

        Args:
            entry: 标准日志条目或字典

        Returns:
            Splunk HEC格式的字典
        """
        # 如果是字典，转换为StandardLogEntry对象
        if isinstance(entry, dict):
            entry = self._dict_to_standard_entry(entry)

        splunk_event = self._create_base_splunk_event(entry)
        self._add_metadata_to_event(splunk_event, entry)
        self._add_optional_fields_to_event(splunk_event, entry)
        self._add_splunk_specific_fields(splunk_event, entry)



        return json.dumps(splunk_event, ensure_ascii=False)

    def _create_base_splunk_event(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        创建基础Splunk事件

        Args:
            entry: 标准日志条目

        Returns:
            基础Splunk事件字典
        """
        return {
            "time": self.timestamp_to_unix_ms(entry.timestamp),
            "host": entry.host or "rqa2025-host",
            "source": entry.source or "rqa2025-logging",
            "sourcetype": self.sourcetype,
            "logger_name": entry.extra_fields.get("logger_name") if hasattr(entry, "extra_fields") and entry.extra_fields else f"{entry.category.value}.{entry.service or 'rqa2025'}",
            "index": f"rqa2025_{entry.category.value}",
            "event": self._create_base_event_data(entry)
        }

    def _create_base_event_data(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        创建基础事件数据

        Args:
            entry: 标准日志条目

        Returns:
            基础事件数据字典
        """
        return {
            "timestamp": self.timestamp_to_iso(entry.timestamp),
            "level": self.convert_log_level(entry.level),
            "message": entry.message,
            "category": entry.category.value,
            "service": entry.service or "infrastructure",
            "environment": entry.environment,
            "trace_id": entry.trace_id,
            "span_id": entry.span_id,
            "user_id": entry.user_id,
            "session_id": entry.session_id,
            "request_id": entry.request_id,
            "correlation_id": entry.correlation_id,
            "log_level": entry.level.value.upper(),
            "log_category": entry.category.value,
        }

    def _add_metadata_to_event(self, splunk_event: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加元数据到事件中

        Args:
            splunk_event: Splunk事件字典
            entry: 标准日志条目
        """
        if entry.metadata:
            splunk_event["event"]["metadata"] = entry.metadata

    def _add_optional_fields_to_event(self, splunk_event: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加可选字段到事件中

        Args:
            splunk_event: Splunk事件字典
            entry: 标准日志条目
        """
        if entry.tags:
            splunk_event["event"]["tags"] = entry.tags

        if entry.extra_fields:
            splunk_event["event"].update(entry.extra_fields)

    def _add_splunk_specific_fields(self, splunk_event: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加Splunk特定字段

        Args:
            splunk_event: Splunk事件字典
            entry: 标准日志条目
        """
        splunk_event["event"].update({
            "splunk_source": entry.source,
            "splunk_sourcetype": self.sourcetype,
            "splunk_index": f"rqa2025_{entry.category.value}",
            "severity": self._map_level_to_severity(entry.level),
            "facility": self._map_category_to_facility(entry.category)
        })

    def supports_batch(self) -> bool:
        """Splunk HEC支持批量"""

        return True

    def format_batch(self, entries: List[StandardLogEntry]) -> Union[str, List[Dict[str, Any]]]:
        """
        批量格式化Splunk日志条目

        Args:
            entries: 日志条目列表

        Returns:
            批量事件列表
        """
        return [self.format_log_entry(entry) for entry in entries]

    def create_hec_payload(self, entries: List[StandardLogEntry]) -> str:
        """
        创建Splunk HEC批量负载

        Args:
            entries: 日志条目列表

        Returns:
            JSON数组字符串
        """
        events = self.format_batch(entries)

        return json.dumps(events, ensure_ascii=False, separators=(',', ':'))

    def _map_level_to_severity(self, level: LogLevel) -> int:
        """映射日志级别到Syslog严重性"""
        mapping = {
            LogLevel.DEBUG: 7,
            LogLevel.INFO: 6,
            LogLevel.WARNING: 4,
            LogLevel.ERROR: 3,
            LogLevel.CRITICAL: 2
        }

        return mapping.get(level, 6)

    def _map_category_to_facility(self, category: LogCategory) -> int:
        """映射日志类别到Syslog设施"""
        mapping = {
            LogCategory.SYSTEM: 0,      # kern
            LogCategory.SECURITY: 4,    # security/auth
            LogCategory.BUSINESS: 16,   # local0
            LogCategory.AUDIT: 17,      # local1
            LogCategory.PERFORMANCE: 18,  # local2
            LogCategory.DATABASE: 19,   # local3
            LogCategory.TRADING: 20,    # local4
            LogCategory.RISK: 21,       # local5
            LogCategory.GENERAL: 23     # local7
        }

        return mapping.get(category, 23)


class SplunkStandard(SplunkStandardFormat):
    """兼容旧测试命名的别名类"""
    pass