"""
newrelic_standard 模块

提供 newrelic_standard 相关功能和接口。
"""

import json

from .base_standard import BaseStandardFormat, StandardFormatType, StandardLogEntry
from typing import Any, Dict, List, Union
"""
RQA2025 基础设施层 - New Relic 标准格式

实现 New Relic 的日志格式。
支持 New Relic Logs API 格式。
"""


class NewRelicStandardFormat(BaseStandardFormat):
    """New Relic 标准格式实现"""

    def __init__(self, license_key: str = None):
        super().__init__(StandardFormatType.NEW_RELIC)
        self.license_key = license_key

    def format_log_entry(self, entry: Union[StandardLogEntry, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        格式化日志条目为New Relic格式

        Args:
            entry: 标准日志条目或字典

        Returns:
            New Relic格式的字典
        """
        # 如果是字典，转换为StandardLogEntry对象
        if isinstance(entry, dict):
            entry = self._dict_to_standard_entry(entry)

        nr_entry = self._create_base_nr_entry(entry)

        self._add_metadata_to_attributes(nr_entry, entry)
        self._add_optional_fields_to_attributes(nr_entry, entry)



        return json.dumps(nr_entry, ensure_ascii=False)

    def _create_base_nr_entry(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        创建New Relic条目的基础字段

        Args:
            entry: 标准日志条目

        Returns:
            包含基础字段和属性的字典
        """
        return {
            "timestamp": self.timestamp_to_unix_ms(entry.timestamp),
            "level": self.convert_log_level(entry.level),
            "logger_name": entry.extra_fields.get("logger_name") if hasattr(entry, "extra_fields") and entry.extra_fields else f"{entry.category.value}.{entry.service or 'rqa2025'}",
            "message": entry.message,
            "service.name": entry.service or "rqa2025",
            "service.version": "1.0.0",
            "host.name": entry.host,
            "entity.name": f"{entry.service or 'rqa2025'}.{entry.environment}",
            "entity.type": "SERVICE",
            "logtype": entry.category.value,
            "attributes": self._create_base_attributes(entry)
        }

    def _create_base_attributes(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        创建基础属性字典

        Args:
            entry: 标准日志条目

        Returns:
            基础属性字典
        """
        return {
            "category": entry.category.value,
            "environment": entry.environment,
            "source": entry.source or "rqa2025-logging",
            "trace.id": entry.trace_id,
            "span.id": entry.span_id,
            "user.id": entry.user_id,
            "session.id": entry.session_id,
            "request.id": entry.request_id,
            "correlation.id": entry.correlation_id,
            "log.level": entry.level.value.upper(),
            "log.category": entry.category.value
        }

    def _add_metadata_to_attributes(self, nr_entry: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加元数据到属性中

        Args:
            nr_entry: New Relic条目字典
            entry: 标准日志条目
        """
        if entry.metadata:
            for key, value in entry.metadata.items():
                nr_entry["attributes"][f"metadata.{key}"] = value

    def _add_optional_fields_to_attributes(self, nr_entry: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加可选字段到属性中

        Args:
            nr_entry: New Relic条目字典
            entry: 标准日志条目
        """
        if entry.tags:
            nr_entry["attributes"]["tags"] = entry.tags

        if entry.extra_fields:
            nr_entry["attributes"].update(entry.extra_fields)

    def supports_batch(self) -> bool:
        """New Relic支持批量"""

        return True

    def format_batch(self, entries: List[StandardLogEntry]) -> Union[str, List[Dict[str, Any]]]:
        """
        批量格式化New Relic日志条目

        Args:
            entries: 日志条目列表

        Returns:
            批量事件列表
        """
        return [self.format_log_entry(entry) for entry in entries]

    def create_batch_payload(self, entries: List[StandardLogEntry]) -> str:
        """
        创建New Relic批量负载

        Args:
            entries: 日志条目列表

        Returns:
            JSON数组字符串
        """
        events = self.format_batch(entries)

        return json.dumps(events, ensure_ascii=False, separators=(',', ':'))

    def create_log_payload(self, entries: List[StandardLogEntry]) -> Dict[str, Any]:
        """
        创建完整的New Relic日志API负载

        Args:
            entries: 日志条目列表

        Returns:
            完整的API负载字典
        """
        return {
            "logs": self.format_batch(entries),
            "common": {
                "attributes": {
                    "service.name": "rqa2025",
                    "service.version": "1.0.0",
                    "environment": "production",
                    "instrumentation.provider": "rqa2025-logging"
                }
            }
        }
