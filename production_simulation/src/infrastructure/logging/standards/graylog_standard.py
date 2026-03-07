"""
graylog_standard 模块

提供 graylog_standard 相关功能和接口。
"""

import json

from ..core.interfaces import LogLevel
from .base_standard import BaseStandardFormat, StandardFormatType, StandardLogEntry
from typing import Any, Dict, List, Union
"""
RQA2025 基础设施层 - Graylog 标准格式

实现 Graylog 的日志格式。
支持 GELF (Graylog Extended Log Format) 格式。
"""


class GraylogStandardFormat(BaseStandardFormat):
    """Graylog GELF 标准格式实现"""

    def __init__(self, facility: str = "rqa2025"):
        super().__init__(StandardFormatType.GRAYLOG)
        self.facility = facility

    def format_log_entry(self, entry: Union[StandardLogEntry, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        格式化日志条目为GELF格式

        Args:
            entry: 标准日志条目或字典

        Returns:
            GELF格式的字典
        """
        # 如果是字典，转换为StandardLogEntry对象
        if isinstance(entry, dict):
            entry = self._dict_to_standard_entry(entry)

        # 创建基础GELF条目
        gelf_entry = self._create_base_gelf_entry(entry)

        # 添加可选字段
        self._add_message_fields(gelf_entry, entry)
        self._add_tracing_fields(gelf_entry, entry)
        self._add_user_session_fields(gelf_entry, entry)
        self._add_metadata_fields(gelf_entry, entry)
        self._add_tags_fields(gelf_entry, entry)
        self._add_extra_fields(gelf_entry, entry)



        return json.dumps(gelf_entry, ensure_ascii=False)

    def _create_base_gelf_entry(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """创建基础GELF条目"""
        return {
            "version": "1.1",
            "host": entry.host or "rqa2025-host",
            "short_message": self._get_short_message(entry.message),
            "timestamp": entry.timestamp.timestamp(),
            "level": self._map_level_to_gelf_level(entry.level),
            "logger_name": entry.extra_fields.get("logger_name") if hasattr(entry, "extra_fields") and entry.extra_fields else f"{entry.category.value}.{entry.service or self.facility}",
            "facility": self.facility,
            "_service": entry.service or "infrastructure",
            "_category": entry.category.value,
            "_environment": entry.environment,
            "_source": entry.source or "rqa2025-logging",
            "_log_level": entry.level.value.upper(),
            "_log_category": entry.category.value
        }

    def _get_short_message(self, message: str) -> str:
        """获取短消息（最多200字符）"""

        return json.dumps(message, ensure_ascii=False)[:200] if len(message) > 200 else message

    def _add_message_fields(self, gelf_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加消息相关字段"""
        if len(entry.message) > 200:
            gelf_entry["full_message"] = entry.message

    def _add_tracing_fields(self, gelf_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加分布式追踪字段"""
        if entry.trace_id:
            gelf_entry["_trace_id"] = entry.trace_id
        if entry.span_id:
            gelf_entry["_span_id"] = entry.span_id
        if entry.correlation_id:
            gelf_entry["_correlation_id"] = entry.correlation_id

    def _add_user_session_fields(self, gelf_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加用户和会话字段"""
        if entry.user_id:
            gelf_entry["_user_id"] = entry.user_id
        if entry.session_id:
            gelf_entry["_session_id"] = entry.session_id
        if entry.request_id:
            gelf_entry["_request_id"] = entry.request_id

    def _add_metadata_fields(self, gelf_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加元数据字段"""
        if entry.metadata:
            for key, value in entry.metadata.items():
                gelf_entry[f"_{key}"] = value

    def _add_tags_fields(self, gelf_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加标签字段"""
        if entry.tags:
            gelf_entry["_tags"] = ",".join(entry.tags)

    def _add_extra_fields(self, gelf_entry: Dict[str, Any], entry: StandardLogEntry):
        """添加额外字段"""
        if entry.extra_fields:
            for key, value in entry.extra_fields.items():
                gelf_entry[f"_{key}"] = value

    def supports_batch(self) -> bool:
        """Graylog GELF支持批量"""

        return True

    def format_batch(self, entries: List[StandardLogEntry]) -> Union[str, List[Dict[str, Any]]]:
        """
        批量格式化Graylog日志条目

        Args:
            entries: 日志条目列表

        Returns:
            GELF事件列表
        """
        return [self.format_log_entry(entry) for entry in entries]

    def create_batch_payload(self, entries: List[StandardLogEntry]) -> str:
        """
        创建Graylog批量负载

        Args:
            entries: 日志条目列表

        Returns:
            JSON数组字符串
        """
        events = self.format_batch(entries)

        return json.dumps(events, ensure_ascii=False, separators=(',', ':'))

    def _map_level_to_gelf_level(self, level: LogLevel) -> int:
        """映射日志级别到GELF级别"""
        # GELF级别：0=Emergency, 1=Alert, 2=Critical, 3=Error, 4=Warning, 5=Notice, 6=Info, 7=Debug
        mapping = {
            LogLevel.CRITICAL: 2,
            LogLevel.ERROR: 3,
            LogLevel.WARNING: 4,
            LogLevel.INFO: 6,
            LogLevel.DEBUG: 7
        }

        return mapping.get(level, 6)
