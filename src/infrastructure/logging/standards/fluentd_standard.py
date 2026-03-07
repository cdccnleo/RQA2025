"""
fluentd_standard 模块

提供 fluentd_standard 相关功能和接口。
"""

import json

import msgpack

from .base_standard import BaseStandardFormat, StandardFormatType, StandardLogEntry
from typing import Any, Dict, List, Union
"""
RQA2025 基础设施层 - Fluentd 标准格式

实现 Fluentd 的日志格式。
支持 Fluentd Forward Protocol 格式。
"""

try:
    HAS_MSGPACK = True
except ImportError:
    HAS_MSGPACK = False


class FluentdStandardFormat(BaseStandardFormat):
    """Fluentd 标准格式实现"""

    def __init__(self, tag_prefix: str = "rqa2025"):
        super().__init__(StandardFormatType.FLUENTD)
        self.tag_prefix = tag_prefix

    def format_log_entry(self, entry: Union[StandardLogEntry, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        格式化日志条目为Fluentd格式

        Args:
            entry: 标准日志条目或字典

        Returns:
            Fluentd格式的字典
        """
        # 如果是字典，转换为StandardLogEntry对象
        if isinstance(entry, dict):
            entry = self._dict_to_standard_entry(entry)

        tag = self._create_fluentd_tag(entry)
        record = self._create_fluentd_record(entry)

        return {
            "tag": tag,
            "timestamp": self.timestamp_to_unix_ns(entry.timestamp),
            "record": record
        }

    def _create_fluentd_tag(self, entry: StandardLogEntry) -> str:
        """
        创建Fluentd标签

        Args:
            entry: 标准日志条目

        Returns:
            Fluentd标签字符串
        """

        return f"{self.tag_prefix}.{entry.category.value}.{entry.service or 'infrastructure'}"

    def _create_fluentd_record(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        创建Fluentd记录

        Args:
            entry: 标准日志条目

        Returns:
            Fluentd记录字典
        """
        record = self._create_base_record(entry)
        self._add_metadata_to_record(record, entry)
        self._add_optional_fields_to_record(record, entry)



        return json.dumps(record, ensure_ascii=False)

    def _create_base_record(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        创建基础记录

        Args:
            entry: 标准日志条目

        Returns:
            基础记录字典
        """
        return {
            "timestamp": self.timestamp_to_iso(entry.timestamp),
            "level": self.convert_log_level(entry.level),
            "logger_name": entry.extra_fields.get("logger_name") if hasattr(entry, "extra_fields") and entry.extra_fields else f"{entry.category.value}.{entry.service or 'rqa2025'}",
            "message": entry.message,
            "category": entry.category.value,
            "service": entry.service or "infrastructure",
            "host": entry.host,
            "environment": entry.environment,
            "source": entry.source or "rqa2025-logging",
            "trace_id": entry.trace_id,
            "span_id": entry.span_id,
            "user_id": entry.user_id,
            "session_id": entry.session_id,
            "request_id": entry.request_id,
            "correlation_id": entry.correlation_id,
            "log_level": entry.level.value.upper(),
            "log_category": entry.category.value
        }

    def _add_metadata_to_record(self, record: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加元数据到记录

        Args:
            record: Fluentd记录字典
            entry: 标准日志条目
        """
        if entry.metadata:
            record["metadata"] = entry.metadata

    def _add_optional_fields_to_record(self, record: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加可选字段到记录

        Args:
            record: Fluentd记录字典
            entry: 标准日志条目
        """
        if entry.tags:
            record["tags"] = entry.tags

        if entry.extra_fields:
            record.update(entry.extra_fields)

    def get_content_type(self) -> str:
        """获取内容类型"""
        return "application/x-msgpack"  # Fluentd通常使用MessagePack

    def supports_batch(self) -> bool:
        """Fluentd支持批量"""

        return True

    def format_batch(self, entries: List[StandardLogEntry]) -> Union[str, List[Dict[str, Any]]]:
        """
        批量格式化Fluentd日志条目

        Args:
            entries: 日志条目列表

        Returns:
            Fluentd批量事件列表
        """
        return [self.format_log_entry(entry) for entry in entries]

    def create_forward_payload(self, entries: List[StandardLogEntry]) -> bytes:
        """
        创建Fluentd Forward协议负载

        Args:
            entries: 日志条目列表

        Returns:
            MessagePack编码的字节数据
        """
        if not HAS_MSGPACK:
            raise ImportError(
                "msgpack is required for Fluentd Forward protocol. Install with: pip install msgpack")

        # Fluentd Forward协议格式
        forward_entries = []

        for entry in entries:
            formatted = self.format_log_entry(entry)
            forward_entries.append([
                formatted["tag"],
                formatted["timestamp"],
                formatted["record"]
            ])

        # 返回MessagePack编码的数据

        import msgpack
        return msgpack.packb(forward_entries)

    def create_json_payload(self, entries: List[StandardLogEntry]) -> str:
        """
        创建JSON格式的Fluentd负载（用于HTTP输入）

        Args:
            entries: 日志条目列表

        Returns:
            JSON字符串
        """
        forward_entries = []

        for entry in entries:
            formatted = self.format_log_entry(entry)
            forward_entries.append({
                "tag": formatted["tag"],
                "timestamp": formatted["timestamp"],
                "record": formatted["record"]
            })



        return json.dumps(forward_entries, ensure_ascii=False, separators=(',', ':'))
