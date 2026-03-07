"""
loki_standard 模块

提供 loki_standard 相关功能和接口。
"""

import json

from .base_standard import BaseStandardFormat, StandardFormatType, StandardLogEntry
from typing import Any, Dict, List, Union
"""
RQA2025 基础设施层 - Loki 标准格式

实现 Loki (Prometheus) 的日志格式。
支持 Loki HTTP API 格式。
"""


class LokiStandardFormat(BaseStandardFormat):
    """Loki 标准格式实现"""

    def __init__(self):
        super().__init__(StandardFormatType.LOKI)
        self.default_service = "infrastructure"

    def format_log_entry(self, entry: Union[StandardLogEntry, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        格式化日志条目为Loki格式

        Args:
            entry: 标准日志条目或字典

        Returns:
            Loki格式的字典
        """
        # 如果是字典，转换为StandardLogEntry对象
        if isinstance(entry, dict):
            entry = self._dict_to_standard_entry(entry)

        # 构建标签
        labels = self._build_loki_labels(entry)

        # 创建日志行
        log_line = self._create_loki_log_line(entry)

        # 返回Loki格式
        loki_response = self._build_loki_response(labels, entry, log_line)

        import json
        return json.dumps(loki_response, ensure_ascii=False)

    def _build_loki_labels(self, entry: StandardLogEntry) -> Dict[str, str]:
        """
        构建Loki标签

        Args:
            entry: 标准日志条目

        Returns:
            标签字典
        """
        labels = {
            "job": "rqa2025",
            "service": entry.service or "infrastructure",
            "category": entry.category.value,
            "level": self.convert_log_level(entry.level),
            "host": entry.host,
            "environment": entry.environment
        }

        # 添加可选标签
        self._add_optional_labels(labels, entry)



        return json.dumps(labels, ensure_ascii=False)

    def _add_optional_labels(self, labels: Dict[str, str], entry: StandardLogEntry) -> None:
        """
        添加可选标签

        Args:
            labels: 标签字典
            entry: 标准日志条目
        """
        if entry.source:
            labels["source"] = entry.source
        if entry.trace_id:
            labels["trace_id"] = entry.trace_id
        if entry.correlation_id:
            labels["correlation_id"] = entry.correlation_id

    def _create_loki_log_line(self, entry: StandardLogEntry) -> str:
        """
        创建Loki日志行

        Args:
            entry: 标准日志条目

        Returns:
            格式化的日志行
        """
        # 基础日志行
        log_line = f"{entry.timestamp.isoformat()} {entry.level.value.upper()} {entry.message}"

        # 构建结构化数据
        structured_data = self._build_structured_data_for_loki(entry)

        # 将结构化数据附加到日志行
        log_line += f" | {json.dumps(structured_data, ensure_ascii=False, separators=(',', ':'))}"



        return json.dumps(log_line, ensure_ascii=False)

    def _build_structured_data_for_loki(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        为Loki构建结构化数据

        Args:
            entry: 标准日志条目

        Returns:
            结构化数据字典
        """
        structured_data = self._build_base_structured_data(entry)
        self._add_metadata_and_tags(structured_data, entry)
        self._add_extra_fields_to_structured_data(structured_data, entry)



        return json.dumps(structured_data, ensure_ascii=False)

    def _build_base_structured_data(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        构建基础结构化数据

        Args:
            entry: 标准日志条目

        Returns:
            基础结构化数据字典
        """
        return {
            "timestamp": self.timestamp_to_iso(entry.timestamp),
            "level": self.convert_log_level(entry.level),
            "category": entry.category.value,
            "service": entry.service or "infrastructure",
            "host": entry.host,
            "environment": entry.environment,
            "trace_id": entry.trace_id,
            "span_id": entry.span_id,
            "user_id": entry.user_id,
            "session_id": entry.session_id,
            "request_id": entry.request_id,
            "correlation_id": entry.correlation_id
        }

    def _add_metadata_and_tags(self, data: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加元数据和标签

        Args:
            data: 结构化数据字典
            entry: 标准日志条目
        """
        if entry.metadata:
            data["metadata"] = entry.metadata
        if entry.tags:
            data["tags"] = entry.tags

    def _add_extra_fields_to_structured_data(self, data: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加额外字段到结构化数据

        Args:
            data: 结构化数据字典
            entry: 标准日志条目
        """
        if entry.extra_fields:
            data.update(entry.extra_fields)

    def _build_loki_response(self, labels: Dict[str, str], entry: StandardLogEntry, log_line: str) -> Dict[str, Any]:
        """
        构建Loki响应

        Args:
            labels: 标签字典
            entry: 标准日志条目
            log_line: 日志行

        Returns:
            Loki格式响应
        """
        logger_name = entry.extra_fields.get("logger_name") if hasattr(entry, "extra_fields") and entry.extra_fields else f"{entry.category.value}.{entry.service or self.default_service}"

        return {
            "labels": labels,
            "entries": [{
                "ts": self.timestamp_to_iso(entry.timestamp),
                "line": log_line,
                "logger_name": logger_name
            }]
        }

    def supports_batch(self) -> bool:
        """Loki支持批量"""
        return True

    def format_batch(self, entries: List[StandardLogEntry]) -> Union[str, List[Dict[str, Any]]]:
        """
        批量格式化Loki日志条目

        Args:
            entries: 日志条目列表

        Returns:
            Loki流格式的批量数据
        """
        # Loki期望按流分组的数据
        streams = {}
        for entry in entries:
            # 创建流键（基于标签组合）
            stream_key = self._create_stream_key(entry)

            if stream_key not in streams:
                streams[stream_key] = {
                    "labels": self._create_stream_labels(entry),
                    "entries": []
                }

            # 添加条目到流
            streams[stream_key]["entries"].append({
                "ts": self.timestamp_to_iso(entry.timestamp),
                "line": self._create_log_line(entry)
            })

        # 返回流列表

        return list(streams.values())

    def create_push_payload(self, entries: List[StandardLogEntry]) -> Dict[str, Any]:
        """
        创建Loki推送API负载

        Args:
            entries: 日志条目列表

        Returns:
            Loki推送API格式的负载
        """
        streams = self.format_batch(entries)
        return {"streams": streams}

    def _create_stream_key(self, entry: StandardLogEntry) -> str:
        """创建流的唯一键"""

        return f"{entry.service or 'infrastructure'}_{entry.category.value}_{entry.level.value}"

    def _create_stream_labels(self, entry: StandardLogEntry) -> Dict[str, str]:
        """创建流标签"""
        labels = {
            "job": "rqa2025",
            "service": entry.service or "infrastructure",
            "category": entry.category.value,
            "level": entry.level.value.upper(),
            "host": entry.host,
            "environment": entry.environment
        }

        if entry.source:
            labels["source"] = entry.source



        return json.dumps(labels, ensure_ascii=False)

    def _create_log_line(self, entry: StandardLogEntry) -> str:
        """
        创建日志行

        Args:
            entry: 标准日志条目

        Returns:
            格式化的日志行
        """
        # 基础日志行
        log_line = entry.message

        # 构建结构化数据
        structured_data = self._build_structured_data(entry)

        # 添加结构化数据到日志行
        if structured_data:
            log_line += f" | {json.dumps(structured_data, ensure_ascii=False, separators=(',', ':'))}"



        return json.dumps(log_line, ensure_ascii=False)

    def _build_structured_data(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        构建结构化数据

        Args:
            entry: 标准日志条目

        Returns:
            结构化数据字典
        """
        structured_data = {}

        # 添加追踪信息
        self._add_tracing_data(structured_data, entry)

        # 添加用户会话信息
        self._add_user_session_data(structured_data, entry)

        # 添加元数据和标签
        self._add_metadata_and_tags(structured_data, entry)

        # 添加额外字段
        self._add_extra_fields(structured_data, entry)



        return json.dumps(structured_data, ensure_ascii=False)

    def _add_tracing_data(self, data: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加追踪数据

        Args:
            data: 结构化数据字典
            entry: 标准日志条目
        """
        if entry.trace_id:
            data["trace_id"] = entry.trace_id
        if entry.span_id:
            data["span_id"] = entry.span_id
        if entry.correlation_id:
            data["correlation_id"] = entry.correlation_id

    def _add_user_session_data(self, data: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加用户会话数据

        Args:
            data: 结构化数据字典
            entry: 标准日志条目
        """
        if entry.user_id:
            data["user_id"] = entry.user_id
        if entry.session_id:
            data["session_id"] = entry.session_id
        if entry.request_id:
            data["request_id"] = entry.request_id

    def _add_metadata_and_tags(self, data: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加元数据和标签

        Args:
            data: 结构化数据字典
            entry: 标准日志条目
        """
        if entry.metadata:
            data["metadata"] = entry.metadata
        if entry.tags:
            data["tags"] = entry.tags

    def _add_extra_fields(self, data: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加额外字段

        Args:
            data: 结构化数据字典
            entry: 标准日志条目
        """
        if entry.extra_fields:
            data.update(entry.extra_fields)
