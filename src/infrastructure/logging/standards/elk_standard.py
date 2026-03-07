"""
elk_standard 模块

提供 elk_standard 相关功能和接口。
"""

import json

from .base_standard import BaseStandardFormat, StandardFormatType, StandardLogEntry
from typing import Any, Dict, List, Union
"""
RQA2025 基础设施层 - ELK Stack 标准格式

实现 Elasticsearch, Logstash, Kibana (ELK) 堆栈的日志格式。
支持 JSON 格式输出，兼容 Elasticsearch 索引和 Kibana 可视化。
"""


class ELKStandardFormat(BaseStandardFormat):
    """ELK Stack 标准格式实现"""

    def __init__(self):
        super().__init__(StandardFormatType.ELK)
        self.default_service = "infrastructure"

    def format_log_entry(self, entry: Union[StandardLogEntry, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        格式化日志条目为ELK兼容的JSON格式

        Args:
            entry: 标准日志条目或字典

        Returns:
            ELK格式的JSON字符串或字典
        """
        # 如果是字典，转换为StandardLogEntry对象
        if isinstance(entry, dict):
            entry = self._dict_to_standard_entry(entry)

        elk_entry = self._create_base_elk_entry(entry)

        self._add_metadata_fields(elk_entry, entry)
        self._add_optional_fields(elk_entry, entry)
        self._add_elk_specific_fields(elk_entry, entry)

        import json
        return json.dumps(elk_entry, ensure_ascii=False)

    def _create_base_elk_entry(self, entry: StandardLogEntry) -> Dict[str, Any]:
        """
        创建ELK条目的基础字段

        Args:
            entry: 标准日志条目

        Returns:
            包含基础字段的字典
        """
        return {
            "@timestamp": self.timestamp_to_iso(entry.timestamp),
            "level": self.convert_log_level(entry.level),
            "logger_name": entry.extra_fields.get("logger_name") if hasattr(entry, "extra_fields") and entry.extra_fields else f"{entry.category.value}.{entry.service or self.default_service}",
            "message": entry.message,
            "category": entry.category.value,
            "source": entry.source or "rqa2025",
            "host": entry.host,
            "service": entry.service or "infrastructure",
            "environment": entry.environment,
            "trace.id": entry.trace_id,
            "span.id": entry.span_id,
            "user.id": entry.user_id,
            "session.id": entry.session_id,
            "request.id": entry.request_id,
            "correlation.id": entry.correlation_id,
        }

    def _add_metadata_fields(self, elk_entry: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加元数据字段

        Args:
            elk_entry: ELK条目字典
            entry: 标准日志条目
        """
        if entry.metadata:
            for key, value in entry.metadata.items():
                elk_entry[f"metadata.{key}"] = value

    def _add_optional_fields(self, elk_entry: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加可选字段（标签和额外字段）

        Args:
            elk_entry: ELK条目字典
            entry: 标准日志条目
        """
        if entry.tags:
            elk_entry["tags"] = entry.tags

        if entry.extra_fields:
            elk_entry.update(entry.extra_fields)

    def _add_elk_specific_fields(self, elk_entry: Dict[str, Any], entry: StandardLogEntry) -> None:
        """
        添加ELK特定的字段

        Args:
            elk_entry: ELK条目字典
            entry: 标准日志条目
        """
        elk_entry.update({
            "log.level": entry.level.value.upper(),
            "log.logger": f"{entry.category.value}.{entry.service}",
            "event.dataset": f"{entry.category.value}.{entry.service}",
            "ecs.version": "1.12.0",
            "agent": {
                "name": "rqa2025-logging",
                "type": "infrastructure",
                "version": "1.0.0"
            },
            "host.name": entry.host,
            "service.name": entry.service or "infrastructure",
            "service.environment": entry.environment
        })

    def _serialize_elk_entry(self, elk_entry: Dict[str, Any]) -> str:
        """
        序列化ELK条目为JSON字符串

        Args:
            elk_entry: ELK条目字典

        Returns:
            JSON格式的字符串
        """

        return json.dumps(elk_entry, ensure_ascii=False, separators=(',', ':'))

    def supports_batch(self) -> bool:
        """ELK支持批量操作"""

        return True

    def format_batch(self, entries: List[StandardLogEntry]) -> Union[str, List[Dict[str, Any]]]:
        """
        批量格式化ELK日志条目

        Args:
            entries: 日志条目列表

        Returns:
            NDJSON格式的批量数据
        """
        if not entries:
            return ""

        # ELK批量API格式：每行一个JSON文档
        batch_lines = []
        for entry in entries:
            formatted = self.format_log_entry(entry)
            if isinstance(formatted, str):
                batch_lines.append(formatted)
            else:
                batch_lines.append(json.dumps(formatted, ensure_ascii=False, separators=(',', ':')))

        return "\n".join(batch_lines) + "\n"

    def create_bulk_index_request(self, entries: List[StandardLogEntry], index_name: str = "logs-rqa2025") -> str:
        """
        创建Elasticsearch批量索引请求

        Args:
            entries: 日志条目列表
            index_name: 索引名称

        Returns:
            Elasticsearch批量请求体
        """
        bulk_lines = []

        for entry in entries:
            # 索引操作元数据
            meta = {
                "index": {
                    "_index": f"{index_name}-{entry.timestamp.strftime('%Y.%m.%d')}",
                    "_id": entry.trace_id or str(hash(entry.message + str(entry.timestamp)))
                }
            }
            bulk_lines.append(json.dumps(meta, separators=(',', ':')))

            # 文档数据
            doc = self.format_log_entry(entry)
            if isinstance(doc, str):
                doc = json.loads(doc)
            bulk_lines.append(json.dumps(doc, ensure_ascii=False, separators=(',', ':')))

        return "\n".join(bulk_lines) + "\n"
