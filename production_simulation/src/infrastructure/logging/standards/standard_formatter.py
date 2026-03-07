
from ..core.interfaces import LogLevel, LogCategory
from .base_standard import StandardFormatType, StandardLogEntry
from .datadog_standard import DatadogStandardFormat
from .elk_standard import ELKStandardFormat
from .fluentd_standard import FluentdStandardFormat
from .graylog_standard import GraylogStandardFormat
from .loki_standard import LokiStandardFormat
from .newrelic_standard import NewRelicStandardFormat
from .splunk_standard import SplunkStandardFormat
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
"""
RQA2025 基础设施层 - 标准格式转换器

提供统一的日志格式转换接口，支持多种日志分析平台的标准格式。
"""


class StandardFormatter:
    """标准格式转换器"""

    def __init__(self):
        self._formatters = {
            StandardFormatType.ELK: ELKStandardFormat(),
            StandardFormatType.SPLUNK: SplunkStandardFormat(),
            StandardFormatType.DATADOG: DatadogStandardFormat(),
            StandardFormatType.NEW_RELIC: NewRelicStandardFormat(),
            StandardFormatType.LOKI: LokiStandardFormat(),
            StandardFormatType.GRAYLOG: GraylogStandardFormat(),
            StandardFormatType.FLUENTD: FluentdStandardFormat(),
        }

    def format_log_entry(
        self,
        entry: StandardLogEntry,
        format_type: StandardFormatType
    ) -> Union[str, Dict[str, Any]]:
        """
        格式化单个日志条目

        Args:
            entry: 标准日志条目
            format_type: 目标格式类型

        Returns:
            格式化后的日志数据
        """
        formatter = self._formatters.get(format_type)
        if not formatter:
            raise ValueError(f"不支持的格式类型: {format_type}")

        if not formatter.validate_entry(entry):
            raise ValueError(f"无效的日志条目: {entry}")

        return formatter.format_log_entry(entry)

    def format_batch(
        self,
        entries: List[StandardLogEntry],
        format_type: StandardFormatType
    ) -> Union[str, List[Dict[str, Any]]]:
        """
        批量格式化日志条目

        Args:
            entries: 日志条目列表
            format_type: 目标格式类型

        Returns:
            批量格式化结果
        """
        formatter = self._formatters.get(format_type)
        if not formatter:
            raise ValueError(f"不支持的格式类型: {format_type}")

        # 验证所有条目
        for entry in entries:
            if not formatter.validate_entry(entry):
                raise ValueError(f"无效的日志条目: {entry}")

        return formatter.format_batch(entries)

    def get_content_type(self, format_type: StandardFormatType) -> str:
        """
        获取指定格式的内容类型

        Args:
            format_type: 格式类型

        Returns:
            MIME类型字符串
        """
        formatter = self._formatters.get(format_type)
        if not formatter:
            return "application/json"  # 默认类型

        return formatter.get_content_type()

    def supports_batch(self, format_type: StandardFormatType) -> bool:
        """
        检查格式是否支持批量操作

        Args:
            format_type: 格式类型

        Returns:
            是否支持批量
        """
        formatter = self._formatters.get(format_type)
        if not formatter:
            return False

        return formatter.supports_batch()

    def get_supported_formats(self) -> List[StandardFormatType]:
        """
        获取所有支持的格式类型

        Returns:
            支持的格式类型列表
        """
        return list(self._formatters.keys())

    @staticmethod
    def create_standard_entry(
        timestamp: datetime,
        level: LogLevel,
        message: str,
        category: LogCategory = LogCategory.SYSTEM,
        source: str = "",
        host: str = "",
        service: str = "",
        environment: str = "production",
        trace_id: Optional[str] = None,
        span_id: Optional[str] = None,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        request_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        extra_fields: Optional[Dict[str, Any]] = None
    ) -> StandardLogEntry:
        """
        创建标准日志条目

        Args:
            timestamp: 时间戳
            level: 日志级别
            message: 日志消息
            category: 日志类别
            source: 日志来源
            host: 主机名
            service: 服务名
            environment: 环境
            trace_id: 追踪ID
            span_id: 跨度ID
            user_id: 用户ID
            session_id: 会话ID
            request_id: 请求ID
            correlation_id: 关联ID
            metadata: 元数据
            tags: 标签列表
            extra_fields: 额外字段

        Returns:
            标准日志条目
        """
        return StandardLogEntry(
            timestamp=timestamp,
            level=level,
            message=message,
            category=category,
            source=source,
            host=host,
            service=service,
            environment=environment,
            trace_id=trace_id,
            span_id=span_id,
            user_id=user_id,
            session_id=session_id,
            request_id=request_id,
            correlation_id=correlation_id,
            metadata=metadata or {},
            tags=tags or [],
            extra_fields=extra_fields or {}
        )

    def convert_from_internal_format(
        self,
        internal_record: Dict[str, Any]
    ) -> StandardLogEntry:
        """
        从内部日志格式转换为标准格式

        Args:
            internal_record: 内部日志记录

        Returns:
            标准日志条目
        """
        # 假设内部格式包含以下字段
        return StandardLogEntry(
            timestamp=internal_record.get("timestamp", datetime.now()),
            level=internal_record.get("level", LogLevel.INFO),
            message=internal_record.get("message", ""),
            category=internal_record.get("category", LogCategory.GENERAL),
            source=internal_record.get("source", ""),
            host=internal_record.get("host", ""),
            service=internal_record.get("service", ""),
            environment=internal_record.get("environment", "production"),
            trace_id=internal_record.get("trace_id"),
            span_id=internal_record.get("span_id"),
            user_id=internal_record.get("user_id"),
            session_id=internal_record.get("session_id"),
            request_id=internal_record.get("request_id"),
            correlation_id=internal_record.get("correlation_id"),
            metadata=internal_record.get("metadata", {}),
            tags=internal_record.get("tags", []),
            extra_fields=internal_record.get("extra_fields", {})
        )
