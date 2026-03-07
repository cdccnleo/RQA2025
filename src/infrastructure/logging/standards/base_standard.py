"""
base_standard 模块

提供 base_standard 相关功能和接口。
"""


import uuid

from ..core.interfaces import LogLevel, LogCategory
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
"""
RQA2025 基础设施层 - 基础标准格式类

定义日志分析平台标准格式的基础接口和抽象类。
"""


class StandardFormatType(Enum):
    """标准格式类型枚举"""
    ELK = "elk"
    SPLUNK = "splunk"
    DATADOG = "datadog"
    NEW_RELIC = "newrelic"
    LOKI = "loki"
    GRAYLOG = "graylog"
    FLUENTD = "fluentd"
    LOGDNA = "logdna"
    PAPERTRAIL = "papertrail"
    SUMO_LOGIC = "sumologic"


@dataclass
class StandardLogEntry:
    """标准日志条目"""
    timestamp: datetime
    level: LogLevel
    message: str
    category: LogCategory = LogCategory.SYSTEM
    source: str = ""
    host: str = ""
    service: str = ""
    environment: str = "production"
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    request_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    extra_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """自动生成缺失的ID"""
        if not self.trace_id:
            self.trace_id = str(uuid.uuid4())
        if not self.request_id:
            self.request_id = str(uuid.uuid4())
        if not self.correlation_id:
            self.correlation_id = self.trace_id


class BaseStandardFormat(ABC):
    """基础标准格式抽象类"""

    def __init__(self, format_type: StandardFormatType):
        self.format_type = format_type

    @abstractmethod
    def format_log_entry(self, entry: Union[StandardLogEntry, Dict[str, Any]]) -> Union[str, Dict[str, Any]]:
        """
        将标准日志条目格式化为特定平台的格式

        Args:
            entry: 标准日志条目

        Returns:
            格式化后的日志字符串或字典
        """

    def get_content_type(self) -> str:
        """
        获取内容类型

        Returns:
            MIME类型字符串，默认返回JSON格式
        """
        return "application/json"

    def _dict_to_standard_entry(self, data: Dict[str, Any]) -> StandardLogEntry:
        """将字典转换为StandardLogEntry对象"""
        # 转换时间戳
        timestamp = data.get('timestamp')
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif not isinstance(timestamp, datetime):
            timestamp = datetime.now()

        # 转换级别
        level_str = data.get('level', 'INFO')
        level = LogLevel[level_str.upper()] if isinstance(level_str, str) else level_str

        # 转换类别
        category_str = data.get('category', 'SYSTEM')
        category = LogCategory[category_str.upper()] if isinstance(category_str, str) else category_str

        return StandardLogEntry(
            timestamp=timestamp,
            level=level,
            message=data.get('message', ''),
            category=category,
            source=data.get('source', ''),
            host=data.get('host', ''),
            service=data.get('service', ''),
            environment=data.get('environment', 'production'),
            trace_id=data.get('trace_id'),
            span_id=data.get('span_id'),
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            correlation_id=data.get('correlation_id'),
            metadata=data.get('metadata', {}),
            tags=data.get('tags', []),
            extra_fields=data  # 保留原始数据以备后用
        )

    @abstractmethod
    def supports_batch(self) -> bool:
        """
        是否支持批量发送

        Returns:
            是否支持批量
        """

    def format_batch(self, entries: List[StandardLogEntry]) -> Union[str, List[Dict[str, Any]]]:
        """
        批量格式化日志条目

        Args:
            entries: 日志条目列表

        Returns:
            批量格式化结果
        """
        if not self.supports_batch():
            raise NotImplementedError(f"{self.format_type.value} 不支持批量格式化")

        return [self.format_log_entry(entry) for entry in entries]

    def validate_entry(self, entry: StandardLogEntry) -> bool:
        """
        验证日志条目

        Args:
            entry: 要验证的日志条目

        Returns:
            验证是否通过
        """
        if not entry.message:
            return False
        if not isinstance(entry.level, LogLevel):
            return False
        if not isinstance(entry.category, LogCategory):
            return False
        return True

    @staticmethod
    def convert_log_level(level: LogLevel) -> str:
        """转换日志级别为字符串"""
        return level.value.upper()

    @staticmethod
    def timestamp_to_iso(timestamp: datetime) -> str:
        """转换时间戳为ISO格式"""
        return timestamp.isoformat()

    @staticmethod
    def timestamp_to_unix_ms(timestamp: datetime) -> int:
        """转换时间戳为Unix毫秒时间戳"""
        return int(timestamp.timestamp() * 1000)

    @staticmethod
    def timestamp_to_unix_ns(timestamp: datetime) -> int:
        """转换时间戳为Unix纳秒时间戳"""
        return int(timestamp.timestamp() * 1_000_000_000)
