"""
structured 模块

提供 structured 相关功能和接口。
"""

import logging

from .base import BaseFormatter
from typing import Any, Dict, Optional
"""
基础设施层 - 结构化日志格式化器

实现结构化格式的日志输出，支持键值对和嵌套结构。
"""


class StructuredFormatter(BaseFormatter):
    """结构化日志格式化器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化结构化格式化器

        Args:
            config: 格式化器配置
        """
        super().__init__(config)
        self.field_separator = self.config.get('field_separator', ' | ')
        self.key_value_separator = self.config.get('key_value_separator', '=')
        self.include_empty_fields = self.config.get('include_empty_fields', False)

    def _format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录为结构化文本

        Args:
            record: 日志记录对象

        Returns:
            结构化格式的字符串
        """
        parts = []

        # 添加基础字段
        self._add_basic_fields(parts, record)

        # 添加消息字段
        self._add_message_field(parts, record)

        # 添加额外字段
        self._add_extra_fields(parts, record)

        # 组合所有字段
        return self.field_separator.join(parts)

    def _add_basic_fields(self, parts: list, record: logging.LogRecord) -> None:
        """
        添加基础字段到格式化部分

        Args:
            parts: 格式化部分列表
            record: 日志记录对象
        """
        if self.include_timestamp:
            parts.append(f"timestamp{self.key_value_separator}{self._format_timestamp(record)}")

        if self.include_level:
            parts.append(f"level{self.key_value_separator}{self._format_level(record)}")

        if self.include_logger_name:
            parts.append(f"logger{self.key_value_separator}{self._format_logger_name(record)}")

    def _add_message_field(self, parts: list, record: logging.LogRecord) -> None:
        """
        添加消息字段到格式化部分

        Args:
            parts: 格式化部分列表
            record: 日志记录对象
        """
        message = self._truncate_message(record.getMessage())
        parts.append(f"message{self.key_value_separator}{message}")

    def _add_extra_fields(self, parts: list, record: logging.LogRecord) -> None:
        """
        添加额外字段到格式化部分

        Args:
            parts: 格式化部分列表
            record: 日志记录对象
        """
        if not hasattr(record, '__dict__'):
            return

        exclude_keys = {
            'name', 'msg', 'args', 'levelname', 'levelno',
            'pathname', 'filename', 'module', 'exc_info',
            'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread',
            'threadName', 'processName', 'process', 'message'
        }

        for key, value in record.__dict__.items():
            if key not in exclude_keys and (value is not None or self.include_empty_fields):
                formatted_value = self._format_value(value)
                parts.append(f"{key}{self.key_value_separator}{formatted_value}")

    def _format_value(self, value: Any) -> str:
        """格式化字段值"""
        if isinstance(value, (list, tuple)):
            return ','.join(str(item) for item in value)
        elif isinstance(value, dict):
            return ','.join(f"{k}:{v}" for k, v in value.items())
        elif isinstance(value, bool):
            return 'true' if value else 'false'
        else:
            return str(value)

    def get_config(self) -> Dict[str, Any]:
        """获取格式化器配置"""
        config = super().get_config()
        config.update({
            'field_separator': self.field_separator,
            'key_value_separator': self.key_value_separator,
            'include_empty_fields': self.include_empty_fields
        })
        return config
