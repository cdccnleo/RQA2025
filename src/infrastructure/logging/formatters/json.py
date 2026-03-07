"""
json 模块

提供 json 相关功能和接口。
"""

import json
import logging

import traceback

from .base import BaseFormatter
from datetime import datetime
from typing import Any, Dict, Optional
"""
基础设施层 - JSON日志格式化器

实现JSON格式的日志输出，便于结构化存储和分析。
"""


class JSONFormatter(BaseFormatter):
    """JSON日志格式化器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化JSON格式化器

        Args:
            config: 格式化器配置
        """
        super().__init__(config)
        self.pretty_print = self.config.get('pretty_print', False)
        self.include_extra = self.config.get('include_extra', True)
        self.include_exc_info = self.config.get('include_exc_info', True)
        self.custom_fields = self.config.get('custom_fields', {})

    def _format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录为JSON

        Args:
            record: 日志记录对象

        Returns:
            JSON格式化的字符串
        """
        try:
            # 构建基础日志数据
            log_data = self._build_base_log_data(record)

            # 添加可选字段
            self._add_optional_fields(log_data, record)

            # JSON序列化
            return self._serialize_to_json(log_data)

        except (TypeError, ValueError, AttributeError) as e:
            # JSON序列化失败时的后备处理
            return self._create_fallback_json(record, e)

    def _build_base_log_data(self, record: logging.LogRecord) -> Dict[str, Any]:
        """
        构建基础日志数据

        Args:
            record: 日志记录对象

        Returns:
            基础日志数据字典
        """
        log_data = {
            'timestamp': self._format_timestamp(record) if self.include_timestamp else None,
            'level': self._format_level(record) if self.include_level else None,
            'logger': self._format_logger_name(record) if self.include_logger_name else None,
            'message': self._truncate_message(record.getMessage()),
            'file': getattr(record, 'pathname', None),
            'line': getattr(record, 'lineno', None),
        }

        # 移除None值
        return {k: v for k, v in log_data.items() if v is not None}

    def _add_optional_fields(self, log_data: Dict[str, Any], record: logging.LogRecord) -> None:
        """
        添加可选字段到日志数据

        Args:
            log_data: 日志数据字典
            record: 日志记录对象
        """
        # 添加异常信息
        if self.include_exc_info and record.exc_info:
            log_data['exception'] = self._format_exception(record.exc_info)

        # 添加额外字段
        self._add_extra_fields(log_data, record)

        # 添加自定义字段
        self._add_custom_fields(log_data)

    def _add_extra_fields(self, log_data: Dict[str, Any], record: logging.LogRecord) -> None:
        """
        添加额外字段

        Args:
            log_data: 日志数据字典
            record: 日志记录对象
        """
        if not (self.include_extra and hasattr(record, '__dict__')):
            return

        exclude_keys = {
            'name', 'msg', 'args', 'levelname', 'levelno',
            'pathname', 'filename', 'module', 'exc_info',
            'exc_text', 'stack_info', 'lineno', 'funcName',
            'created', 'msecs', 'relativeCreated', 'thread',
            'threadName', 'processName', 'process', 'message'
        }

        for key, value in record.__dict__.items():
            if key not in exclude_keys:
                log_data[f'extra_{key}'] = value

    def _add_custom_fields(self, log_data: Dict[str, Any]) -> None:
        """
        添加自定义字段

        Args:
            log_data: 日志数据字典
        """
        for key, value in self.custom_fields.items():
            log_data[key] = value

    def _serialize_to_json(self, log_data: Dict[str, Any]) -> str:
        """
        序列化日志数据为JSON字符串

        Args:
            log_data: 日志数据字典

        Returns:
            JSON格式字符串
        """
        if self.pretty_print:
            return json.dumps(log_data, indent=2, ensure_ascii=False, default=str)
        else:
            return json.dumps(log_data, ensure_ascii=False, default=str)

    def _create_fallback_json(self, record: logging.LogRecord, error: Exception) -> str:
        """
        创建后备JSON格式（当序列化失败时）

        Args:
            record: 日志记录对象
            error: 序列化错误

        Returns:
            后备JSON格式字符串
        """
        fallback_data = {
            'timestamp': datetime.now().isoformat(),
            'level': 'ERROR',
            'logger': 'JSONFormatter',
            'message': f'Failed to format log record: {error}',
            'original_message': str(record.getMessage())[:500]  # 限制长度
        }
        return json.dumps(fallback_data, ensure_ascii=False, default=str)

    def _format_exception(self, exc_info) -> Dict[str, Any]:
        """格式化异常信息"""
        if not exc_info:
            return {}

        exc_type, exc_value, exc_traceback = exc_info

        return {
            'type': exc_type.__name__ if exc_type else 'Unknown',
            'message': str(exc_value) if exc_value else '',
            'traceback': traceback.format_exception(exc_type, exc_value, exc_traceback)
        }

    def add_custom_field(self, key: str, value: Any) -> None:
        """添加自定义字段"""
        self.custom_fields[key] = value

    def remove_custom_field(self, key: str) -> None:
        """移除自定义字段"""
        self.custom_fields.pop(key, None)

    def get_config(self) -> Dict[str, Any]:
        """获取格式化器配置"""
        config = super().get_config()
        config.update({
            'pretty_print': self.pretty_print,
            'include_extra': self.include_extra,
            'include_exc_info': self.include_exc_info,
            'custom_fields': self.custom_fields.copy()
        })
        return config
