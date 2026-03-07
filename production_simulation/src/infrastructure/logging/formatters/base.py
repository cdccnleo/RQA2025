"""
base 模块

提供 base 相关功能和接口。
"""

import logging

import time

from ..core.interfaces import ILogFormatter, LogFormat
from abc import abstractmethod
from typing import Any, Dict, Optional
"""
基础设施层 - 日志格式化器基础实现

定义日志格式化器的基础接口和实现。
"""


class BaseFormatter(ILogFormatter):
    """基础日志格式化器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础格式化器

        Args:
            config: 格式化器配置
        """
        self.config = config or {}
        self.name = self.config.get('name', self.__class__.__name__)
        self.date_format = self.config.get('date_format', '%Y-%m-%d %H:%M:%S')
        self.include_level = self.config.get('include_level', True)
        self.include_logger_name = self.config.get('include_logger_name', True)
        self.include_timestamp = self.config.get('include_timestamp', True)
        self.max_message_length = self.config.get('max_message_length', 0)  # 0表示不限制

    def format(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        try:
            return self._format(record)
        except Exception as e:
            # 格式化失败时的后备格式
            return f"[FORMAT_ERROR] {record.getMessage()} - Error: {e}"

    @abstractmethod
    def _format(self, record: logging.LogRecord) -> str:
        """实际的格式化逻辑，由子类实现"""

    def set_format(self, format_type: LogFormat) -> None:
        """
        设置格式类型

        Args:
            format_type: 格式类型
        """
        # 基础实现，子类可以重写

    def _truncate_message(self, message: str) -> str:
        """截断过长的消息"""
        if self.max_message_length > 0 and len(message) > self.max_message_length:
            return message[:self.max_message_length - 3] + "..."
        return message

    def _format_timestamp(self, record: logging.LogRecord) -> str:
        """格式化时间戳"""
        ct = time.localtime(record.created)
        return time.strftime(self.date_format, ct)

    def _format_level(self, record: logging.LogRecord) -> str:
        """格式化日志级别"""
        return record.levelname

    def _format_logger_name(self, record: logging.LogRecord) -> str:
        """格式化日志器名称"""
        return record.name

    def get_config(self) -> Dict[str, Any]:
        """获取格式化器配置"""
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'date_format': self.date_format,
            'include_level': self.include_level,
            'include_logger_name': self.include_logger_name,
            'include_timestamp': self.include_timestamp,
            'max_message_length': self.max_message_length
        }
