"""
console 模块

提供 console 相关功能和接口。
"""

import logging
import sys

from .base import BaseHandler
from typing import Any, Dict, Optional
"""
基础设施层 - 控制台日志处理器

实现控制台日志输出功能。
"""


class ConsoleHandler(BaseHandler):
    """控制台日志处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化控制台处理器

        Args:
            config: 处理器配置
        """
        super().__init__(config)
        self.stream = self.config.get('stream', sys.stdout)
        self.colorize = self.config.get('colorize', False)
        self._formatter = None

    def _emit(self, record: logging.LogRecord) -> None:
        """发出日志记录到控制台"""
        try:
            message = self._format_record(record)
            if self.colorize:
                message = self._colorize_message(message, record.levelno)

            print(message, file=self.stream, flush=True)
        except Exception as e:
            # 控制台输出失败时的后备处理
            print(f"[ERROR] Failed to write to console: {e}", file=sys.stderr)

    def _format_record(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        if self._formatter:
            return self._formatter.format(record)
        else:
            # 默认格式
            return f"{record.levelname}: {record.getMessage()}"

    def _colorize_message(self, message: str, level: int) -> str:
        """为消息添加颜色"""
        colors = {
            logging.DEBUG: '\033[36m',    # 青色
            logging.INFO: '\033[32m',     # 绿色
            logging.WARNING: '\033[33m',  # 黄色
            logging.ERROR: '\033[31m',    # 红色
            logging.CRITICAL: '\033[35m',  # 紫色
        }

        color = colors.get(level, '')
        reset = '\033[0m' if color else ''

        return f"{color}{message}{reset}"

    def set_formatter(self, formatter: logging.Formatter) -> None:
        """设置格式化器"""
        self._formatter = formatter

    def _close(self) -> None:
        """关闭控制台处理器"""
        # 控制台处理器通常不需要特殊关闭操作

    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        status = super().get_status()
        status.update({
            'stream': 'stdout' if self.stream == sys.stdout else 'stderr',
            'colorize': self.colorize,
            'has_formatter': self._formatter is not None
        })
        return status
