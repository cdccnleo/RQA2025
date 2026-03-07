"""
text 模块

提供 text 相关功能和接口。
"""

import logging

from .base import BaseFormatter
from typing import Any, Dict, Optional
"""
基础设施层 - 文本日志格式化器

实现文本格式的日志输出。
"""


class TextFormatter(BaseFormatter):
    """文本日志格式化器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化文本格式化器

        Args:
            config: 格式化器配置
        """
        super().__init__(config)
        self.template = self.config.get('template',
                                        '{timestamp} {level} {logger}: {message}')

    def _format(self, record: logging.LogRecord) -> str:
        """格式化日志记录为文本"""
        # 获取各个部分的格式化结果
        parts = {}

        # 总是准备所有可能的字段，根据配置决定是否包含
        if self.include_timestamp:
            parts['timestamp'] = self._format_timestamp(record)
        else:
            parts['timestamp'] = ''  # 提供空字符串以避免KeyError

        if self.include_level:
            parts['level'] = self._format_level(record)
        else:
            parts['level'] = ''

        if self.include_logger_name:
            parts['logger'] = self._format_logger_name(record)
        else:
            parts['logger'] = ''

        parts['message'] = self._truncate_message(record.getMessage())

        # 使用模板格式化
        try:
            formatted = self.template.format(**parts)
        except KeyError as e:
            # 模板中引用了未定义的字段
            raise ValueError(
                f"Template contains undefined field '{e.args[0]}'")

        return formatted

    def set_template(self, template: str) -> None:
        """设置格式化模板"""
        self.template = template

    def get_config(self) -> Dict[str, Any]:
        """获取格式化器配置"""
        config = super().get_config()
        config['template'] = self.template
        return config
