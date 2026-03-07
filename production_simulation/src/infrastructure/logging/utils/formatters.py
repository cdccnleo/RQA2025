"""
formatters 模块

提供 formatters 相关功能和接口。
"""

import json

from datetime import datetime
from typing import Any
"""
基础设施层 - 日志格式化工具

提供各种日志格式化器的实现。
"""


class LogFormatter:
    """日志格式化工具类"""

    @staticmethod
    def format_text(record: Any, include_colors: bool = False) -> str:
        """格式化为文本格式"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        level = record.levelname
        component = record.name.split('.')[-1] if '.' in record.name else record.name
        message = record.getMessage()

        if include_colors:
            colors = {
                'DEBUG': '\033[36m',
                'INFO': '\033[32m',
                'WARNING': '\033[33m',
                'ERROR': '\033[31m',
                'CRITICAL': '\033[35m',
                'RESET': '\033[0m'
            }
            color = colors.get(level, colors['RESET'])
            reset = colors['RESET']
            return f"[{timestamp}] [{color}{level}{reset}] [{component}] {message}"
        else:
            return f"[{timestamp}] [{level}] [{component}] {message}"

    @staticmethod
    def format_json(record: Any) -> str:
        """格式化为JSON格式"""
        data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'component': record.name.split('.')[-1] if '.' in record.name else record.name,
            'message': record.getMessage(),
            'pathname': record.pathname,
            'lineno': record.lineno,
            'funcName': record.funcName
        }

        if hasattr(record, 'extra_data') and record.extra_data:
            data.update(record.extra_data)

        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def format_structured(record: Any) -> str:
        """格式化为结构化文本格式"""
        base = LogFormatter.format_text(record, include_colors=False)

        if hasattr(record, 'extra_data') and record.extra_data:
            fields = " ".join([f"{k}={v}" for k, v in record.extra_data.items()])
            return f"{base} | {fields}"

        return base
