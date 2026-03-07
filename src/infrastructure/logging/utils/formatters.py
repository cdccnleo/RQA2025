"""
formatters 模块

提供 formatters 相关功能和接口。
"""

import json
import logging

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
        try:
            message = record.getMessage()
        except (AttributeError, TypeError):
            message = getattr(record, 'message', str(record))

        data = {
            'timestamp': datetime.now().isoformat(),
            'level': getattr(record, 'levelname', logging.getLevelName(getattr(record, 'levelno', logging.INFO))),
            'component': getattr(record, 'name', '').split('.')[-1] if '.' in getattr(record, 'name', '') else getattr(record, 'name', ''),
            'message': message,
            'pathname': getattr(record, 'pathname', ''),
            'lineno': getattr(record, 'lineno', 0),
            'funcName': getattr(record, 'funcName', '')
        }

        if hasattr(record, 'extra_data') and record.extra_data:
            # 只添加可序列化的额外数据
            for key, value in record.extra_data.items():
                try:
                    json.dumps({key: value})
                    data[key] = value
                except (TypeError, ValueError):
                    # 跳过不可序列化的值
                    pass

        # 添加异常信息（如果有）
        if getattr(record, 'exc_text', None):
            data['exception'] = {
                'traceback': record.exc_text
            }

        return json.dumps(data, ensure_ascii=False)

    @staticmethod
    def format_structured(record: Any) -> str:
        """格式化为结构化文本格式"""
        base = LogFormatter.format_text(record, include_colors=False)

        if hasattr(record, 'extra_data') and record.extra_data:
            fields = " ".join([f"{k}={v}" for k, v in record.extra_data.items()])
            return f"{base} | {fields}"

        return base
