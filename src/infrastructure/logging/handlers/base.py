"""
base 模块

提供 base 相关功能和接口。
"""

import logging

from ..core.interfaces import ILogHandler, LogLevel
from abc import abstractmethod
from typing import Any, Dict, Optional
"""
基础设施层 - 日志处理器基础实现

定义日志处理器的基础接口和实现。
"""


class BaseHandler(ILogHandler):
    """基础日志处理器"""

    def handle(self, record: Any) -> None:
        """处理日志记录（ILogHandler接口）"""
        # 将标准logging.LogRecord转换为我们期望的格式
        if hasattr(record, 'getMessage'):
            # 已经是LogRecord
            self.emit(record)
        else:
            # 其他格式，尝试转换
            self.emit(record)

    def get_level(self) -> LogLevel:
        """获取当前日志级别"""
        # 将logging级别转换为LogLevel枚举
        level_mapping = {
            10: LogLevel.DEBUG,
            20: LogLevel.INFO,
            30: LogLevel.WARNING,
            40: LogLevel.ERROR,
            50: LogLevel.CRITICAL
        }
        return level_mapping.get(self.level, LogLevel.INFO)

    def set_level(self, level: LogLevel) -> None:
        """设置处理器级别（ILogHandler接口）"""
        # 将LogLevel枚举转换为logging级别
        level_mapping = {
            LogLevel.DEBUG: 10,
            LogLevel.INFO: 20,
            LogLevel.WARNING: 30,
            LogLevel.ERROR: 40,
            LogLevel.CRITICAL: 50
        }
        self.level = level_mapping.get(level, 20)

    def _get_level_value(self) -> int:
        """获取当前级别的整数值"""
        if isinstance(self.level, LogLevel):
            level_mapping = {
                LogLevel.DEBUG: 10,
                LogLevel.INFO: 20,
                LogLevel.WARNING: 30,
                LogLevel.ERROR: 40,
                LogLevel.CRITICAL: 50
            }
            return level_mapping.get(self.level, 20)
        else:
            # 如果已经是整数，直接返回
            return self.level

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化基础处理器

        Args:
            config: 处理器配置
        """
        self.config = config or {}
        self.name = self.config.get('name', self.__class__.__name__)
        self.level = self.config.get('level', logging.INFO)
        self.enabled = self.config.get('enabled', True)
        self._closed = False

    def emit(self, record: logging.LogRecord) -> None:
        """发出日志记录"""
        if not self.enabled or self._closed:
            return

        # 检查日志级别是否满足要求
        if hasattr(record, 'levelno'):
            # 将LogLevel枚举转换为对应的整数值进行比较
            current_level = self._get_level_value()
            if record.levelno < current_level:
                return

        try:
            self._emit(record)
        except Exception as e:
            # 处理处理器内部错误
            self._handle_error(record, e)

    @abstractmethod
    def _emit(self, record: logging.LogRecord) -> None:
        """实际的日志发出逻辑，由子类实现"""

    def _handle_error(self, record: logging.LogRecord, error: Exception) -> None:
        """处理处理器错误"""
        # 默认行为：静默失败，避免递归错误

    def close(self) -> None:
        """关闭处理器"""
        self._closed = True
        self._close()

    def _close(self) -> None:
        """实际的关闭逻辑，由子类实现"""

    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        return {
            'name': self.name,
            'enabled': self.enabled,
            'level': self.level,
            'closed': self._closed,
            'type': self.__class__.__name__
        }

    def enable(self) -> None:
        """启用处理器"""
        self.enabled = True

    def disable(self) -> None:
        """禁用处理器"""
        self.enabled = False
