"""
unified_logging_interface 模块

提供 unified_logging_interface 相关功能和接口。
"""

import logging

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional
#!/usr/bin/env python3
"""
统一日志管理接口

定义基础设施层日志管理的标准接口，确保所有日志管理器实现统一的API。
"""


class LogLevel(Enum):
    """日志级别"""
    DEBUG = 10
    INFO = 20
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


class LogFormat(Enum):
    """日志格式"""
    JSON = "json"
    TEXT = "text"
    STRUCTURED = "structured"
    XML = "xml"


class LogCategory(Enum):
    """日志分类"""
    SYSTEM = "system"
    BUSINESS = "business"
    SECURITY = "security"
    PERFORMANCE = "performance"
    AUDIT = "audit"
    ERROR = "error"


class ILogHandler(ABC):
    """
    日志处理器接口
    """

    @abstractmethod
    def emit(self, record: logging.LogRecord) -> None:
        """
        处理日志记录

        Args:
            record: 日志记录
        """

    @abstractmethod
    def close(self) -> None:
        """关闭处理器"""

    @abstractmethod
    def flush(self) -> None:
        """刷新缓冲区"""


class ILogFormatter(ABC):
    """
    日志格式化器接口
    """

    @abstractmethod
    def format(self, record: logging.LogRecord) -> str:
        """
        格式化日志记录

        Args:
            record: 日志记录

        Returns:
            格式化后的字符串
        """


class ILogger(ABC):
    """
    日志器统一接口

    所有日志器实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """
        记录DEBUG级别日志

        Args:
            message: 日志消息
            **kwargs: 额外参数
        """

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """
        记录INFO级别日志

        Args:
            message: 日志消息
            **kwargs: 额外参数
        """

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """
        记录WARNING级别日志

        Args:
            message: 日志消息
            **kwargs: 额外参数
        """

    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """
        记录ERROR级别日志

        Args:
            message: 日志消息
            **kwargs: 额外参数
        """

    @abstractmethod
    def critical(self, message: str, **kwargs) -> None:
        """
        记录CRITICAL级别日志

        Args:
            message: 日志消息
            **kwargs: 额外参数
        """

    @abstractmethod
    def log(self, level: LogLevel, message: str, **kwargs) -> None:
        """
        记录指定级别日志

        Args:
            level: 日志级别
            message: 日志消息
            **kwargs: 额外参数
        """

    @abstractmethod
    def is_enabled_for(self, level: LogLevel) -> bool:
        """
        检查指定级别是否启用

        Args:
            level: 日志级别

        Returns:
            是否启用
        """

    @abstractmethod
    def add_handler(self, handler: ILogHandler) -> bool:
        """
        添加日志处理器

        Args:
            handler: 日志处理器

        Returns:
            是否添加成功
        """

    @abstractmethod
    def remove_handler(self, handler: ILogHandler) -> bool:
        """
        移除日志处理器

        Args:
            handler: 日志处理器

        Returns:
            是否移除成功
        """

    @abstractmethod
    def set_level(self, level: LogLevel) -> None:
        """
        设置日志级别

        Args:
            level: 日志级别
        """

    @abstractmethod
    def get_level(self) -> LogLevel:
        """
        获取当前日志级别

        Returns:
            日志级别
        """

    @abstractmethod
    def set_formatter(self, formatter: ILogFormatter) -> None:
        """
        设置日志格式化器

        Args:
            formatter: 日志格式化器
        """

    @abstractmethod
    def get_formatter(self) -> Optional[ILogFormatter]:
        """
        获取日志格式化器

        Returns:
            日志格式化器
        """

    @abstractmethod
    def add_filter(self, filter_func: callable) -> bool:
        """
        添加日志过滤器

        Args:
            filter_func: 过滤函数

        Returns:
            是否添加成功
        """

    @abstractmethod
    def remove_filter(self, filter_func: callable) -> bool:
        """
        移除日志过滤器

        Args:
            filter_func: 过滤函数

        Returns:
            是否移除成功
        """

    @abstractmethod
    def get_handlers(self) -> List[ILogHandler]:
        """
        获取所有处理器

        Returns:
            处理器列表
        """

    @abstractmethod
    def flush(self) -> None:
        """刷新所有处理器"""

    @abstractmethod
    def close(self) -> None:
        """关闭日志器"""


class ILogMonitor(ABC):
    """
    日志监控器接口
    """

    @abstractmethod
    def on_log_record(self, record: logging.LogRecord) -> None:
        """
        处理日志记录

        Args:
            record: 日志记录
        """

    @abstractmethod
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取日志统计信息

        Returns:
            统计信息字典
        """

    @abstractmethod
    def get_recent_logs(self, count: int = 100) -> List[Dict[str, Any]]:
        """
        获取最近的日志记录

        Args:
            count: 获取数量

        Returns:
            日志记录列表
        """


class ILogManager(ABC):
    """
    日志管理器接口
    """

    @abstractmethod
    def get_logger(self, name: str) -> ILogger:
        """
        获取日志器

        Args:
            name: 日志器名称

        Returns:
            日志器实例
        """

    @abstractmethod
    def configure_logger(self, name: str, config: Dict[str, Any]) -> bool:
        """
        配置日志器

        Args:
            name: 日志器名称
            config: 配置字典

        Returns:
            是否配置成功
        """

    @abstractmethod
    def remove_logger(self, name: str) -> bool:
        """
        移除日志器

        Args:
            name: 日志器名称

        Returns:
            是否移除成功
        """

    @abstractmethod
    def get_all_loggers(self) -> Dict[str, ILogger]:
        """
        获取所有日志器

        Returns:
            日志器字典
        """

    @abstractmethod
    def set_global_level(self, level: LogLevel) -> None:
        """
        设置全局日志级别

        Args:
            level: 日志级别
        """

    @abstractmethod
    def get_global_level(self) -> LogLevel:
        """
        获取全局日志级别

        Returns:
            日志级别
        """

    @abstractmethod
    def add_global_handler(self, handler: ILogHandler) -> bool:
        """
        添加全局处理器

        Args:
            handler: 日志处理器

        Returns:
            是否添加成功
        """

    @abstractmethod
    def remove_global_handler(self, handler: ILogHandler) -> bool:
        """
        移除全局处理器

        Args:
            handler: 日志处理器

        Returns:
            是否移除成功
        """

    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统状态

        Returns:
            系统状态字典
        """
