"""
监控系统共享接口

定义监控系统各组件使用的标准接口
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class ILogger(ABC):
    """日志接口"""

    @abstractmethod
    def debug(self, message: str, **kwargs) -> None:
        """调试日志"""
        pass

    @abstractmethod
    def info(self, message: str, **kwargs) -> None:
        """信息日志"""
        pass

    @abstractmethod
    def warning(self, message: str, **kwargs) -> None:
        """警告日志"""
        pass

    @abstractmethod
    def error(self, message: str, **kwargs) -> None:
        """错误日志"""
        pass

    @abstractmethod
    def critical(self, message: str, **kwargs) -> None:
        """严重错误日志"""
        pass


class IErrorHandler(ABC):
    """错误处理器接口"""

    @abstractmethod
    def handle_error(self, error: Exception, message: str, **kwargs) -> None:
        """处理错误"""
        pass


class StandardLogger(ILogger):
    """标准日志实现"""

    def __init__(self, name: str = "monitor"):
        self.logger = logging.getLogger(name)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def debug(self, message: str, **kwargs) -> None:
        """调试日志"""
        self.logger.debug(message, **kwargs)

    def info(self, message: str, **kwargs) -> None:
        """信息日志"""
        self.logger.info(message, **kwargs)

    def warning(self, message: str, **kwargs) -> None:
        """警告日志"""
        self.logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs) -> None:
        """错误日志"""
        self.logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs) -> None:
        """严重错误日志"""
        self.logger.critical(message, **kwargs)


class BaseErrorHandler(IErrorHandler):
    """基础错误处理器实现"""

    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)

    def handle_error(self, error: Exception, message: str, **kwargs) -> None:
        """处理错误"""
        error_msg = f"{message}: {str(error)}"
        self.logger.error(error_msg, **kwargs)
        # 可以在这里添加更多的错误处理逻辑，如发送告警等
