"""
remote 模块

提供 remote 相关功能和接口。
"""

import logging

import socket
import time

from .base import BaseHandler
from ..core.exceptions import LogHandlerError as HandlerError
from datetime import datetime
from typing import Any, Dict, Optional
"""
基础设施层 - 远程日志处理器

实现远程日志输出功能，支持网络传输。
"""


class RemoteHandler(BaseHandler):
    """远程日志处理器"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        初始化远程处理器

        Args:
            config: 处理器配置
        """
        super().__init__(config)

        # 初始化logger
        import logging
        self._logger = logging.getLogger(f"{self.__class__.__name__}")

        self.host = self.config.get('host', 'localhost')
        self.port = self.config.get('port', 514)  # 默认syslog端口
        self.protocol = self.config.get('protocol', 'tcp')  # tcp 或 udp
        self.timeout = self.config.get('timeout', 5.0)
        self.retry_count = self.config.get('retry_count', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)

        self._socket = None
        self._connected = False
        self._formatter = None

    def _emit(self, record: logging.LogRecord) -> None:
        """
        发出日志记录到远程服务器

        Args:
            record: 日志记录
        """
        message = self._format_record(record)
        self._send_with_retry(message)

    def _send_with_retry(self, message: str) -> None:
        """
        带重试机制发送消息

        Args:
            message: 要发送的消息

        Raises:
            HandlerError: 发送失败
        """
        for attempt in range(self.retry_count + 1):
            try:
                self._ensure_connection()
                if self._socket:
                    self._send_message(message)
                return
            except Exception as e:
                self._handle_send_failure(e, attempt)

        # 如果所有重试都失败
        raise HandlerError(f"Failed to send log after {self.retry_count + 1} attempts")

    def _ensure_connection(self) -> None:
        """确保连接已建立"""
        if not self._connected:
            self._connect()

    def _handle_send_failure(self, error: Exception, attempt: int) -> None:
        """
        处理发送失败

        Args:
            error: 发送错误
            attempt: 当前尝试次数
        """
        self._connected = False
        if attempt < self.retry_count:
            self._logger.warning(
                f"Failed to send log to {self.host}:{self.port}, attempt {attempt + 1}: {error}")
            time.sleep(self.retry_delay)
        else:
            raise HandlerError(f"Failed to send log after {self.retry_count + 1} attempts: {error}")

    def _format_record(self, record: logging.LogRecord) -> str:
        """格式化日志记录"""
        if self._formatter:
            return self._formatter.format(record)
        else:
            # 默认格式
            timestamp = datetime.fromtimestamp(record.created).isoformat()
            return f"<{record.levelno}>{timestamp} {record.name}: {record.getMessage()}"

    def _connect(self) -> None:
        """连接到远程服务器"""
        try:
            if self.protocol.lower() == 'tcp':
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self._socket.settimeout(self.timeout)
                self._socket.connect((self.host, self.port))
            elif self.protocol.lower() == 'udp':
                self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self._socket.settimeout(self.timeout)
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}")

            self._connected = True

        except Exception as e:
            raise ConnectionError(f"Failed to connect to {self.host}:{self.port}: {e}")

    def _send_message(self, message: str) -> None:
        """发送消息"""
        if not self._socket or not self._connected:
            raise ConnectionError("Not connected to remote server")

        data = message.encode('utf-8')

        if self.protocol.lower() == 'tcp':
            self._socket.sendall(data)
        elif self.protocol.lower() == 'udp':
            self._socket.sendto(data, (self.host, self.port))

    def _close(self) -> None:
        """关闭远程处理器"""
        if self._socket:
            try:
                self._socket.close()
            except Exception:
                pass
            self._socket = None
        self._connected = False

    def set_formatter(self, formatter: logging.Formatter) -> None:
        """设置格式化器"""
        self._formatter = formatter

    def setFormatter(self, formatter: logging.Formatter) -> None:
        """设置格式化器（兼容logging接口）"""
        self.set_formatter(formatter)

    def get_status(self) -> Dict[str, Any]:
        """获取处理器状态"""
        status = super().get_status()
        status.update({
            'host': self.host,
            'port': self.port,
            'protocol': self.protocol,
            'connected': self._connected,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'has_formatter': self._formatter is not None
        })
        return status

    def get_config(self) -> Dict[str, Any]:
        """获取处理器配置"""
        config = {
            'name': self.name,
            'host': self.host,
            'port': self.port,
            'protocol': self.protocol,
            'timeout': self.timeout,
            'retry_count': self.retry_count,
            'retry_delay': self.retry_delay
        }
        return config

    def is_connected(self) -> bool:
        """检查是否已连接"""
        return self._connected
