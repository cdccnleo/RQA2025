"""
日志通知通道模块

实现通过日志系统记录通知的功能
"""

import logging
import logging.handlers
from typing import Dict, Any, Optional
from datetime import datetime
from pathlib import Path

try:
    from .notification_service import NotificationChannel, NotificationLevel, NotificationResult
except ImportError:
    from notification_service import NotificationChannel, NotificationLevel, NotificationResult


class LogNotificationChannel(NotificationChannel):
    """
    日志通知通道

    通过日志系统记录通知信息

    Attributes:
        logger_name: 日志记录器名称
        log_file: 日志文件路径
        max_bytes: 单个日志文件最大大小
        backup_count: 备份文件数量
        console_output: 是否输出到控制台
    """

    # 通知级别到日志级别的映射
    LEVEL_MAPPING = {
        NotificationLevel.DEBUG: logging.DEBUG,
        NotificationLevel.INFO: logging.INFO,
        NotificationLevel.WARNING: logging.WARNING,
        NotificationLevel.ERROR: logging.ERROR,
        NotificationLevel.CRITICAL: logging.CRITICAL,
    }

    def __init__(
        self,
        name: str = "log",
        logger_name: str = "notification.log",
        log_file: Optional[str] = None,
        max_bytes: int = 10 * 1024 * 1024,  # 10MB
        backup_count: int = 5,
        console_output: bool = True,
        log_format: Optional[str] = None,
        enabled: bool = True
    ):
        """
        初始化日志通知通道

        Args:
            name: 通道名称
            logger_name: 日志记录器名称
            log_file: 日志文件路径
            max_bytes: 单个日志文件最大大小
            backup_count: 备份文件数量
            console_output: 是否输出到控制台
            log_format: 日志格式
            enabled: 是否启用
        """
        super().__init__(name, enabled)
        self.logger_name = logger_name
        self.log_file = log_file
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.console_output = console_output

        # 创建专用日志记录器
        self._log = logging.getLogger(logger_name)
        self._log.setLevel(logging.DEBUG)

        # 避免重复添加处理器
        self._log.handlers = []

        # 设置日志格式
        self.log_format = log_format or "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        formatter = logging.Formatter(self.log_format)

        # 添加文件处理器
        if log_file:
            self._setup_file_handler(formatter)

        # 添加控制台处理器
        if console_output:
            self._setup_console_handler(formatter)

        # 存储历史记录
        self._history: list = []
        self._max_history = 1000

    def _setup_file_handler(self, formatter: logging.Formatter) -> None:
        """
        配置文件处理器

        Args:
            formatter: 日志格式器
        """
        try:
            # 确保目录存在
            log_path = Path(self.log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_handler = logging.handlers.RotatingFileHandler(
                self.log_file,
                maxBytes=self.max_bytes,
                backupCount=self.backup_count,
                encoding="utf-8"
            )
            file_handler.setFormatter(formatter)
            self._log.addHandler(file_handler)
            self.logger.debug(f"日志文件处理器已配置: {self.log_file}")
        except Exception as e:
            self.logger.error(f"配置日志文件处理器失败: {e}")

    def _setup_console_handler(self, formatter: logging.Formatter) -> None:
        """
        配置控制台处理器

        Args:
            formatter: 日志格式器
        """
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self._log.addHandler(console_handler)
        self.logger.debug("控制台日志处理器已配置")

    def send(
        self,
        message: str,
        level: NotificationLevel = NotificationLevel.INFO,
        **kwargs: Any
    ) -> NotificationResult:
        """
        记录日志通知

        Args:
            message: 通知内容
            level: 通知级别
            **kwargs: 额外参数
                - extra: 额外日志字段
                - exc_info: 是否包含异常信息
                - stack_info: 是否包含堆栈信息

        Returns:
            发送结果
        """
        # 验证消息
        if not self.validate_message(message):
            return NotificationResult(
                channel_name=self.name,
                success=False,
                error="消息验证失败"
            )

        try:
            # 获取日志级别
            log_level = self.LEVEL_MAPPING.get(level, logging.INFO)

            # 构建额外字段
            extra = kwargs.get("extra", {})
            extra["notification_channel"] = self.name
            extra["notification_level"] = level.name

            # 记录日志
            self._log.log(
                log_level,
                message,
                extra=extra,
                exc_info=kwargs.get("exc_info", False),
                stack_info=kwargs.get("stack_info", False)
            )

            # 添加到历史记录
            self._add_to_history(message, level)

            return NotificationResult(
                channel_name=self.name,
                success=True,
                response_data={
                    "logger_name": self.logger_name,
                    "log_level": logging.getLevelName(log_level),
                    "timestamp": datetime.now().isoformat(),
                    "log_file": self.log_file
                }
            )

        except Exception as e:
            error_msg = f"记录日志失败: {e}"
            self.logger.error(error_msg)
            return NotificationResult(
                channel_name=self.name,
                success=False,
                error=error_msg
            )

    def _add_to_history(self, message: str, level: NotificationLevel) -> None:
        """
        添加到历史记录

        Args:
            message: 消息内容
            level: 通知级别
        """
        self._history.append({
            "timestamp": datetime.now().isoformat(),
            "level": level.name,
            "message": message[:200]  # 限制长度
        })

        # 限制历史记录大小
        if len(self._history) > self._max_history:
            self._history = self._history[-self._max_history:]

    def get_history(
        self,
        level: Optional[NotificationLevel] = None,
        limit: int = 100
    ) -> list:
        """
        获取历史记录

        Args:
            level: 过滤级别
            limit: 返回数量限制

        Returns:
            历史记录列表
        """
        history = self._history

        if level:
            history = [
                h for h in history
                if h["level"] == level.name
            ]

        return history[-limit:]

    def clear_history(self) -> None:
        """清空历史记录"""
        self._history = []
        self.logger.debug("历史记录已清空")

    def set_level(self, level: NotificationLevel) -> None:
        """
        设置日志级别

        Args:
            level: 最低记录级别
        """
        log_level = self.LEVEL_MAPPING.get(level, logging.INFO)
        self._log.setLevel(log_level)
        self.logger.debug(f"日志级别设置为: {logging.getLevelName(log_level)}")

    def is_enabled_for_level(self, level: NotificationLevel) -> bool:
        """
        检查通道是否对指定级别启用

        Args:
            level: 通知级别

        Returns:
            是否启用
        """
        if not self.enabled:
            return False

        log_level = self.LEVEL_MAPPING.get(level, logging.INFO)
        return self._log.isEnabledFor(log_level)

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息

        Returns:
            统计信息字典
        """
        level_counts = {}
        for record in self._history:
            level = record["level"]
            level_counts[level] = level_counts.get(level, 0) + 1

        return {
            "total_records": len(self._history),
            "level_distribution": level_counts,
            "logger_name": self.logger_name,
            "log_file": self.log_file,
            "handlers_count": len(self._log.handlers)
        }

    def rotate_log(self) -> bool:
        """
        手动轮转日志文件

        Returns:
            是否成功
        """
        for handler in self._log.handlers:
            if isinstance(handler, logging.handlers.RotatingFileHandler):
                try:
                    handler.doRollover()
                    self.logger.info("日志文件已轮转")
                    return True
                except Exception as e:
                    self.logger.error(f"日志轮转失败: {e}")
                    return False
        return False

    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典

        Returns:
            配置字典
        """
        return {
            "name": self.name,
            "enabled": self.enabled,
            "logger_name": self.logger_name,
            "log_file": self.log_file,
            "max_bytes": self.max_bytes,
            "backup_count": self.backup_count,
            "console_output": self.console_output,
            "log_format": self.log_format,
            "handlers_count": len(self._log.handlers)
        }
