"""
error 模块

提供 error 相关功能和接口。
"""

import logging

from datetime import datetime
from typing import Dict, Any
"""
基础设施层工具系统 - 统一错误处理器

提供统一的错误处理和日志记录功能。
"""


class UnifiedErrorHandler:
    """统一错误处理器"""

    def __init__(self, logger_name: str = "infrastructure.error"):
        """
        初始化错误处理器

        Args:
            logger_name: 日志器名称
        """
        self.logger = logging.getLogger(logger_name)
        self.error_stats: Dict[str, int] = {}
        self.last_errors: list = []

    def handle(self, error: Exception, context: str = "", level: str = "error") -> None:
        """
        处理错误

        Args:
            error: 异常对象
            context: 错误上下文信息
            level: 日志级别 ('debug', 'info', 'warning', 'error', 'critical')
        """
        try:
            error_type = error.__class__.__name__
            self._update_error_stats(error_type)
            self._record_error_details(error_type, error, context, level)
            self._log_error(error_type, error, context, level)
        except Exception as e:
            self._handle_final_failure(e, error)

    def _update_error_stats(self, error_type: str) -> None:
        """更新错误统计"""
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1

    def _record_error_details(self, error_type: str, error: Exception, context: str, level: str) -> None:
        """记录错误详情"""
        try:
            error_info = self._create_error_info(error_type, error, context, level)
            self.last_errors.append(error_info)
            self._manage_error_history()
        except Exception as e:
            self._handle_error_recording_failure(e)

    def _create_error_info(self, error_type: str, error: Exception, context: str, level: str) -> Dict[str, Any]:
        """创建错误信息字典"""
        return {
            "error_type": error_type,
            "message": str(error),
            "context": context,
            "timestamp": datetime.now().isoformat(),
            "level": level,
        }

    def _manage_error_history(self) -> None:
        """管理错误历史大小"""
        while len(self.last_errors) > 100:
            self.last_errors.pop(0)

    def _log_error(self, error_type: str, error: Exception, context: str, level: str) -> None:
        """记录错误日志"""
        try:
            log_message = self._create_log_message(error_type, error, context)
            self._execute_logging(log_message, level)
        except Exception as log_error:
            self._handle_logging_failure(log_error, error_type, error)

    def _create_log_message(self, error_type: str, error: Exception, context: str) -> str:
        """创建日志消息"""
        return f"[{context}] {error_type}: {error}" if context else f"{error_type}: {error}"

    def _execute_logging(self, log_message: str, level: str) -> None:
        """执行日志记录"""
        if level == "debug":
            self.logger.debug(log_message, exc_info=True)
        elif level == "info":
            self.logger.info(log_message, exc_info=True)
        elif level == "warning":
            self.logger.warning(log_message, exc_info=True)
        elif level == "critical":
            self.logger.critical(log_message, exc_info=True)
        else:  # error (default)
            self.logger.error(log_message, exc_info=True)

    def _handle_error_recording_failure(self, recording_error: Exception) -> None:
        """处理错误记录失败"""
        print(f"Failed to record error info: {recording_error}")

    def _handle_logging_failure(self, log_error: Exception, error_type: str, original_error: Exception) -> None:
        """处理日志记录失败"""
        print(f"Logging failed: {log_error}")
        print(f"Original error: {error_type}: {original_error}")

    def _handle_final_failure(self, handler_error: Exception, original_error: Exception) -> None:
        """处理最后的错误处理失败"""
        print(f"Critical error in error handler: {handler_error}")
        print(f"Original error was: {original_error}")

    def get_error_stats(self) -> Dict[str, Any]:
        """
        获取错误统计信息

        Returns:
            Dict[str, Any]: 错误统计
        """
        return {
            "total_errors": sum(self.error_stats.values()),
            "error_types": self.error_stats.copy(),
            "recent_errors_count": len(self.last_errors),
        }

    def get_recent_errors(self, limit: int = 10) -> list:
        """
        获取最近的错误

        Args:
            limit: 返回的最大错误数

        Returns:
            list: 最近错误列表（倒序，最新的在前）
        """
        return list(reversed(self.last_errors[-limit:]))

    def clear_stats(self) -> None:
        """清空错误统计"""
        self.error_stats.clear()
        self.last_errors.clear()

    def handle_connection_error(self, error: Exception, host: str = "", port: int = 0) -> None:
        """
        处理连接错误

        Args:
            error: 连接异常
            host: 主机地址
            port: 端口号
        """
        context = f"Connection failed to {host}:{port}" if host else "Connection failed"
        self.handle(error, context, "error")

    def handle_timeout_error(self, error: Exception, operation: str = "", timeout: float = 0.0) -> None:
        """
        处理超时错误

        Args:
            error: 超时异常
            operation: 操作名称
            timeout: 超时时间（秒）
        """
        context = f"Operation '{operation}' timed out after {timeout}s" if operation else f"Timeout after {timeout}s"
        self.handle(error, context, "warning")

    def handle_validation_error(self, error: Exception, field: str = "", value: Any = None) -> None:
        """
        处理验证错误

        Args:
            error: 验证异常
            field: 字段名
            value: 字段值
        """
        context = f"Validation failed for field '{field}'" if field else "Validation failed"
        self.handle(error, context, "warning")


# 创建默认的全局错误处理器实例
default_error_handler = UnifiedErrorHandler()


def get_error_handler() -> UnifiedErrorHandler:
    """
    获取默认错误处理器实例

    Returns:
        UnifiedErrorHandler: 默认错误处理器
    """
    return default_error_handler
