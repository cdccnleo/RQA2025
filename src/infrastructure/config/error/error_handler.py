"""配置错误处理器

提供统一的配置错误处理机制，包括：
- 错误分类与处理
- 日志记录
- 错误恢复
- 告警通知
"""

import logging
from typing import Dict, Any, Callable, List, Optional
from collections import defaultdict
import time

from .exceptions import (
    ConfigError,
    ConfigErrorType,
    ConfigLoadError,
    ConfigValidationError
)

logger = logging.getLogger(__name__)

class ConfigErrorHandler:
    """配置错误处理器"""

    def __init__(self, log_context: Optional[Dict[str, Any]] = None):
        """初始化错误处理器

        Args:
            log_context: 初始日志上下文
        """
        self._log_context = log_context or {}
        self._handlers = defaultdict(list)
        self._error_counts = defaultdict(int)
        self._error_thresholds = {
            ConfigErrorType.LOAD_FAILURE: 3,      # 3次加载失败触发告警
            ConfigErrorType.VALIDATION_FAILURE: 5, # 5次验证失败触发告警
            ConfigErrorType.CACHE_FAILURE: 10     # 10次缓存失败触发告警
        }
        self._recovery_strategies = {
            ConfigErrorType.LOAD_FAILURE: self._handle_load_failure,
            ConfigErrorType.VALIDATION_FAILURE: self._handle_validation_failure,
            ConfigErrorType.CACHE_FAILURE: self._handle_cache_failure
        }

    def handle(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        log_level: str = 'ERROR'
    ) -> None:
        """处理错误

        Args:
            error: 异常实例
            context: 错误上下文
            log_level: 日志级别
        """
        error_context = {**self._log_context, **(context or {})}

        if isinstance(error, ConfigError):
            self._handle_config_error(error, error_context, log_level)
        else:
            self._handle_generic_error(error, error_context, log_level)

    def add_handler(
        self,
        error_type: ConfigErrorType,
        handler: Callable[[ConfigError, Dict[str, Any]], None]
    ) -> None:
        """添加自定义错误处理器

        Args:
            error_type: 错误类型
            handler: 处理函数
        """
        self._handlers[error_type].append(handler)

    def set_error_threshold(self, error_type: ConfigErrorType, threshold: int) -> None:
        """设置错误阈值

        Args:
            error_type: 错误类型
            threshold: 阈值
        """
        self._error_thresholds[error_type] = threshold

    def add_recovery_strategy(
        self,
        error_type: ConfigErrorType,
        strategy: Callable[[ConfigError, Dict[str, Any]], None]
    ) -> None:
        """添加错误恢复策略

        Args:
            error_type: 错误类型
            strategy: 恢复策略函数
        """
        self._recovery_strategies[error_type] = strategy

    def update_log_context(self, **kwargs) -> None:
        """更新日志上下文"""
        self._log_context.update(kwargs)

    def _handle_config_error(
        self,
        error: ConfigError,
        context: Dict[str, Any],
        log_level: str
    ) -> None:
        """处理配置错误

        Args:
            error: 配置错误
            context: 错误上下文
            log_level: 日志级别
        """
        # 记录错误
        self._log_error(error, context, log_level)

        # 更新错误计数
        self._error_counts[error.error_type] += 1

        # 检查是否超过阈值
        if self._check_threshold(error.error_type):
            self._trigger_alert(error, context)

        # 执行错误恢复
        self._try_recover(error, context)

        # 执行自定义处理器
        for handler in self._handlers[error.error_type]:
            try:
                handler(error, context)
            except Exception as e:
                logger.error(f"Error handler failed: {str(e)}", exc_info=True)

    def _handle_generic_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        log_level: str
    ) -> None:
        """处理通用错误

        Args:
            error: 异常
            context: 错误上下文
            log_level: 日志级别
        """
        self._log_error(error, context, log_level)

    def _log_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        log_level: str
    ) -> None:
        """记录错误日志

        Args:
            error: 异常
            context: 错误上下文
            log_level: 日志级别
        """
        log_func = getattr(logger, log_level.lower())
        log_func(
            f"{type(error).__name__}: {str(error)}",
            extra={
                "error_context": context,
                "timestamp": time.time()
            }
        )

    def _check_threshold(self, error_type: ConfigErrorType) -> bool:
        """检查错误是否超过阈值

        Args:
            error_type: 错误类型

        Returns:
            bool: 是否超过阈值
        """
        threshold = self._error_thresholds.get(error_type)
        if threshold is None:
            return False
        return self._error_counts[error_type] >= threshold

    def _trigger_alert(self, error: ConfigError, context: Dict[str, Any]) -> None:
        """触发告警

        Args:
            error: 配置错误
            context: 错误上下文
        """
        logger.critical(
            f"Error threshold exceeded for {error.error_type}",
            extra={
                "error_count": self._error_counts[error.error_type],
                "error_context": context
            }
        )

    def _try_recover(self, error: ConfigError, context: Dict[str, Any]) -> None:
        """尝试错误恢复

        Args:
            error: 配置错误
            context: 错误上下文
        """
        strategy = self._recovery_strategies.get(error.error_type)
        if strategy:
            try:
                strategy(error, context)
            except Exception as e:
                logger.error(f"Recovery failed: {str(e)}", exc_info=True)

    def _handle_load_failure(self, error: ConfigError, context: Dict[str, Any]) -> None:
        """处理加载失败

        Args:
            error: 配置错误
            context: 错误上下文
        """
        logger.info("Attempting to load from backup source")
        # 实现从备份源加载的逻辑

    def _handle_validation_failure(self, error: ConfigError, context: Dict[str, Any]) -> None:
        """处理验证失败

        Args:
            error: 配置错误
            context: 错误上下文
        """
        logger.info("Rolling back to last valid configuration")
        # 实现回滚到上一个有效配置的逻辑

    def _handle_cache_failure(self, error: ConfigError, context: Dict[str, Any]) -> None:
        """处理缓存失败

        Args:
            error: 配置错误
            context: 错误上下文
        """
        logger.info("Clearing and rebuilding cache")
        # 实现清理并重建缓存的逻辑
