"""
error_handler 模块

提供 error_handler 相关功能和接口。
"""

import logging

import prometheus_client as prom
# 创建日志记录器

from ..core.interfaces import IErrorHandler, ErrorSeverity, ErrorCategory, ErrorContext
from threading import Lock
from typing import Dict, Any, Optional, Callable, Type, List
"""
基础设施层 - 错误处理组件 (修复版)

通用错误处理器，提供基本的错误处理功能，兼容测试接口
"""

logger = logging.getLogger(__name__)


class ErrorHandler(IErrorHandler):
    """
    通用错误处理器

    提供基本的错误处理功能，兼容测试接口
    """

    def __init__(self, max_history: int = 1000):
        """初始化错误处理器"""
        self.handlers: Dict[Type[Exception], Callable] = {}
        self.strategies: Dict[str, Callable] = {}
        self.error_stats: Dict[str, int] = {}
        self._error_history: List[Dict[str, Any]] = []
        self._max_history = max_history
        self._lock = Lock()

    def register_handler(self, error_type: Type[Exception], handler: Callable) -> None:
        """注册错误处理器"""
        with self._lock:
            self.handlers[error_type] = handler

    def register_strategy(self, strategy_name: str, strategy: Callable) -> None:
        """注册错误处理策略"""
        with self._lock:
            self.strategies[strategy_name] = strategy

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理错误"""
        error_type = type(error).__name__

        # 更新统计
        with self._lock:
            self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1

        # 创建错误上下文
        error_context = ErrorContext(
            error=error,
            severity=self._determine_severity(error),
            category=self._determine_category(error),
            context=context
        )

        # 记录到历史
        with self._lock:
            self._error_history.append(error_context.to_dict())
            if len(self._error_history) > self._max_history:
                self._error_history.pop(0)

        # 查找处理器
        handler = self.handlers.get(type(error))
        if handler:
            try:
                result = handler(error, context)
                result['handled'] = True
                result['error_context'] = error_context.to_dict()
                return result
            except Exception as handler_error:
                logger.error(f"Handler failed: {handler_error}")

        # 默认处理
        return {
            'handled': False,
            'error_type': error_type,
            'message': str(error),
            'severity': error_context.severity.value,
            'category': error_context.category.value,
            'context': context,
            'error_context': error_context.to_dict()
        }

    def handle(self, error: Exception, context: Optional[Dict[str, Any]] = None,
               strategy: Optional[str] = None) -> Dict[str, Any]:
        """处理错误（别名方法）"""
        if strategy and strategy in self.strategies:
            strategy_func = self.strategies[strategy]
            try:
                result = strategy_func(error, context)
                # 确保返回包含error字段的结果
                if isinstance(result, dict) and "error" not in result:
                    result["error"] = str(error)
                return result
            except Exception as strategy_error:
                logger.error(f"Strategy {strategy} failed: {strategy_error}")

        result = self.handle_error(error, context)
        # 为测试兼容性添加error字段
        if "error" not in result:
            result["error"] = str(error)
        return result

    def get_error_stats(self) -> Dict[str, Any]:
        """获取错误统计"""
        with self._lock:
            return dict(self.error_stats)

    def get_metrics(self) -> Dict[str, Any]:
        """获取指标信息"""
        with self._lock:
            return {
                'total_errors': sum(self.error_stats.values()),
                'total_handled': sum(self.error_stats.values()),  # 简化为等于总错误数
                'errors_by_type': dict(self.error_stats),  # 别名
                'error_types': dict(self.error_stats),
                'registered_handlers': len(self.handlers),
                'registered_strategies': len(self.strategies)
            }

    def clear_stats(self) -> None:
        """清空错误统计"""
        with self._lock:
            self.error_stats.clear()

    def get_registered_handlers(self) -> List[str]:
        """获取已注册的处理器"""
        with self._lock:
            return [str(key) for key in self.handlers.keys()]

    def get_registered_strategies(self) -> List[str]:
        """获取已注册的策略"""
        with self._lock:
            return list(self.strategies.keys())

    def get_error_history(self) -> List[Dict[str, Any]]:
        """获取错误历史"""
        with self._lock:
            return self._error_history.copy()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            return {
                'total_errors': sum(self.error_stats.values()),
                'errors_by_type': dict(self.error_stats),
                'registered_handlers': len(self.handlers),
                'registered_strategies': len(self.strategies),
                'max_history': self._max_history,
                'current_history_size': len(self._error_history)
            }

    def clear_history(self) -> None:
        """清空错误历史"""
        with self._lock:
            self._error_history.clear()

    def _determine_severity(self, error: Exception) -> ErrorSeverity:
        """确定错误严重程度"""
        error_type = type(error).__name__

        if 'Critical' in error_type or 'Fatal' in error_type or 'SystemExit' in error_type:
            return ErrorSeverity.CRITICAL
        elif 'Database' in error_type or 'Network' in error_type or 'Timeout' in error_type:
            return ErrorSeverity.ERROR
        elif 'Validation' in error_type or 'ValueError' in error_type or 'Config' in error_type:
            return ErrorSeverity.WARNING
        else:
            return ErrorSeverity.INFO

    def _determine_category(self, error: Exception) -> ErrorCategory:
        """确定错误类别"""
        error_type = type(error).__name__

        if 'Network' in error_type or 'Connection' in error_type or 'Timeout' in error_type:
            return ErrorCategory.NETWORK
        elif 'Database' in error_type or 'SQL' in error_type:
            return ErrorCategory.DATABASE
        elif 'Config' in error_type or 'Settings' in error_type:
            return ErrorCategory.CONFIGURATION
        elif 'Security' in error_type or 'Auth' in error_type or 'Permission' in error_type:
            return ErrorCategory.SECURITY
        elif 'Performance' in error_type or 'Timeout' in error_type:
            return ErrorCategory.PERFORMANCE
        else:
            return ErrorCategory.UNKNOWN


# 延迟导入Prometheus客户端，提高导入性能
_prometheus_imported = False
_prometheus_client = None


def _get_prometheus_client():
    """延迟导入Prometheus客户端"""
    global _prometheus_imported, _prometheus_client
    if not _prometheus_imported:
        try:
            _prometheus_client = prom
            _prometheus_imported = True
        except ImportError:
            _prometheus_client = None
            _prometheus_imported = True
    return _prometheus_client


# 默认错误处理器实例
_default_error_handler = None
_error_handler_lock = Lock()


def get_default_error_handler() -> ErrorHandler:
    """获取默认错误处理器实例"""
    global _default_error_handler
    if _default_error_handler is None:
        with _error_handler_lock:
            if _default_error_handler is None:
                _default_error_handler = ErrorHandler()
    return _default_error_handler

# 便捷函数


def handle_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """便捷错误处理函数"""
    handler = get_default_error_handler()
    return handler.handle_error(error, context)


def register_error_handler(error_type: Type[Exception], handler: Callable) -> None:
    """注册错误处理器"""
    handler_instance = get_default_error_handler()
    handler_instance.register_handler(error_type, handler)


def register_error_strategy(strategy_name: str, strategy: Callable) -> None:
    """注册错误处理策略"""
    handler_instance = get_default_error_handler()
    handler_instance.register_strategy(strategy_name, strategy)
