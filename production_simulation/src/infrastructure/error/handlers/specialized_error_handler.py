"""
specialized_error_handler 模块

提供 specialized_error_handler 相关功能和接口。
"""

import logging
import threading
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List, Callable, Type

from ..core.interfaces import IErrorHandler, ErrorSeverity, ErrorCategory, ErrorContext
from .connection_error_handler import ConnectionErrorHandler
from .io_error_handler import IOErrorHandler
from .retry_manager import RetryManager, RetryConfig, RetryStrategy

# 这些是Python标准库的异常类型，不需要导入
# ConnectionError, TimeoutError, IOError, OSError 都是内置异常

"""
基础设施层 - 专用错误处理器

重构后的专用错误处理器，使用组合模式将不同职责分离到专门的处理器中。
主要职责是协调各个专门的处理器，而不是直接处理所有错误类型。
"""

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """失败类型枚举"""
    ARCHIVE_WRITE_FAILED = "archive_write_failed"
    ARCHIVE_READ_FAILED = "archive_read_failed"
    ARCHIVE_DELETE_FAILED = "archive_delete_failed"
    ARCHIVE_LIST_FAILED = "archive_list_failed"
    ARCHIVE_COMPRESSION_FAILED = "archive_compression_failed"
    ARCHIVE_EXTRACTION_FAILED = "archive_extraction_failed"


class InfluxDBErrorType(Enum):
    """InfluxDB错误类型"""
    CONNECTION_FAILED = "connection_failed"
    WRITE_FAILED = "write_failed"
    QUERY_FAILED = "query_failed"
    AUTHENTICATION_FAILED = "authentication_failed"
    TIMEOUT = "timeout"


# RetryStrategy 和 RetryConfig 现在从 retry_manager 模块导入


@dataclass
class FailureContext:
    """失败上下文"""
    failure_type: FailureType
    operation: str
    resource_path: str
    error_details: Dict[str, Any]
    timestamp: float
    retry_count: int = 0


@dataclass
class ErrorProcessingParams:
    """错误处理参数对象 - 用于简化方法签名"""
    error: Exception
    context: Optional[Dict[str, Any]]
    error_type: str
    error_context: ErrorContext


@dataclass 
class HandlerResultParams:
    """处理器结果参数对象 - 用于简化结果处理方法签名"""
    result: Dict[str, Any]
    error: Exception
    error_type: str
    error_context: ErrorContext
    context: Optional[Dict[str, Any]]


class SpecializedErrorHandler(IErrorHandler):
    """
    专用错误处理器 - 重构版

    使用组合模式将不同职责分离到专门的处理器中：
    - ConnectionErrorHandler: 处理连接和网络错误
    - IOErrorHandler: 处理IO和系统错误  
    - RetryManager: 管理重试逻辑
    
    主要职责是协调各个专门的处理器，简化代码结构。
    """

    def __init__(self, max_history: int = 1000):
        self._handlers: Dict[Type[Exception], Callable] = {}
        self._strategies: Dict[str, Callable] = {}
        self._error_history: List[Dict[str, Any]] = []
        self._max_history = max_history
        
        # 使用专门的处理器组件
        self._connection_handler = ConnectionErrorHandler()
        self._io_handler = IOErrorHandler()
        self._retry_manager = RetryManager()
        
        # 归档失败统计
        self._failure_stats: Dict[str, int] = {}

        self._register_default_handlers()

    def _register_default_handlers(self):
        """注册默认错误处理器 - 使用专门的处理器组件"""
        # 连接相关错误 - 委托给ConnectionErrorHandler
        self.register_handler(ConnectionError, self._connection_handler.handle_connection_error)  # type: ignore
        self.register_handler(TimeoutError, self._connection_handler.handle_timeout_error)  # type: ignore

        # IO和系统错误 - 委托给IOErrorHandler
        self.register_handler(IOError, self._io_handler.handle_io_error)  # type: ignore
        self.register_handler(OSError, self._io_handler.handle_os_error)  # type: ignore

    def register_handler(self, error_type: Type[Exception], handler: Callable) -> None:
        """注册错误处理器"""
        self._handlers[error_type] = handler

    def register_strategy(self, strategy_name: str, strategy: Callable) -> None:
        """注册错误处理策略"""
        self._strategies[strategy_name] = strategy

    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理错误 - 重构后减少复杂度"""
        error_type = self._determine_error_type_name(error)
        error_context = self._create_error_context(error, context)
        
        self._record_error_to_history(error_context, error_type)
        
        # 创建参数对象
        processing_params = ErrorProcessingParams(
            error=error,
            context=context,
            error_type=error_type,
            error_context=error_context
        )
        
        # 尝试使用处理器处理
        handler_result = self._try_handler_processing_with_params(processing_params)
        if handler_result is not None:
            return handler_result
        
        # 默认处理
        return self._create_default_result_with_params(processing_params)

    def _try_handler_processing_with_params(self, params: ErrorProcessingParams) -> Optional[Dict[str, Any]]:
        """使用参数对象尝试处理器处理"""
        handler = self._find_handler(params.error)
        if not handler:
            return None
            
        try:
            result = handler(params.error, params.context)
            self._enrich_handler_result_with_params(
                HandlerResultParams(
                    result=result,
                    error=params.error,
                    error_type=params.error_type,
                    error_context=params.error_context,
                    context=params.context
                )
            )
            self._handle_retry_if_needed(result, params.error, params.context)
            return result
        except Exception as handler_error:
            logger.error(f"Handler failed: {handler_error}")
            return None

    def _create_default_result_with_params(self, params: ErrorProcessingParams) -> Dict[str, Any]:
        """使用参数对象创建默认处理结果"""
        return {
            'handled': False,
            'error_type': params.error_type,
            'message': str(params.error),
            'severity': params.error_context.severity.value,
            'category': params.error_context.category.value,
            'context': params.context,
            'error_context': params.error_context.to_dict()
        }

    def _determine_error_type_name(self, error: Exception) -> str:
        """确定错误类型名称"""
        error_type_name = type(error).__name__
        
        if error_type_name == 'OSError':
            # 根据创建方式判断是IOError还是OSError
            if isinstance(error, IOError):
                return 'IOError'
            else:
                return 'OSError'
        else:
            return error_type_name

    def _create_error_context(self, error: Exception, context: Optional[Dict[str, Any]]) -> ErrorContext:
        """创建错误上下文"""
        return ErrorContext(
            error=error,
            severity=self._classify_severity(error),
            category=self._classify_category(error),
            context=context
        )

    def _record_error_to_history(self, error_context: ErrorContext, error_type: str) -> None:
        """记录错误到历史"""
        history_entry = error_context.to_dict()
        history_entry['error_type'] = error_type  # 覆盖默认的error_type
        with self._lock:
            self._error_history.append(history_entry)
            if len(self._error_history) > self._max_history:
                self._error_history.pop(0)

    def _try_handler_processing(self, error: Exception, context: Optional[Dict[str, Any]], 
                              error_type: str, error_context: ErrorContext) -> Optional[Dict[str, Any]]:
        """尝试使用处理器处理错误 - 保持向后兼容性"""
        params = ErrorProcessingParams(
            error=error,
            context=context,
            error_type=error_type,
            error_context=error_context
        )
        return self._try_handler_processing_with_params(params)

    def _find_handler(self, error: Exception):
        """查找合适的处理器"""
        handler = self._handlers.get(type(error))
        if not handler and isinstance(error, OSError):
            handler = self._handlers.get(OSError)
        return handler

    def _enrich_handler_result_with_params(self, params: HandlerResultParams) -> None:
        """使用参数对象丰富处理器结果"""
        params.result.setdefault('handled', True)
        params.result.setdefault('error_type', params.error_type)
        params.result.setdefault('message', str(params.error))
        params.result.setdefault('severity', params.error_context.severity.value)
        params.result.setdefault('category', params.error_context.category.value)
        params.result.setdefault('context', params.context)
        params.result['error_context'] = params.error_context.to_dict()

    def _enrich_handler_result(self, result: Dict[str, Any], error: Exception, 
                             error_type: str, error_context: ErrorContext, 
                             context: Optional[Dict[str, Any]]) -> None:
        """丰富处理器结果 - 保持向后兼容性"""
        params = HandlerResultParams(
            result=result,
            error=error,
            error_type=error_type,
            error_context=error_context,
            context=context
        )
        self._enrich_handler_result_with_params(params)

    def _handle_retry_if_needed(self, result: Dict[str, Any], error: Exception, 
                              context: Optional[Dict[str, Any]]) -> None:
        """如果需要，处理重试"""
        if 'retry_config' in result:
            retry_config = result['retry_config']
            if isinstance(retry_config, dict):
                retry_config = RetryConfig(**retry_config)
            retry_result = self._retry_manager.execute_retry(retry_config, error, context)
            result['retry_result'] = retry_result

    def _create_default_result(self, error: Exception, error_type: str, 
                             error_context: ErrorContext, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """创建默认处理结果 - 保持向后兼容性"""
        params = ErrorProcessingParams(
            error=error,
            context=context,
            error_type=error_type,
            error_context=error_context
        )
        return self._create_default_result_with_params(params)

    def _classify_severity(self, error: Exception) -> ErrorSeverity:
        """分类错误严重程度"""
        error_type = type(error).__name__

        if 'Critical' in error_type or 'Fatal' in error_type or 'KeyboardInterrupt' in error_type:
            return ErrorSeverity.CRITICAL
        elif 'Connection' in error_type or 'Timeout' in error_type or 'InfluxDB' in error_type:
            return ErrorSeverity.ERROR
        elif 'IOError' in error_type or 'OSError' in error_type:
            return ErrorSeverity.WARNING
        else:
            return ErrorSeverity.INFO

    def _classify_category(self, error: Exception) -> ErrorCategory:
        """分类错误类别"""
        error_type = type(error).__name__

        if 'Connection' in error_type or 'Network' in error_type or 'Timeout' in error_type:
            return ErrorCategory.NETWORK
        elif 'InfluxDB' in error_type or 'Database' in error_type:
            return ErrorCategory.DATABASE
        elif 'IO' in error_type or 'OS' in error_type:
            return ErrorCategory.SYSTEM
        else:
            return ErrorCategory.UNKNOWN

    def add_retry_config(self, name: str, config: RetryConfig) -> None:
        """添加重试配置 - 委托给重试管理器"""
        self._retry_manager.add_retry_config(name, config)

    def get_retry_config(self, name: str) -> Optional[RetryConfig]:
        """获取重试配置 - 委托给重试管理器"""
        return self._retry_manager.get_retry_config(name)

    def get_error_history(self) -> List[Dict[str, Any]]:
        """获取错误历史"""
        return self._error_history.copy()

    def clear_history(self) -> None:
        """清空错误历史"""
        self._error_history.clear()

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_errors = len(self._error_history)
        severity_stats = {}
        category_stats = {}

        for error_info in self._error_history:
            severity = error_info.get('severity', 'unknown')
            category = error_info.get('category', 'unknown')

            severity_stats[severity] = severity_stats.get(severity, 0) + 1
            category_stats[category] = category_stats.get(category, 0) + 1

        return {
            'total_errors': total_errors,
            'severity_distribution': severity_stats,
            'category_distribution': category_stats,
            'registered_handlers': len(self._handlers),
            'registered_strategies': len(self._strategies),
            'retry_configs': len(self._retry_manager._retry_configs),
            'failure_stats': self._failure_stats.copy(),
            'max_history': self._max_history,
            'current_history_size': len(self._error_history)
        }

    def get_registered_handlers(self) -> List[str]:
        """获取已注册的处理器"""
        return [cls.__name__ for cls in self._handlers.keys()]

    def get_registered_strategies(self) -> List[str]:
        """获取已注册的策略"""
        return list(self._strategies.keys())

    # 为了兼容接口，需要添加_lock属性
    @property
    def _lock(self):
        """获取锁对象"""
        if not hasattr(self, '__lock'):
            self.__lock = threading.Lock()
        return self.__lock
