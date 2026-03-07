
import time

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Dict, List, Optional, Type, Callable
"""
基础设施层错误处理 - 统一接口定义

定义错误处理系统的核心接口，确保所有组件遵循统一的设计规范。
"""


class ErrorSeverity(Enum):
    """错误严重程度枚举"""
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    FATAL = "fatal"


class ErrorCategory(Enum):
    """错误分类枚举"""
    SYSTEM = "system"
    BUSINESS = "business"
    NETWORK = "network"
    DATABASE = "database"
    CONFIGURATION = "configuration"
    SECURITY = "security"
    PERFORMANCE = "performance"
    UNKNOWN = "unknown"


class IErrorComponent(ABC):
    """错误组件统一接口"""

    @abstractmethod
    def handle_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理错误"""

    @abstractmethod
    def get_error_history(self) -> List[Dict[str, Any]]:
        """获取错误历史"""

    @abstractmethod
    def clear_history(self) -> None:
        """清空错误历史"""

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""


class IErrorHandler(IErrorComponent):
    """错误处理器统一接口"""

    @abstractmethod
    def register_handler(self, error_type: Type[Exception], handler: Callable) -> None:
        """注册错误处理器"""

    @abstractmethod
    def register_strategy(self, strategy_name: str, strategy: Callable) -> None:
        """注册错误处理策略"""

    @abstractmethod
    def get_registered_handlers(self) -> List[str]:
        """获取已注册的处理器"""

    @abstractmethod
    def get_registered_strategies(self) -> List[str]:
        """获取已注册的策略"""


class ICircuitBreaker(ABC):
    """熔断器统一接口"""

    @abstractmethod
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """执行函数调用"""

    @abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """获取状态"""

    @abstractmethod
    def reset(self) -> None:
        """重置熔断器"""

    @abstractmethod
    def trip(self) -> None:
        """触发熔断"""


class IRetryPolicy(ABC):
    """重试策略统一接口"""

    @abstractmethod
    def execute(self, func: Callable, *args, **kwargs) -> Any:
        """执行重试"""

    @abstractmethod
    def get_retry_stats(self) -> Dict[str, Any]:
        """获取重试统计"""

    @abstractmethod
    def reset_stats(self) -> None:
        """重置统计"""


class IExceptionHandler(ABC):
    """异常处理器统一接口"""

    @abstractmethod
    def handle_exception(self, exception: Exception, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """处理异常"""

    @abstractmethod
    def register_exception_handler(self, exception_type: Type[Exception], handler: Callable) -> None:
        """注册异常处理器"""

    @abstractmethod
    def get_exception_stats(self) -> Dict[str, Any]:
        """获取异常统计"""


class IErrorRecovery(ABC):
    """错误恢复统一接口"""

    @abstractmethod
    def recover(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> bool:
        """执行恢复"""

    @abstractmethod
    def get_recovery_strategies(self) -> List[str]:
        """获取恢复策略"""

    @abstractmethod
    def add_recovery_strategy(self, name: str, strategy: Callable) -> None:
        """添加恢复策略"""


class IErrorMonitor(ABC):
    """错误监控统一接口"""

    @abstractmethod
    def record_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """记录错误"""

    @abstractmethod
    def get_error_metrics(self) -> Dict[str, Any]:
        """获取错误指标"""

    @abstractmethod
    def get_error_trends(self, time_range: str) -> Dict[str, Any]:
        """获取错误趋势"""


# 类型别名
ErrorHandlerFunc = Callable[[Exception, Optional[Dict[str, Any]]], Dict[str, Any]]
ExceptionHandlerFunc = Callable[[Exception, Optional[Dict[str, Any]]], Dict[str, Any]]
RecoveryStrategyFunc = Callable[[Exception, Optional[Dict[str, Any]]], bool]
ErrorFilterFunc = Callable[[Exception], bool]


class ErrorContext:
    """错误上下文"""

    def __init__(self,
                 error: Exception,
                 severity: ErrorSeverity = ErrorSeverity.ERROR,
                 category: ErrorCategory = ErrorCategory.UNKNOWN,
                 context: Optional[Dict[str, Any]] = None,
                 timestamp: Optional[float] = None,
                 boundary_check: Optional[List[Dict[str, Any]]] = None):
        self.error = error
        self.severity = severity
        self.category = category
        self.context = context or {}
        self.timestamp = timestamp or self._get_timestamp()
        self.boundary_check = boundary_check or []

    def _get_timestamp(self) -> float:
        """获取时间戳"""
        return time.time()

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {
            'error_type': type(self.error).__name__,
            'message': str(self.error),
            'severity': self.severity.value,
            'category': self.category.value,
            'context': self.context,
            'timestamp': self.timestamp
        }
        if self.boundary_check:
            result['boundary_check'] = self.boundary_check
        return result


class ErrorResult:
    """错误处理结果"""

    def __init__(self,
                 handled: bool = False,
                 result: Optional[Any] = None,
                 error_context: Optional[ErrorContext] = None,
                 recovery_attempted: bool = False,
                 recovery_successful: bool = False):
        self.handled = handled
        self.result = result
        self.error_context = error_context
        self.recovery_attempted = recovery_attempted
        self.recovery_successful = recovery_successful

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'handled': self.handled,
            'result': self.result,
            'error_context': self.error_context.to_dict() if self.error_context else None,
            'recovery_attempted': self.recovery_attempted,
            'recovery_successful': self.recovery_successful
        }
