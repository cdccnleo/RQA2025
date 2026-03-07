"""
exceptions 模块

提供 exceptions 相关功能和接口。
"""

import logging


from typing import Dict, Any, Optional
"""
基础设施异常模块
定义基础设施层使用的异常类
"""


class InfrastructureError(Exception):
    """基础设施基础异常类"""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "INFRASTRUCTURE_ERROR"
        self.details = details or {}

        # 自动记录异常信息
        try:
            logging.error(
                f"基础设施异常: {self.error_code} - {message}",
                extra={"error_code": self.error_code,
                       "error_type": self.__class__.__name__, "details": self.details},
            )
        except Exception:
            # 避免日志记录失败导致程序崩溃
            pass

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message,
            "details": self.details,
        }


class ConfigurationError(InfrastructureError):
    """配置相关异常"""

    def __init__(self, message: str, config_key: Optional[str] = None):

        super().__init__(message, "CONFIG_ERROR", {"config_key": config_key})


class DataProcessingError(InfrastructureError):
    """数据处理异常"""

    def __init__(
        self,
        message: str,
        data_source: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(
            message,
            "DATA_PROCESSING_ERROR",
            {"data_source": data_source, "operation": operation},
        )


class ConnectionError(InfrastructureError):
    """连接相关异常"""

    def __init__(self, message: str, host: Optional[str] = None, port: Optional[int] = None):
        super().__init__(message, "CONNECTION_ERROR", {"host": host, "port": port})


class ServiceDiscoveryError(InfrastructureError):
    """服务发现异常"""

    def __init__(self, message: str, service_name: Optional[str] = None):

        super().__init__(message, "SERVICE_DISCOVERY_ERROR", {"service_name": service_name})


class HealthCheckError(InfrastructureError):
    """健康检查异常"""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        check_type: Optional[str] = None,
    ):
        super().__init__(
            message,
            "HEALTH_CHECK_ERROR",
            {"service_name": service_name, "check_type": check_type},
        )


class EventBusError(InfrastructureError):
    """事件总线异常"""

    def __init__(
        self,
        message: str,
        event_type: Optional[str] = None,
        subscriber: Optional[str] = None,
    ):
        super().__init__(
            message,
            "EVENT_BUS_ERROR",
            {"event_type": event_type, "subscriber": subscriber},
        )


class MonitoringError(InfrastructureError):
    """监控相关异常"""

    def __init__(self, message: str, metric_name: Optional[str] = None):

        super().__init__(message, "MONITORING_ERROR", {"metric_name": metric_name})


class LoggingError(InfrastructureError):
    """日志相关异常"""

    def __init__(self, message: str, logger_name: Optional[str] = None):

        super().__init__(message, "LOGGING_ERROR", {"logger_name": logger_name})


class DataLoaderError(InfrastructureError):
    """数据加载器异常"""

    def __init__(
        self,
        message: str,
        loader_name: Optional[str] = None,
        data_source: Optional[str] = None,
    ):
        super().__init__(
            message,
            "DATA_LOADER_ERROR",
            {"loader_name": loader_name, "data_source": data_source},
        )


class CacheError(InfrastructureError):
    """缓存相关异常"""

    def __init__(
        self,
        message: str,
        cache_key: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message, "CACHE_ERROR", {"cache_key": cache_key, "operation": operation})


class SecurityError(InfrastructureError):
    """安全相关异常"""

    def __init__(self, message: str, security_context: Optional[str] = None):

        super().__init__(message, "SECURITY_ERROR", {"security_context": security_context})


class ResourceLimitError(InfrastructureError):
    """资源限制异常"""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        current_usage: Optional[float] = None,
        limit: Optional[float] = None,
    ):
        super().__init__(
            message,
            "RESOURCE_LIMIT_ERROR",
            {
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
            },
        )


class ValidationError(InfrastructureError):
    """验证异常"""

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message, "VALIDATION_ERROR", {"field": field, "value": value})


class TimeoutError(InfrastructureError):
    """超时异常"""

    def __init__(self, message: str, timeout_seconds: Optional[float] = None):

        super().__init__(message, "TIMEOUT_ERROR", {"timeout_seconds": timeout_seconds})


class DataVersionError(InfrastructureError):
    """数据版本异常"""

    def __init__(
        self,
        message: str,
        version: Optional[str] = None,
        operation: Optional[str] = None,
    ):
        super().__init__(message, "DATA_VERSION_ERROR", {
            "version": version, "operation": operation})


class DependencyError(InfrastructureError):
    """依赖关系异常"""

    def __init__(self, message: str, dependency_name: Optional[str] = None):

        super().__init__(message, "DEPENDENCY_ERROR", {"dependency_name": dependency_name})
