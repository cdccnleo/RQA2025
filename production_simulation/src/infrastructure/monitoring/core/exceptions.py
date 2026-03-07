
from typing import Any, Dict, Optional
"""
监控管理模块异常处理
Monitoring Management Module Exception Handling

定义监控管理相关的异常类和错误处理机制
"""


class MonitoringException(Exception):
    """监控基础异常类"""

    def __init__(self, message: str, monitor_type: Optional[str] = None,
                 component_name: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.monitor_type = monitor_type
        self.component_name = component_name
        self.details = details or {}
        self.message = message


class MonitorConfigurationError(MonitoringException):
    """监控配置错误"""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 expected_value: Any = None, actual_value: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.expected_value = expected_value
        self.actual_value = actual_value


class MetricCollectionError(MonitoringException):
    """指标收集错误"""

    def __init__(self, message: str, metric_name: Optional[str] = None,
                 collection_method: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.metric_name = metric_name
        self.collection_method = collection_method


class AlertProcessingError(MonitoringException):
    """告警处理错误"""

    def __init__(self, message: str, alert_id: Optional[str] = None,
                 alert_rule: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.alert_id = alert_id
        self.alert_rule = alert_rule


class NotificationError(MonitoringException):
    """通知错误"""

    def __init__(self, message: str, notification_type: Optional[str] = None,
                 recipient: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.notification_type = notification_type
        self.recipient = recipient


class HealthCheckError(MonitoringException):
    """健康检查错误"""

    def __init__(self, message: str, check_type: Optional[str] = None,
                 check_target: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.check_type = check_type
        self.check_target = check_target


class ThresholdExceededError(MonitoringException):
    """阈值超限错误"""

    def __init__(self, message: str, metric_name: Optional[str] = None,
                 threshold_value: Optional[float] = None,
                 actual_value: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.metric_name = metric_name
        self.threshold_value = threshold_value
        self.actual_value = actual_value


class MonitorConnectionError(MonitoringException):
    """监控连接错误"""

    def __init__(self, message: str, target_host: Optional[str] = None,
                 target_port: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.target_host = target_host
        self.target_port = target_port


class DataProcessingError(MonitoringException):
    """数据处理错误"""

    def __init__(self, message: str, data_type: Optional[str] = None,
                 processing_step: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.data_type = data_type
        self.processing_step = processing_step


class StorageError(MonitoringException):
    """存储错误"""

    def __init__(self, message: str, storage_type: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.storage_type = storage_type
        self.operation = operation


class AlertRuleError(MonitoringException):
    """告警规则错误"""

    def __init__(self, message: str, rule_id: Optional[str] = None,
                 rule_condition: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.rule_id = rule_id
        self.rule_condition = rule_condition


class PerformanceMonitorError(MonitoringException):
    """性能监控错误"""

    def __init__(self, message: str, performance_metric: Optional[str] = None,
                 expected_performance: Optional[float] = None,
                 actual_performance: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.performance_metric = performance_metric
        self.expected_performance = expected_performance
        self.actual_performance = actual_performance


class DisasterRecoveryError(MonitoringException):
    """灾难恢复错误"""

    def __init__(self, message: str, recovery_phase: Optional[str] = None,
                 failure_point: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.recovery_phase = recovery_phase
        self.failure_point = failure_point


class ComponentMonitorError(MonitoringException):
    """组件监控错误"""

    def __init__(self, message: str, component_type: Optional[str] = None,
                 component_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.component_type = component_type
        self.component_id = component_id


class ContinuousMonitoringError(MonitoringException):
    """连续监控错误"""

    def __init__(self, message: str, monitoring_cycle: Optional[int] = None,
                 failure_cycle: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.monitoring_cycle = monitoring_cycle
        self.failure_cycle = failure_cycle


class ExceptionMonitorError(MonitoringException):
    """异常监控错误"""

    def __init__(self, message: str, exception_type: Optional[str] = None,
                 exception_count: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.exception_type = exception_type
        self.exception_count = exception_count


class ProductionMonitorError(MonitoringException):
    """生产环境监控错误"""

    def __init__(self, message: str, environment: Optional[str] = None,
                 production_metric: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.environment = environment
        self.production_metric = production_metric


class LoggerPoolMonitorError(MonitoringException):
    """日志池监控错误"""

    def __init__(self, message: str, pool_name: Optional[str] = None,
                 pool_size: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.pool_name = pool_name
        self.pool_size = pool_size


class SystemMonitorError(MonitoringException):
    """系统监控错误"""

    def __init__(self, message: str, system_component: Optional[str] = None,
                 system_metric: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.system_component = system_component
        self.system_metric = system_metric

# ============================================================================
# 异常处理装饰器
# ============================================================================


def handle_monitoring_exception(operation: str = "monitoring_operation"):
    """
    监控异常处理装饰器

    Args:
        operation: 操作名称，用于日志记录

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MonitoringException:
                # 重新抛出监控异常，不做额外处理
                raise
            except Exception as e:
                # 将其他异常包装为监控异常
                error_msg = f"{operation} 失败: {str(e)}"
                raise MonitoringException(error_msg, operation=operation,
                                          details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_metric_collection_exception(metric_name: str = "unknown_metric",
                                       collection_method: str = "unknown"):
    """
    指标收集异常处理装饰器

    Args:
        metric_name: 指标名称
        collection_method: 收集方法

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except MetricCollectionError:
                raise
            except Exception as e:
                error_msg = f"指标收集失败 {metric_name}: {str(e)}"
                raise MetricCollectionError(error_msg, metric_name=metric_name,
                                            collection_method=collection_method,
                                            details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_alert_processing_exception(alert_id: str = "unknown",
                                      alert_rule: str = "unknown"):
    """
    告警处理异常处理装饰器

    Args:
        alert_id: 告警ID
        alert_rule: 告警规则

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except AlertProcessingError:
                raise
            except Exception as e:
                error_msg = f"告警处理失败 {alert_id}: {str(e)}"
                raise AlertProcessingError(error_msg, alert_id=alert_id,
                                           alert_rule=alert_rule,
                                           details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_health_check_exception(check_type: str = "unknown",
                                  check_target: str = "unknown"):
    """
    健康检查异常处理装饰器

    Args:
        check_type: 检查类型
        check_target: 检查目标

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except HealthCheckError:
                raise
            except Exception as e:
                error_msg = f"健康检查失败 {check_type} -> {check_target}: {str(e)}"
                raise HealthCheckError(error_msg, check_type=check_type,
                                       check_target=check_target,
                                       details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_threshold_check_exception(metric_name: str = "unknown",
                                     threshold: float = 0.0):
    """
    阈值检查异常处理装饰器

    Args:
        metric_name: 指标名称
        threshold: 阈值

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ThresholdExceededError:
                raise
            except Exception as e:
                error_msg = f"阈值检查失败 {metric_name}: {str(e)}"
                raise MonitoringException(error_msg, metric_name=metric_name,
                                          details={"threshold": threshold,
                                                   "original_error": str(e)}) from e
        return wrapper
    return decorator
