"""
RQA2025系统统一异常处理框架
Unified Exception Handling Framework for RQA2025 System

定义系统的统一异常类体系，实现层级化异常管理和统一处理机制
"""

from .core_exceptions import (
    ConnectionError
)
from typing import Any, Dict, Optional, List, Callable
from datetime import datetime
import logging
import json
from functools import wraps

# 配置日志
logger = logging.getLogger(__name__)


class RQA2025Exception(Exception):
    """
    RQA2025系统基础异常类

    所有系统异常的根类，提供统一的异常信息结构和处理机制
    """

    def __init__(self,
                 message: str,
                 error_code: int = -1,
                 error_type: str = "UNKNOWN",
                 context: Optional[Dict[str, Any]] = None,
                 severity: str = "ERROR"):
        """
        初始化异常

        Args:
            message: 异常消息
            error_code: 错误代码
            error_type: 错误类型
            context: 异常上下文信息
            severity: 严重程度 (CRITICAL, ERROR, WARNING, INFO)
        """
        super().__init__(message)

        self.message = message
        self.error_code = error_code
        self.error_type = error_type
        self.context = context or {}
        self.severity = severity.upper()
        self.timestamp = datetime.now().isoformat()
        self.stack_trace = self._get_stack_trace()

        # 自动记录异常
        self._log_exception()

        # 触发监控和告警检查
        if global_exception_config.get_config('monitoring_enabled'):
            global_exception_monitor.check_alerts(self)

    def _get_stack_trace(self) -> str:
        """获取堆栈跟踪信息"""
        import traceback
        return ''.join(traceback.format_exception(*traceback.sys.exc_info()))

    def _log_exception(self):
        """记录异常到日志"""
        # 创建可序列化的上下文
        serializable_context = {}
        for key, value in self.context.items():
            try:
                json.dumps(value)  # 测试是否可序列化
                serializable_context[key] = value
            except (TypeError, ValueError):
                # 如果不可序列化，转换为字符串
                serializable_context[key] = str(value)

        log_data = {
            'timestamp': self.timestamp,
            'error_type': self.error_type,
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity,
            'context': serializable_context
        }

        try:
            log_message = json.dumps(log_data, ensure_ascii=False)
        except (TypeError, ValueError):
            # 如果仍然无法序列化，使用简化版本
            log_message = f"{self.error_type}:{self.error_code} {self.message}"

        if self.severity == 'CRITICAL':
            logger.critical(f"Critical Exception: {log_message}")
        elif self.severity == 'ERROR':
            logger.error(f"Exception: {log_message}")
        elif self.severity == 'WARNING':
            logger.warning(f"Warning Exception: {log_message}")
        else:
            logger.info(f"Info Exception: {log_message}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_type': self.error_type,
            'error_code': self.error_code,
            'message': self.message,
            'severity': self.severity,
            'timestamp': self.timestamp,
            'context': self.context,
            'stack_trace': self.stack_trace
        }

    def to_json(self) -> str:
        """转换为JSON字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self):
        """字符串表示"""
        context_str = f" [{self.context}]" if self.context else ""
        return f"[{self.error_type}:{self.error_code}] {self.message}{context_str}"


# ==================== 业务层异常 ====================

class BusinessException(RQA2025Exception):
    """业务层异常基类"""


class ValidationError(BusinessException):
    """数据验证异常"""

    def __init__(self, message: str, field: str = None, value: Any = None):
        super().__init__(
            message=message,
            error_code=1001,
            error_type="VALIDATION_ERROR",
            context={'field': field, 'value': str(value)[:100] if value is not None else None},
            severity="WARNING"
        )


class BusinessLogicError(BusinessException):
    """业务逻辑异常"""

    def __init__(self, message: str, operation: str = None, entity_id: str = None):
        super().__init__(
            message=message,
            error_code=1002,
            error_type="BUSINESS_LOGIC_ERROR",
            context={'operation': operation, 'entity_id': entity_id}
        )


class WorkflowError(BusinessException):
    """工作流异常"""

    def __init__(self, message: str, workflow_id: str = None, step: str = None):
        super().__init__(
            message=message,
            error_code=1003,
            error_type="WORKFLOW_ERROR",
            context={'workflow_id': workflow_id, 'step': step}
        )


class TradingError(BusinessException):
    """交易异常"""

    def __init__(self, message: str, order_id: str = None, symbol: str = None):
        super().__init__(
            message=message,
            error_code=1004,
            error_type="TRADING_ERROR",
            context={'order_id': order_id, 'symbol': symbol}
        )


class RiskError(BusinessException):
    """风险控制异常"""

    def __init__(self, message: str, risk_type: str = None, threshold: float = None):
        super().__init__(
            message=message,
            error_code=1005,
            error_type="RISK_ERROR",
            context={'risk_type': risk_type, 'threshold': threshold}
        )


class StrategyError(BusinessException):
    """策略异常"""

    def __init__(self, message: str, strategy_id: str = None, signal: str = None):
        super().__init__(
            message=message,
            error_code=1006,
            error_type="STRATEGY_ERROR",
            context={'strategy_id': strategy_id, 'signal': signal}
        )


# ==================== 基础设施层异常 ====================

class InfrastructureException(RQA2025Exception):
    """基础设施层异常基类"""


class ConfigurationError(InfrastructureException):
    """配置异常"""

    def __init__(self, message: str, config_key: str = None, config_file: str = None):
        super().__init__(
            message=message,
            error_code=2001,
            error_type="CONFIGURATION_ERROR",
            context={'config_key': config_key, 'config_file': config_file}
        )


class CacheError(InfrastructureException):
    """缓存异常"""

    def __init__(self, message: str, cache_key: str = None, cache_type: str = None):
        super().__init__(
            message=message,
            error_code=2002,
            error_type="CACHE_ERROR",
            context={'cache_key': cache_key, 'cache_type': cache_type}
        )


class LoggingError(InfrastructureException):
    """日志异常"""

    def __init__(self, message: str, log_file: str = None, log_level: str = None):
        super().__init__(
            message=message,
            error_code=2003,
            error_type="LOGGING_ERROR",
            context={'log_file': log_file, 'log_level': log_level}
        )


class MonitoringError(InfrastructureException):
    """监控异常"""

    def __init__(self, message: str, metric_name: str = None, metric_value: Any = None):
        super().__init__(
            message=message,
            error_code=2004,
            error_type="MONITORING_ERROR",
            context={'metric_name': metric_name, 'metric_value': metric_value}
        )


class DatabaseError(InfrastructureException):
    """数据库异常"""

    def __init__(self, message: str, db_name: str = None, operation: str = None, query: str = None):
        super().__init__(
            message=message,
            error_code=2005,
            error_type="DATABASE_ERROR",
            context={'db_name': db_name, 'operation': operation,
                     'query': query[:100] if query else None}
        )


class QueryError(DatabaseError):
    """查询异常"""

    def __init__(self, message: str, db_name: str = None, query: str = None):
        super().__init__(
            message=message,
            error_code=2006,
            error_type="QUERY_ERROR",
            context={'db_name': db_name, 'query': query[:100] if query else None}
        )


class ConnectionError(InfrastructureException):
    """连接异常"""

    def __init__(self, message: str, host: str = None, port: int = None, timeout: float = None):
        super().__init__(
            message=message,
            error_code=2007,
            error_type="CONNECTION_ERROR",
            context={'host': host, 'port': port, 'timeout': timeout}
        )


class NetworkError(InfrastructureException):
    """网络异常"""

    def __init__(self, message: str, endpoint: str = None, status_code: int = None):
        super().__init__(
            message=message,
            error_code=2006,
            error_type="NETWORK_ERROR",
            context={'endpoint': endpoint, 'status_code': status_code}
        )


class ResourceError(InfrastructureException):
    """资源异常"""

    def __init__(self, message: str, resource_type: str = None, current_usage: float = None, limit: float = None):
        super().__init__(
            message=message,
            error_code=2007,
            error_type="RESOURCE_ERROR",
            context={'resource_type': resource_type, 'current_usage': current_usage, 'limit': limit}
        )


class FileSystemError(InfrastructureException):
    """文件系统异常"""

    def __init__(self, message: str, file_path: str = None, operation: str = None):
        super().__init__(
            message=message,
            error_code=2008,
            error_type="FILE_SYSTEM_ERROR",
            context={'file_path': file_path, 'operation': operation}
        )


class HealthCheckError(InfrastructureException):
    """健康检查异常"""

    def __init__(self, message: str, check_target: str = None, check_type: str = None):
        super().__init__(
            message=message,
            error_code=2009,
            error_type="HEALTH_CHECK_ERROR",
            context={'check_target': check_target, 'check_type': check_type},
            severity="WARNING"
        )


# ==================== 系统层异常 ====================

class SystemException(RQA2025Exception):
    """系统层异常基类"""


class SecurityError(SystemException):
    """安全异常"""

    def __init__(self, message: str, user_id: str = None, permission: str = None):
        super().__init__(
            message=message,
            error_code=3001,
            error_type="SECURITY_ERROR",
            context={'user_id': user_id, 'permission': permission},
            severity="CRITICAL"
        )


class PerformanceError(SystemException):
    """性能异常"""

    def __init__(self, message: str, metric_name: str = None, threshold: float = None, actual_value: float = None):
        super().__init__(
            message=message,
            error_code=3002,
            error_type="PERFORMANCE_ERROR",
            context={'metric_name': metric_name,
                     'threshold': threshold, 'actual_value': actual_value}
        )


class ConcurrencyError(SystemException):
    """并发异常"""

    def __init__(self, message: str, resource: str = None, thread_count: int = None):
        super().__init__(
            message=message,
            error_code=3003,
            error_type="CONCURRENCY_ERROR",
            context={'resource': resource, 'thread_count': thread_count}
        )


class AsyncError(SystemException):
    """异步处理异常"""

    def __init__(self, message: str, task_id: str = None, coroutine_name: str = None):
        super().__init__(
            message=message,
            error_code=3004,
            error_type="ASYNC_ERROR",
            context={'task_id': task_id, 'coroutine_name': coroutine_name}
        )


# ==================== 外部服务异常 ====================

class ExternalServiceException(RQA2025Exception):
    """外部服务异常基类"""


class ThirdPartyAPIError(ExternalServiceException):
    """第三方API异常"""

    def __init__(self, message: str, api_name: str = None, endpoint: str = None, response_code: int = None):
        super().__init__(
            message=message,
            error_code=4001,
            error_type="THIRD_PARTY_API_ERROR",
            context={'api_name': api_name, 'endpoint': endpoint, 'response_code': response_code}
        )


class DataSourceError(ExternalServiceException):
    """数据源异常"""

    def __init__(self, message: str, data_source: str = None, data_type: str = None):
        super().__init__(
            message=message,
            error_code=4002,
            error_type="DATA_SOURCE_ERROR",
            context={'data_source': data_source, 'data_type': data_type}
        )


# ==================== 异常处理装饰器和策略 ====================

class ExceptionHandler:
    """异常处理器"""

    def __init__(self, service_name: str = "unknown"):
        self.service_name = service_name
        self.handlers: Dict[type, Callable] = {}

    def register_handler(self, exception_type: type, handler: Callable):
        """注册异常处理器"""
        self.handlers[exception_type] = handler

    def handle_exception(self, exception: Exception) -> Any:
        """处理异常"""
        for exc_type, handler in self.handlers.items():
            if isinstance(exception, exc_type):
                return handler(exception)

        # 默认处理
        if isinstance(exception, RQA2025Exception):
            raise exception
        else:
            # 包装未知异常
            raise RQA2025Exception(
                f"未知异常: {str(exception)}",
                error_code=9999,
                error_type="UNKNOWN_ERROR",
                context={'original_exception': type(exception).__name__}
            ) from exception


class ExceptionHandlingStrategy:
    """异常处理策略"""

    def __init__(self,
                 service_name: str = "unknown",
                 log_level: str = "error",
                 re_raise: bool = True,
                 enable_metrics: bool = True,
                 enable_alerts: bool = False):
        self.service_name = service_name
        self.log_level = log_level
        self.re_raise = re_raise
        self.enable_metrics = enable_metrics
        self.enable_alerts = enable_alerts

        # 异常类型映射
        self.exception_mappings = {
            # 网络相关
            (ConnectionError, TimeoutError, OSError): NetworkError,

            # 文件系统相关
            (IOError, OSError): FileSystemError,

            # 数据库相关
            # (这些通常需要更具体的处理)

            # 资源相关
            MemoryError: lambda e: ResourceError(f"内存不足: {str(e)}", resource_type="memory"),

            # 并发相关
            # (这些通常需要更具体的处理)

            # 安全相关
            # (这些通常需要更具体的处理)
        }

    def map_exception(self, exception: Exception) -> RQA2025Exception:
        """将原始异常映射为系统异常"""
        for exc_types, target_exception in self.exception_mappings.items():
            if isinstance(exception, exc_types):
                if callable(target_exception):
                    return target_exception(exception)
                else:
                    return target_exception(f"{self.service_name}: {str(exception)}")

        # 默认映射为系统异常
        return SystemException(f"{self.service_name}: {str(exception)}", error_code=5000, error_type="SYSTEM_ERROR")

    def should_alert(self, exception: RQA2025Exception) -> bool:
        """判断是否需要告警"""
        if not self.enable_alerts:
            return False

        # 严重异常需要告警
        return exception.severity in ['CRITICAL', 'ERROR']

    def get_retry_policy(self, exception: RQA2025Exception) -> Optional[Dict[str, Any]]:
        """获取重试策略"""
        # 网络异常可以重试
        if isinstance(exception, NetworkError):
            return {
                'max_attempts': 3,
                'delay_seconds': 1,
                'backoff_factor': 2
            }

        # 数据库连接异常可以重试
        if isinstance(exception, DatabaseError) and 'connection' in str(exception).lower():
            return {
                'max_attempts': 5,
                'delay_seconds': 0.5,
                'backoff_factor': 1.5
            }

        return None


class RetryMechanism:
    """重试机制"""

    def __init__(self, max_attempts: int = 3, delay_seconds: float = 1.0, backoff_factor: float = 2.0):
        self.max_attempts = max_attempts
        self.delay_seconds = delay_seconds
        self.backoff_factor = backoff_factor

    def execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """带重试的执行"""
        import time

        last_exception = None
        current_delay = self.delay_seconds

        for attempt in range(self.max_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e

                if attempt < self.max_attempts - 1:  # 不是最后一次尝试
                    logger.warning(
                        f"执行失败 (尝试 {attempt + 1}/{self.max_attempts}): {e}，{current_delay}秒后重试")
                    time.sleep(current_delay)
                    current_delay *= self.backoff_factor
                else:
                    logger.error(f"执行失败，已达到最大重试次数 ({self.max_attempts}): {e}")
                    raise e

        raise last_exception


# ==================== 异常监控和告警 ====================

class ExceptionMonitor:
    """异常监控器"""

    def __init__(self, service_name: str = "rqa2025"):
        self.service_name = service_name
        self.alert_thresholds = {
            'error_rate_per_minute': 5,  # 每分钟错误率阈值
            'critical_errors_per_hour': 10,  # 每小时严重错误阈值
            'repeated_errors_threshold': 3  # 重复错误告警阈值
        }
        self.alert_callbacks: List[Callable] = []
        self.monitoring_active = False

    def add_alert_callback(self, callback: Callable):
        """添加告警回调"""
        self.alert_callbacks.append(callback)

    def start_monitoring(self):
        """启动监控"""
        self.monitoring_active = True
        logger.info(f"异常监控已启动: {self.service_name}")

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring_active = False
        logger.info(f"异常监控已停止: {self.service_name}")

    def check_alerts(self, exception: RQA2025Exception):
        """检查是否需要告警"""
        if not self.monitoring_active:
            return

        alerts = []

        # 检查错误率
        error_rate = global_exception_stats.get_error_rate()
        if error_rate > self.alert_thresholds['error_rate_per_minute']:
            alerts.append({
                'type': 'high_error_rate',
                'message': f'错误率过高: {error_rate:.2f} 次/分钟',
                'severity': 'warning'
            })

        # 检查严重错误
        recent_critical = sum(1 for exc in global_exception_stats.recent_exceptions[-60:]  # 最近1小时
                              if exc['severity'] == 'CRITICAL')
        if recent_critical > self.alert_thresholds['critical_errors_per_hour']:
            alerts.append({
                'type': 'critical_errors',
                'message': f'严重错误过多: {recent_critical} 个/小时',
                'severity': 'critical'
            })

        # 检查重复错误
        error_type = exception.error_type
        recent_same_errors = sum(1 for exc in global_exception_stats.recent_exceptions[-10:]  # 最近10个
                                 if exc['error_type'] == error_type)
        if recent_same_errors >= self.alert_thresholds['repeated_errors_threshold']:
            alerts.append({
                'type': 'repeated_errors',
                'message': f'重复错误过多: {error_type} ({recent_same_errors}次)',
                'severity': 'warning'
            })

        # 触发告警
        for alert in alerts:
            self._trigger_alert(alert, exception)

    def _trigger_alert(self, alert: Dict[str, Any], exception: RQA2025Exception):
        """触发告警"""
        alert_data = {
            'service': self.service_name,
            'alert_type': alert['type'],
            'severity': alert['severity'],
            'message': alert['message'],
            'exception': exception.to_dict(),
            'timestamp': datetime.now().isoformat()
        }

        # 记录告警日志
        if alert['severity'] == 'critical':
            logger.critical(f"CRITICAL ALERT: {json.dumps(alert_data, ensure_ascii=False)}")
        else:
            logger.warning(f"ALERT: {json.dumps(alert_data, ensure_ascii=False)}")

        # 调用告警回调
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                logger.error(f"告警回调执行失败: {e}")


class ExceptionLogger:
    """异常日志增强器"""

    def __init__(self, service_name: str = "rqa2025"):
        self.service_name = service_name
        self.log_formats = {
            'structured': self._format_structured,
            'simple': self._format_simple,
            'detailed': self._format_detailed
        }
        self.current_format = 'structured'

    def set_log_format(self, format_name: str):
        """设置日志格式"""
        if format_name in self.log_formats:
            self.current_format = format_name
        else:
            raise ValueError(f"不支持的日志格式: {format_name}")

    def log_exception(self, exception: RQA2025Exception, additional_context: Optional[Dict[str, Any]] = None):
        """记录异常"""
        formatter = self.log_formats[self.current_format]
        log_message = formatter(exception, additional_context)

        # 根据严重程度选择日志级别
        if exception.severity == 'CRITICAL':
            logger.critical(log_message)
        elif exception.severity == 'ERROR':
            logger.error(log_message)
        elif exception.severity == 'WARNING':
            logger.warning(log_message)
        else:
            logger.info(log_message)

    def _format_structured(self, exception: RQA2025Exception, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """结构化日志格式"""
        log_data = exception.to_dict()
        if additional_context:
            log_data['additional_context'] = additional_context

        try:
            return json.dumps(log_data, ensure_ascii=False, indent=None)
        except (TypeError, ValueError):
            return f"EXCEPTION: {exception.error_type}:{exception.error_code} {exception.message}"

    def _format_simple(self, exception: RQA2025Exception, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """简单日志格式"""
        context_str = ""
        if additional_context:
            context_str = f" [{additional_context}]"

        return f"{exception.error_type}:{exception.error_code} {exception.message}{context_str}"

    def _format_detailed(self, exception: RQA2025Exception, additional_context: Optional[Dict[str, Any]] = None) -> str:
        """详细日志格式"""
        lines = [
            f"异常详情: {exception.error_type}:{exception.error_code}",
            f"消息: {exception.message}",
            f"严重程度: {exception.severity}",
            f"时间戳: {exception.timestamp}",
            f"服务: {self.service_name}"
        ]

        if exception.context:
            lines.append(f"上下文: {exception.context}")

        if additional_context:
            lines.append(f"额外上下文: {additional_context}")

        if exception.stack_trace:
            lines.append(f"堆栈跟踪: {exception.stack_trace[:500]}...")  # 限制长度

        return " | ".join(lines)


class ExceptionConfiguration:
    """异常处理配置管理"""

    def __init__(self):
        self.config = {
            'monitoring_enabled': True,
            'alerts_enabled': True,
            'log_format': 'structured',
            'retry_enabled': True,
            'statistics_enabled': True,
            'alert_thresholds': {
                'error_rate_per_minute': 5,
                'critical_errors_per_hour': 10,
                'repeated_errors_threshold': 3
            }
        }

    def update_config(self, new_config: Dict[str, Any]):
        """更新配置"""
        def update_nested_dict(target: Dict[str, Any], source: Dict[str, Any]):
            for key, value in source.items():
                if isinstance(value, dict) and key in target and isinstance(target[key], dict):
                    update_nested_dict(target[key], value)
                else:
                    target[key] = value

        update_nested_dict(self.config, new_config)
        logger.info(f"异常处理配置已更新: {new_config}")

    def get_config(self, key: Optional[str] = None) -> Any:
        """获取配置"""
        if key is None:
            return self.config.copy()

        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return None
        return value

    def reset_to_defaults(self):
        """重置为默认配置"""
        self.__init__()
        logger.info("异常处理配置已重置为默认值")


# 全局实例
global_exception_monitor = ExceptionMonitor("rqa2025_system")
global_exception_logger = ExceptionLogger("rqa2025_system")
global_exception_config = ExceptionConfiguration()

# 全局异常处理器实例
global_exception_handler = ExceptionHandler("rqa2025_system")

# 默认异常处理策略
default_business_strategy = ExceptionHandlingStrategy(
    "business", log_level="warning", re_raise=True)
default_infrastructure_strategy = ExceptionHandlingStrategy(
    "infrastructure", log_level="error", re_raise=True)
default_system_strategy = ExceptionHandlingStrategy("system", log_level="critical", re_raise=True)


def _resolve_config_settings(log_level: str, enable_retry: bool,
                           enable_alerts: bool, use_config: bool) -> tuple:
    """解析配置设置"""
    if use_config:
        actual_log_level = global_exception_config.get_config('log_format') or log_level
        actual_retry = global_exception_config.get_config(
            'retry_enabled') if enable_retry else False
        actual_alerts = global_exception_config.get_config(
            'alerts_enabled') if enable_alerts else False
    else:
        actual_log_level = log_level
        actual_retry = enable_retry
        actual_alerts = enable_alerts

    return actual_log_level, actual_retry, actual_alerts


def _create_exception_strategy(service_name: str, actual_log_level: str,
                             re_raise: bool, actual_alerts: bool):
    """创建异常处理策略"""
    return ExceptionHandlingStrategy(
        service_name=service_name,
        log_level=actual_log_level,
        re_raise=re_raise,
        enable_alerts=actual_alerts
    )


def _execute_with_retry_handling(func: Callable, args: tuple, kwargs: dict,
                               strategy: ExceptionHandlingStrategy, re_raise: bool) -> Any:
    """执行带重试处理的函数"""
    retry_mechanism = RetryMechanism()

    def execute_func():
        return func(*args, **kwargs)

    try:
        return retry_mechanism.execute_with_retry(execute_func)
    except Exception as e:
        # 重试失败，使用策略处理异常
        system_exception = strategy.map_exception(e)
        _log_exception_details(system_exception, func, args, kwargs)

        if re_raise:
            raise system_exception from e
        else:
            logger.error(f"Exception handled (not re-raised): {system_exception}")
            return None


def _execute_with_normal_handling(func: Callable, args: tuple, kwargs: dict,
                                strategy: ExceptionHandlingStrategy, re_raise: bool) -> Any:
    """执行带正常异常处理的函数"""
    try:
        return func(*args, **kwargs)
    except RQA2025Exception:
        # 重新抛出系统定义的异常
        raise
    except Exception as e:
        # 使用策略映射异常
        system_exception = strategy.map_exception(e)
        _log_exception_details(system_exception, func, args, kwargs)

        if re_raise:
            raise system_exception from e
        else:
            logger.error(f"Exception handled (not re-raised): {system_exception}")
            return None


def _log_exception_details(system_exception: RQA2025Exception, func: Callable,
                         args: tuple, kwargs: dict) -> None:
    """记录异常详细信息"""
    global_exception_logger.log_exception(system_exception, {
        'function': f"{func.__module__}.{func.__name__}",
        'args_count': len(args),
        'kwargs_keys': list(kwargs.keys()) if kwargs else []
    })


def handle_exceptions(service_name: str = "unknown",
                      log_level: str = "error",
                      re_raise: bool = True,
                      enable_retry: bool = False,
                      enable_alerts: bool = False,
                      use_config: bool = True) -> Callable:
    """
    统一异常处理装饰器 (重构版)

    将长函数拆分为多个职责单一的小函数，提高可维护性。

    Args:
        service_name: 服务名称，用于日志记录
        log_level: 日志级别
        re_raise: 是否重新抛出异常
        enable_retry: 是否启用重试机制
        enable_alerts: 是否启用告警
        use_config: 是否使用全局配置

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # 解析配置设置
            actual_log_level, actual_retry, actual_alerts = _resolve_config_settings(
                log_level, enable_retry, enable_alerts, use_config)

            # 创建异常处理策略
            strategy = _create_exception_strategy(
                service_name, actual_log_level, re_raise, actual_alerts)

            # 根据是否启用重试选择处理方式
            if actual_retry:
                return _execute_with_retry_handling(
                    func, args, kwargs, strategy, re_raise)
            else:
                return _execute_with_normal_handling(
                    func, args, kwargs, strategy, re_raise)

        return wrapper
    return decorator


def handle_business_exceptions(func: Callable) -> Callable:
    """业务逻辑异常处理装饰器"""
    return handle_exceptions("business_logic", "warning", enable_retry=True)(func)


def handle_infrastructure_exceptions(func: Callable) -> Callable:
    """基础设施异常处理装饰器"""
    return handle_exceptions("infrastructure", "error", enable_retry=True, enable_alerts=True)(func)


def handle_system_exceptions(func: Callable) -> Callable:
    """系统异常处理装饰器"""
    return handle_exceptions("system", "critical", enable_alerts=True)(func)


def handle_external_service_exceptions(func: Callable) -> Callable:
    """外部服务异常处理装饰器"""
    return handle_exceptions("external_service", "warning", enable_retry=True)(func)


def handle_database_exceptions(func: Callable) -> Callable:
    """数据库异常处理装饰器"""
    return handle_exceptions("database", "error", enable_retry=True)(func)


def handle_network_exceptions(func: Callable) -> Callable:
    """网络异常处理装饰器"""
    return handle_exceptions("network", "warning", enable_retry=True)(func)


def validate_not_none(value: Any, param_name: str, context: Optional[Dict[str, Any]] = None):
    """
    验证参数不为None

    Args:
        value: 参数值
        param_name: 参数名称
        context: 验证上下文

    Raises:
        ValidationError: 参数为空
    """
    if value is None:
        raise ValidationError(
            f"参数 '{param_name}' 不能为空",
            field=param_name
        )


def validate_range(value: float, min_val: float, max_val: float, param_name: str):
    """
    验证数值范围

    Args:
        value: 数值
        min_val: 最小值
        max_val: 最大值
        param_name: 参数名称

    Raises:
        ValidationError: 数值超出范围
    """
    if not (min_val <= value <= max_val):
        raise ValidationError(
            f"参数 '{param_name}' 必须在 {min_val} 到 {max_val} 之间，实际值: {value}",
            field=param_name
        )


def validate_string_length(value: str, min_len: int = 0, max_len: int = None, param_name: str = None):
    """
    验证字符串长度

    Args:
        value: 字符串值
        min_len: 最小长度
        max_len: 最大长度
        param_name: 参数名称

    Raises:
        ValidationError: 字符串长度不符合要求
    """
    if not isinstance(value, str):
        raise ValidationError(f"参数 '{param_name}' 必须是字符串类型", field=param_name)

    if len(value) < min_len:
        raise ValidationError(
            f"参数 '{param_name}' 长度不能小于 {min_len}，实际长度: {len(value)}",
            field=param_name
        )

    if max_len is not None and len(value) > max_len:
        raise ValidationError(
            f"参数 '{param_name}' 长度不能大于 {max_len}，实际长度: {len(value)}",
            field=param_name
        )


# ==================== 异常统计和监控 ====================

class ExceptionStatistics:
    """异常统计"""

    def __init__(self):
        self.stats: Dict[str, int] = {}
        self.recent_exceptions: List[Dict[str, Any]] = []
        self.max_recent_count = 100

    def record_exception(self, exception: RQA2025Exception):
        """记录异常"""
        error_type = exception.error_type
        self.stats[error_type] = self.stats.get(error_type, 0) + 1

        # 记录最近异常
        self.recent_exceptions.append({
            'timestamp': exception.timestamp,
            'error_type': error_type,
            'message': exception.message,
            'severity': exception.severity
        })

        # 保持最近异常数量限制
        if len(self.recent_exceptions) > self.max_recent_count:
            self.recent_exceptions.pop(0)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            'total_exceptions': sum(self.stats.values()),
            'exception_types': self.stats.copy(),
            'recent_exceptions': self.recent_exceptions[-10:]  # 返回最近10个
        }

    def get_error_rate(self, time_window_minutes: int = 60) -> float:
        """获取错误率（每分钟错误数）"""
        # 计算指定时间窗口内的错误率
        cutoff_time = datetime.now().timestamp() - (time_window_minutes * 60)
        recent_count = sum(1 for exc in self.recent_exceptions
                           if datetime.fromisoformat(exc['timestamp']).timestamp() > cutoff_time)
        return recent_count / time_window_minutes


# 全局异常统计实例
global_exception_stats = ExceptionStatistics()


def record_exception(exception: RQA2025Exception):
    """记录异常到全局统计"""
    global_exception_stats.record_exception(exception)


# 修改RQA2025Exception的初始化，使其自动记录到统计
_original_init = RQA2025Exception.__init__


def _enhanced_init(self, *args, **kwargs):
    _original_init(self, *args, **kwargs)
    record_exception(self)


RQA2025Exception.__init__ = _enhanced_init


def get_exception_stats() -> Dict[str, Any]:
    """获取全局异常统计"""
    return global_exception_stats.get_stats()


# ==================== 便捷管理函数 ====================

def init_exception_monitoring(service_name: str = "rqa2025_system"):
    """
    初始化异常监控系统

    Args:
        service_name: 服务名称
    """
    global global_exception_monitor, global_exception_logger

    global_exception_monitor = ExceptionMonitor(service_name)
    global_exception_logger = ExceptionLogger(service_name)

    # 启动监控
    global_exception_monitor.start_monitoring()
    logger.info(f"异常监控系统已初始化: {service_name}")


def shutdown_exception_monitoring():
    """关闭异常监控系统"""
    if global_exception_monitor.monitoring_active:
        global_exception_monitor.stop_monitoring()
        logger.info("异常监控系统已关闭")


def configure_exceptions(config: Dict[str, Any]):
    """
    配置异常处理系统

    Args:
        config: 配置字典
    """
    global_exception_config.update_config(config)
    logger.info("异常处理系统配置已更新")


def get_exception_config(key: Optional[str] = None) -> Any:
    """
    获取异常处理配置

    Args:
        key: 配置键，支持点分隔的嵌套键

    Returns:
        配置值
    """
    return global_exception_config.get_config(key)


def add_exception_alert_callback(callback: Callable):
    """
    添加异常告警回调

    Args:
        callback: 回调函数，接收告警数据字典
    """
    global_exception_monitor.add_alert_callback(callback)
    logger.info("异常告警回调已添加")


def set_exception_log_format(format_name: str):
    """
    设置异常日志格式

    Args:
        format_name: 格式名称 ('structured', 'simple', 'detailed')
    """
    global_exception_logger.set_log_format(format_name)
    logger.info(f"异常日志格式已设置为: {format_name}")


def get_exception_health_report() -> Dict[str, Any]:
    """
    获取异常处理系统健康报告

    Returns:
        健康报告字典
    """
    stats = global_exception_stats.get_stats()

    return {
        'monitoring_active': global_exception_monitor.monitoring_active,
        'config': global_exception_config.get_config(),
        'statistics': stats,
        'alert_callbacks_count': len(global_exception_monitor.alert_callbacks),
        'timestamp': datetime.now().isoformat()
    }


# ==================== 向后兼容性 ====================

# 保持向后兼容性，导入原有的异常类


# 创建兼容性映射
BusinessLogicError.__bases__ = (BusinessException,)
TradingError.__bases__ = (BusinessException,)
RiskError.__bases__ = (BusinessException,)
StrategyError.__bases__ = (BusinessException,)

# 导出所有异常类
__all__ = [
    # 基础异常
    'RQA2025Exception',

    # 业务层异常
    'BusinessException', 'ValidationError', 'BusinessLogicError', 'WorkflowError',
    'TradingError', 'RiskError', 'StrategyError',

    # 基础设施层异常
    'InfrastructureException', 'ConfigurationError', 'CacheError', 'LoggingError',
    'MonitoringError', 'DatabaseError', 'QueryError', 'ConnectionError', 'NetworkError', 'ResourceError',
    'FileSystemError', 'HealthCheckError',

    # 系统层异常
    'SystemException', 'SecurityError', 'PerformanceError', 'ConcurrencyError', 'AsyncError',

    # 外部服务异常
    'ExternalServiceException', 'ThirdPartyAPIError', 'DataSourceError',

    # 处理工具
    'ExceptionHandler', 'ExceptionStatistics', 'ExceptionHandlingStrategy', 'RetryMechanism',
    'ExceptionMonitor', 'ExceptionLogger', 'ExceptionConfiguration',
    'handle_exceptions', 'handle_business_exceptions', 'handle_infrastructure_exceptions',
    'handle_system_exceptions', 'handle_external_service_exceptions', 'handle_database_exceptions',
    'handle_network_exceptions', 'validate_not_none', 'validate_range', 'validate_string_length',
    'get_exception_stats', 'init_exception_monitoring', 'shutdown_exception_monitoring',
    'configure_exceptions', 'get_exception_config', 'add_exception_alert_callback',
    'set_exception_log_format', 'get_exception_health_report'
]
