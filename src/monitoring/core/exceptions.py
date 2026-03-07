"""
监控层异常处理
Monitoring Layer Exception Handling

定义监控相关的异常类和错误处理机制
"""


class MonitoringException(Exception):
    """监控基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class MetricsCollectionError(MonitoringException):
    """指标收集错误"""

    def __init__(self, message: str, metric_name: str = None):
        super().__init__(f"指标收集失败 - {metric_name}: {message}")
        self.metric_name = metric_name


class AlertProcessingError(MonitoringException):
    """告警处理错误"""

    def __init__(self, message: str, alert_id: str = None):
        super().__init__(f"告警处理失败 - {alert_id}: {message}")
        self.alert_id = alert_id


class ConfigurationError(MonitoringException):
    """配置错误"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(f"配置错误 - {config_key}: {message}")
        self.config_key = config_key


class HealthCheckError(MonitoringException):
    """健康检查错误"""

    def __init__(self, message: str, component: str = None):
        super().__init__(f"健康检查失败 - {component}: {message}")
        self.component = component


class ResourceExhaustionError(MonitoringException):
    """资源耗尽错误"""

    def __init__(self, message: str, resource_type: str = None):
        super().__init__(f"资源耗尽 - {resource_type}: {message}")
        self.resource_type = resource_type


class DataPersistenceError(MonitoringException):
    """数据持久化错误"""

    def __init__(self, message: str, data_type: str = None):
        super().__init__(f"数据持久化失败 - {data_type}: {message}")
        self.data_type = data_type


def handle_monitoring_exception(func):
    """
    装饰器：统一处理监控异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except MonitoringException:
            # 重新抛出监控异常
            raise
        except Exception as e:
            # 将其他异常包装为监控异常
            raise MonitoringException(f"意外错误: {str(e)}") from e

    return wrapper


def validate_metric_data(metric_name: str, value, expected_type=None):
    """
    验证指标数据

    Args:
        metric_name: 指标名称
        value: 指标值
        expected_type: 期望的数据类型

    Raises:
        MetricsCollectionError: 数据验证失败
    """
    if value is None:
        raise MetricsCollectionError(f"指标值不能为空", metric_name)

    if expected_type and not isinstance(value, expected_type):
        raise MetricsCollectionError(
            f"指标值类型错误，期望{expected_type.__name__}，实际{type(value).__name__}",
            metric_name
        )


def validate_config_key(config: dict, key: str, required: bool = False):
    """
    验证配置键是否存在

    Args:
        config: 配置字典
        key: 配置键
        required: 是否必需

    Raises:
        ConfigurationError: 配置验证失败
    """
    if required and key not in config:
        raise ConfigurationError(f"必需配置项缺失: {key}", key)

    if key in config and config[key] is None:
        raise ConfigurationError(f"配置项值不能为空: {key}", key)
