"""
exceptions 模块

提供 exceptions 相关功能和接口。
"""

import os

"""
基础设施层异常处理
Infrastructure Layer Exception Handling

定义基础设施相关的异常类和错误处理机制
"""


class InfrastructureException(Exception):
    """基础设施基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ConfigurationError(InfrastructureException):
    """配置异常"""

    def __init__(self, message: str, config_key: str = None):
        super().__init__(f"配置错误 - {config_key}: {message}")
        self.config_key = config_key


class CacheError(InfrastructureException):
    """缓存异常"""

    def __init__(self, message: str, cache_key: str = None, error_code: int = 2001):
        super().__init__(f"缓存错误 - {cache_key}: {message}", error_code)
        self.cache_key = cache_key


class LoggingError(InfrastructureException):
    """日志异常"""

    def __init__(self, message: str, log_file: str = None, error_code: int = 3001):
        if log_file:
            message = f"日志错误 - {log_file}: {message}"
        super().__init__(message, error_code)
        self.log_file = log_file


class MonitoringError(InfrastructureException):
    """监控异常"""

    def __init__(self, message: str, metric_name: str = None, error_code: int = 4001):
        if metric_name:
            message = f"监控错误 - {metric_name}: {message}"
        super().__init__(message, error_code)
        self.metric_name = metric_name


class ResourceError(InfrastructureException):
    """资源异常"""

    def __init__(self, message: str, resource_type: str = None, error_code: int = 5001):
        if resource_type:
            message = f"资源错误 - {resource_type}: {message}"
        super().__init__(message, error_code)
        self.resource_type = resource_type


class NetworkError(InfrastructureException):
    """网络异常"""

    def __init__(self, message: str, endpoint: str = None, error_code: int = 6001):
        if endpoint:
            message = f"网络错误 - {endpoint}: {message}"
        super().__init__(message, error_code)
        self.endpoint = endpoint


class DatabaseError(InfrastructureException):
    """数据库异常"""

    def __init__(self, message: str, error_code: int = 7001, table_name: str = None):
        if table_name:
            message = f"数据库错误 - {table_name}: {message}"
        super().__init__(message, error_code)
        self.table_name = table_name


class FileSystemError(InfrastructureException):
    """文件系统异常"""

    def __init__(self, message: str, file_path: str = None):
        super().__init__(f"文件系统错误 - {file_path}: {message}")
        self.file_path = file_path


class SecurityError(InfrastructureException):
    """安全异常"""

    def __init__(self, message: str, security_context: str = None):
        super().__init__(f"安全错误 - {security_context}: {message}")
        self.security_context = security_context


class HealthCheckError(InfrastructureException):
    """健康检查异常"""

    def __init__(self, message: str, check_target: str = None):
        super().__init__(f"健康检查失败 - {check_target}: {message}")
        self.check_target = check_target


class VersionError(InfrastructureException):
    """版本异常"""

    def __init__(self, message: str, version: str = None):
        super().__init__(f"版本错误 - {version}: {message}")
        self.version = version


def handle_infrastructure_exception(func):
    """
    装饰器：统一处理基础设施异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except InfrastructureException:
            # 重新抛出基础设施异常
            raise
        except Exception as e:
            # 将其他异常包装为基础设施异常
            raise InfrastructureException(f"意外基础设施错误: {str(e)}") from e

    return wrapper


def validate_config_value(value, expected_type=None, min_value=None, max_value=None):
    """
    验证配置值

    Args:
        value: 配置值
        expected_type: 期望类型
        min_value: 最小值
        max_value: 最大值

    Raises:
        ConfigurationError: 配置验证失败
    """
    if expected_type and not isinstance(value, expected_type):
        raise ConfigurationError(
            f"配置值类型错误，期望{expected_type.__name__}，实际{type(value).__name__}"
        )

    if isinstance(value, (int, float)):
        if min_value is not None and value < min_value:
            raise ConfigurationError(f"配置值不能小于{min_value}")
        if max_value is not None and value > max_value:
            raise ConfigurationError(f"配置值不能大于{max_value}")


def validate_resource_limits(current_usage: float, limit: float, resource_name: str):
    """
    验证资源限制

    Args:
        current_usage: 当前使用量
        limit: 限制值
        resource_name: 资源名称

    Raises:
        ResourceError: 资源验证失败
    """
    if current_usage > limit:
        raise ResourceError(
            f"{resource_name} 使用量 {current_usage} 超过限制 {limit}",
            resource_name
        )


def validate_file_path(file_path: str, must_exist: bool = False):
    """
    验证文件路径

    Args:
        file_path: 文件路径
        must_exist: 是否必须存在

    Raises:
        FileSystemError: 文件路径验证失败
    """
    if not file_path:
        raise FileSystemError("文件路径不能为空")

    if must_exist and not os.path.exists(file_path):
        raise FileSystemError(f"文件不存在: {file_path}", file_path)

    if must_exist and os.path.isdir(file_path):
        raise FileSystemError(f"路径是目录而非文件: {file_path}", file_path)


def check_health_status(is_healthy: bool, component_name: str):
    """
    检查健康状态

    Args:
        is_healthy: 是否健康
        component_name: 组件名称

    Raises:
        HealthCheckError: 健康检查失败
    """
    if not is_healthy:
        raise HealthCheckError(f"组件 {component_name} 健康检查失败", component_name)
