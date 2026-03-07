"""
适配器层异常处理
Adapters Layer Exception Handling

定义数据适配器相关的异常类和错误处理机制
"""

from typing import Any


class AdapterException(Exception):
    """适配器基础异常类"""

    def __init__(self, message: str, error_code: int = -1):
        """__init__ 函数的文档字符串"""

        super().__init__(message)
        self.error_code = error_code
        self.message = message


class ConnectionError(AdapterException):
    """连接错误"""

    def __init__(self, message: str, adapter_name: str = None):
        """__init__ 函数的文档字符串"""

        super().__init__(f"连接失败 - {adapter_name}: {message}")
        self.adapter_name = adapter_name


class DataSourceError(AdapterException):
    """数据源错误"""

    def __init__(self, message: str, source_name: str = None):
        """__init__ 函数的文档字符串"""

        super().__init__(f"数据源错误 - {source_name}: {message}")
        self.source_name = source_name


class DataTransformationError(AdapterException):
    """数据转换错误"""

    def __init__(self, message: str, field_name: str = None):
        """__init__ 函数的文档字符串"""

        super().__init__(f"数据转换失败 - {field_name}: {message}")
        self.field_name = field_name


class ValidationError(AdapterException):
    """验证错误"""

    def __init__(self, message: str, validation_rule: str = None):
        """__init__ 函数的文档字符串"""

        super().__init__(f"数据验证失败 - {validation_rule}: {message}")
        self.validation_rule = validation_rule


class ConfigurationError(AdapterException):
    """配置错误"""

    def __init__(self, message: str, config_key: str = None):
        """__init__ 函数的文档字符串"""

        super().__init__(f"配置错误 - {config_key}: {message}")
        self.config_key = config_key


class ResourceExhaustionError(AdapterException):
    """资源耗尽错误"""

    def __init__(self, message: str, resource_type: str = None):
        """__init__ 函数的文档字符串"""

        super().__init__(f"资源耗尽 - {resource_type}: {message}")
        self.resource_type = resource_type


class TimeoutError(AdapterException):
    """超时错误"""

    def __init__(self, message: str, timeout_seconds: int = None):
        """__init__ 函数的文档字符串"""

        super().__init__(f"操作超时 - {timeout_seconds}秒: {message}")
        self.timeout_seconds = timeout_seconds


class AuthenticationError(AdapterException):
    """认证错误"""

    def __init__(self, message: str, auth_method: str = None):
        """__init__ 函数的文档字符串"""

        super().__init__(f"认证失败 - {auth_method}: {message}")
        self.auth_method = auth_method


class RateLimitError(AdapterException):
    """速率限制错误"""

    def __init__(self, message: str, limit_type: str = None):
        """__init__ 函数的文档字符串"""

        super().__init__(f"速率限制 - {limit_type}: {message}")
        self.limit_type = limit_type


def handle_adapter_exception(func) -> Any:
    """
    装饰器：统一处理适配器异常

    Args:
        func: 被装饰的函数

    Returns:
        包装后的函数
    """
    def wrapper(*args, **kwargs) -> Any:
        """wrapper 函数的文档字符串"""

        try:
            return func(*args, **kwargs)
        except AdapterException:
            # 重新抛出适配器异常
            raise
        except Exception as e:
            # 将其他异常包装为适配器异常
            raise AdapterException(f"意外适配器错误: {str(e)}") from e

    return wrapper


def validate_adapter_config(config: dict, required_keys: list):
    """
    验证适配器配置

    Args:
        config: 配置字典
        required_keys: 必需的键列表

    Raises:
        ConfigurationError: 配置验证失败
    """
    if not config:
        raise ConfigurationError("适配器配置不能为空")

    missing_keys = [key for key in required_keys if key not in config]
    if missing_keys:
        raise ConfigurationError(f"缺少必需配置项: {missing_keys}")


def validate_data_source(source_config: dict):
    """
    验证数据源配置

    Args:
        source_config: 数据源配置字典

    Raises:
        DataSourceError: 数据源验证失败
    """
    required_fields = ['type', 'connection_string']
    missing_fields = [field for field in required_fields if field not in source_config]

    if missing_fields:
        raise DataSourceError(f"数据源配置缺少必需字段: {missing_fields}")

    # 验证连接字符串格式
    connection_string = source_config.get('connection_string', '')
    if not connection_string:
        raise DataSourceError("连接字符串不能为空")


def validate_data_quality(data: dict, quality_checks: dict = None):
    """
    验证数据质量

    Args:
        data: 数据字典
        quality_checks: 质量检查配置

    Raises:
        ValidationError: 数据质量验证失败
    """
    if not data:
        raise ValidationError("数据不能为空")

    if quality_checks:
        # 检查必需字段
        required_fields = quality_checks.get('required_fields', [])
        for field in required_fields:
            if field not in data:
                raise ValidationError(f"缺少必需字段: {field}", "required_field")

        # 检查数据类型
        type_rules = quality_checks.get('field_types', {})
        for field, expected_type in type_rules.items():
            if field in data:
                actual_value = data[field]
                if not isinstance(actual_value, expected_type):
                    raise ValidationError(
                        f"字段类型不匹配: {field} 期望{expected_type.__name__}，实际{type(actual_value).__name__}",
                        "field_type"
                    )


def check_adapter_health(adapter_metrics: dict) -> dict:
    """
    检查适配器健康状态

    Args:
        adapter_metrics: 适配器指标字典

    Returns:
        健康检查结果字典
    """
    health_status = {
        'overall_health': 'healthy',
        'warnings': [],
        'critical_issues': []
    }

    # 检查连接状态
    connection_status = adapter_metrics.get('connection_status', 'unknown')
    if connection_status != 'connected':
        health_status['critical_issues'].append(f"连接状态异常: {connection_status}")

    # 检查错误率
    error_rate = adapter_metrics.get('error_rate', 0)
    if error_rate > 0.1:  # 10%错误率
        health_status['critical_issues'].append(f"错误率过高: {error_rate:.1%}")
    elif error_rate > 0.05:  # 5%错误率
        health_status['warnings'].append(f"错误率偏高: {error_rate:.1%}")

    # 检查延迟
    avg_latency = adapter_metrics.get('avg_latency_ms', 0)
    if avg_latency > 5000:  # 5秒
        health_status['critical_issues'].append(f"平均延迟过高: {avg_latency}ms")
    elif avg_latency > 1000:  # 1秒
        health_status['warnings'].append(f"平均延迟偏高: {avg_latency}ms")

    # 确定整体健康状态
    if health_status['critical_issues']:
        health_status['overall_health'] = 'critical'
    elif health_status['warnings']:
        health_status['overall_health'] = 'warning'

    return health_status
