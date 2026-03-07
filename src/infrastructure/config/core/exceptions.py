
from typing import Any, Dict, Optional
"""
配置管理模块异常处理
Config Management Module Exception Handling

定义配置管理相关的异常类和错误处理机制
"""

import sys
import os
# Add src to path for absolute imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_path = os.path.join(current_dir, '..', '..', '..')
if src_path not in sys.path:
    sys.path.insert(0, src_path)

try:
    from src.infrastructure.core.exceptions import InfrastructureException
except ImportError:
    InfrastructureException = Exception


class ConfigException(InfrastructureException):
    """配置基础异常类"""

    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None, error_type: Optional[str] = None):
        super().__init__(message)
        self.config_key = config_key
        self.details = details or {}
        self.message = message
        self.error_type = error_type or config_key


class ConfigLoadError(ConfigException):
    """配置加载错误"""

    def __init__(self, message: str, source: Optional[str] = None, details: Optional[Dict[str, Any]] = None, **kwargs):
        merged_details = dict(details or {})
        merged_details.setdefault('source', source)
        super().__init__(message, details=merged_details, **kwargs)
        self.source = merged_details.get('source')
        self.context = merged_details


class ConfigValidationError(ConfigException):
    """配置验证错误"""

    def __init__(self, message: str, expected_type: Optional[str] = None,
                 actual_type: Optional[str] = None, value: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.expected_type = expected_type
        self.actual_type = actual_type
        self.value = value


class ConfigTypeError(ConfigValidationError):
    """配置类型错误"""

    def __init__(self, message: str, expected_type: Optional[str] = None,
                 actual_type: Optional[str] = None, value: Any = None, **kwargs):
        super().__init__(message, expected_type=expected_type,
                         actual_type=actual_type, value=value, **kwargs)


class ConfigKeyError(ConfigException):
    """配置键错误"""

    def __init__(self, message: str, key: Optional[str] = None, **kwargs):
        super().__init__(message, config_key=key, **kwargs)
        self.key = key


class ConfigNotFoundError(ConfigKeyError):
    """配置未找到错误"""

    def __init__(self, message: str, key: Optional[str] = None, **kwargs):
        super().__init__(message, key=key, **kwargs)


class ConfigDuplicateError(ConfigException):
    """配置重复错误"""

    def __init__(self, message: str, key: Optional[str] = None, **kwargs):
        super().__init__(message, config_key=key, **kwargs)
        self.key = key


class ConfigSecurityError(ConfigException):
    """配置安全错误"""

    def __init__(self, message: str, security_issue: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.security_issue = security_issue


class ConfigEncryptionError(ConfigSecurityError):
    """配置加密错误"""

    def __init__(self, message: str, algorithm: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.algorithm = algorithm


class ConfigAccessError(ConfigSecurityError):
    """配置访问错误"""

    def __init__(self, message: str, user: Optional[str] = None,
                 permission: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.user = user
        self.permission = permission


class ConfigQuotaError(ConfigException):
    """配置配额错误"""

    def __init__(self, message: str, quota_type: Optional[str] = None,
                 current_usage: Optional[int] = None, limit: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.limit = limit


class ConfigVersionError(ConfigException):
    """配置版本错误"""

    def __init__(self, message: str, version: Optional[str] = None,
                 expected_version: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.version = version
        self.expected_version = expected_version


class ConfigMergeError(ConfigException):
    """配置合并错误"""

    def __init__(self, message: str, conflict_keys: Optional[list] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.conflict_keys = conflict_keys or []


class ConfigTimeoutError(ConfigException):
    """配置超时错误"""

    def __init__(self, message: str, timeout: Optional[float] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.timeout = timeout
        self.operation = operation


class ConfigConnectionError(ConfigException):
    """配置连接错误"""

    def __init__(self, message: str, host: Optional[str] = None,
                 port: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.host = host
        self.port = port


class ConfigStorageError(ConfigException):
    """配置存储错误"""

    def __init__(self, message: str, storage_type: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.storage_type = storage_type
        self.operation = operation


class ConfigCacheError(ConfigException):
    """配置缓存错误"""

    def __init__(self, message: str, cache_key: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, config_key=cache_key, **kwargs)
        self.cache_key = cache_key
        self.operation = operation


class ConfigMonitorError(ConfigException):
    """配置监控错误"""

    def __init__(self, message: str, metric: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.metric = metric


class ConfigPerformanceError(ConfigException):
    """配置性能错误"""

    def __init__(self, message: str, threshold: Optional[float] = None,
                 actual_value: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.threshold = threshold
        self.actual_value = actual_value


# ============================================================================
# 兼容性异常类 - 用于与通用基础设施异常兼容
# ============================================================================

class ConfigurationError(ConfigException):
    """配置相关异常 - 兼容性版本"""

    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(message, config_key)


class ValidationError(ConfigValidationError):
    """验证异常 - 兼容性版本"""

    def __init__(self, message: str, field: Optional[str] = None, value: Optional[Any] = None):
        super().__init__(message, expected_type="any", actual_type=type(value).__name__ if value is not None else "None", value=value)
        self.field = field


class InfrastructureException(ConfigException):
    """基础设施异常 - 兼容性版本"""

    def __init__(self, message: str, component: Optional[str] = None):
        super().__init__(message)
        self.component = component


# ============================================================================
# 异常处理装饰器
# ============================================================================


def handle_config_exception(operation: str = "config: Dict[str, Any]_operation"):
    """
    配置异常处理装饰器

    Args:
        operation: 操作名称，用于日志记录

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ConfigException:
                # 重新抛出配置异常，不做额外处理
                raise
            except Exception as e:
                # 将其他异常包装为配置异常
                error_msg = f"{operation} 失败: {str(e)}"
                raise ConfigException(error_msg, details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_config_load_exception(source: str = "unknown"):
    """
    配置加载异常处理装饰器

    Args:
        source: 配置源，用于错误信息

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ConfigLoadError:
                raise
            except Exception as e:
                error_msg = f"从 {source} 加载配置失败: {str(e)}"
                raise ConfigLoadError(error_msg, source=source,
                                      details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_config_validation_exception(field: str = "unknown"):
    """
    配置验证异常处理装饰器

    Args:
        field: 验证字段名

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except ConfigValidationError:
                raise
            except Exception as e:
                error_msg = f"验证字段 '{field}' 失败: {str(e)}"
                raise ConfigValidationError(error_msg, details={"original_error": str(e)}) from e
        return wrapper
    return decorator




