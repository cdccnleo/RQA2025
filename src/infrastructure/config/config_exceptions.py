
from typing import Dict, Any, Optional

from .core.config_strategy import ConfigLoadError as StrategyConfigLoadError
from .core.exceptions import ConfigException as CoreConfigException

"""
配置系统异常定义
定义配置管理过程中可能出现的各种异常
"""


class ConfigError(CoreConfigException):
    """配置系统基础异常"""

    def __init__(self, message: str, config_key: Optional[str] = None, details: Optional[Dict[str, Any]] = None, error_type: Optional[str] = None):
        super().__init__(
            message,
            config_key=config_key,
            details=details,
            error_type=error_type or "config_error",
        )


class ConfigValidationError(ConfigError):
    """配置验证错误"""

    def __init__(self, message: str, config_key_or_errors=None, validation_errors: Optional[list] = None, **kwargs):
        if isinstance(config_key_or_errors, list):
            config_key = kwargs.get('config_key')
            errors = config_key_or_errors
        elif 'config_key' in kwargs:
            config_key = kwargs['config_key']
            errors = config_key_or_errors if isinstance(config_key_or_errors, list) else validation_errors
        else:
            config_key = config_key_or_errors
            errors = validation_errors

        super().__init__(message, config_key, {'validation_errors': errors or []})
        self.errors = errors or []


class ConfigLoadError(StrategyConfigLoadError, ConfigError):
    """配置加载错误"""

    def __init__(self, message: str, source_or_context=None, details: Optional[Dict[str, Any]] = None, **kwargs):
        context: Dict[str, Any] = {}

        if isinstance(source_or_context, dict):
            context.update(source_or_context)
            source = kwargs.get('source', context.get('source'))
        else:
            source = kwargs.get('source', source_or_context)
            if details:
                context.update(details)
            if 'context' in kwargs and isinstance(kwargs['context'], dict):
                context.update(kwargs['context'])

        context.setdefault('source', source)
        config_key = kwargs.get('config_key')

        StrategyConfigLoadError.__init__(self, message, context=context)
        ConfigError.__init__(
            self,
            message,
            config_key=config_key,
            details=context,
            error_type=kwargs.get('error_type', 'config_load_error'),
        )
        self.context = context
        self.source = context.get('source')
        self.details = context


class ConfigNotFoundError(ConfigError):
    """配置未找到错误"""

    def __init__(self, config_key: str, searched_locations: Optional[list] = None):
        message = f"配置项 '{config_key}' 未找到"
        super().__init__(message, config_key=config_key, details={
            'searched_locations': searched_locations or []})


class ConfigTypeError(ConfigError):
    """配置类型错误"""

    def __init__(self, message: str, expected_type: Optional[str] = None, actual_type: Optional[str] = None, config_key: Optional[str] = None, value: Any = None):
        # 处理不同的调用方式
        if expected_type is not None and actual_type is not None:
            # 标准用法：ConfigTypeError(message, expected_type, actual_type)
            # 或者 ConfigTypeError(message, expected_type, actual_type, config_key, value)
            super().__init__(message, config_key, {
                'expected_type': expected_type, 'actual_type': actual_type, 'value': value})
        else:
            # 兼容用法：只提供message
            super().__init__(message, config_key, {})


class ConfigValueError(ConfigError):
    """配置值错误"""

    def __init__(self, message: str, config_key: Optional[str] = None, expected_type: Optional[str] = None, actual_type: Optional[str] = None, value: Any = None):
        if config_key and expected_type and actual_type:
            # 标准用法：提供所有参数
            full_message = f"配置项 '{config_key}' 类型错误，期望 {expected_type}，实际 {actual_type}"
            if message:
                full_message += f": {message}"
            super().__init__(full_message, config_key, {
                'expected_type': expected_type, 'actual_type': actual_type, 'value': value})
        else:
            # 兼容用法：只提供message
            super().__init__(message, config_key, {
                'expected_type': expected_type, 'actual_type': actual_type, 'value': value})


class ConfigAccessError(ConfigError):
    """配置访问错误"""

    def __init__(self, message: str, config_key: Optional[str] = None, reason: Optional[str] = None):
        super().__init__(message, config_key, {'reason': reason})


class ConfigSecurityError(ConfigError):
    """配置安全错误"""

    def __init__(self, message: str, config_key: Optional[str] = None, security_issue: Optional[str] = None):
        super().__init__(message, config_key, {'security_issue': security_issue})


class ConfigFormatError(ConfigError):
    """配置格式错误"""

    def __init__(self, message: str, format_type: Optional[str] = None, line_number: Optional[int] = None):
        super().__init__(message, details={
            'format_type': format_type,
            'line_number': line_number
        })


class ConfigMergeError(ConfigError):
    """配置合并错误"""

    def __init__(self, message: str, conflicting_keys: Optional[list] = None):
        super().__init__(message, details={'conflicting_keys': conflicting_keys or []})


class ConfigBackupError(ConfigError):
    """配置备份错误"""

    def __init__(self, message: str, backup_path: Optional[str] = None):
        super().__init__(message, details={'backup_path': backup_path})


class ConfigRestoreError(ConfigError):
    """配置恢复错误"""

    def __init__(self, message: str, backup_path: Optional[str] = None, restore_point: Optional[str] = None):
        super().__init__(message, details={
            'backup_path': backup_path,
            'restore_point': restore_point
        })


class ConfigVersionError(ConfigError):
    """配置版本错误"""

    def __init__(self, message: str, version: Optional[str] = None, expected_version: Optional[str] = None):
        super().__init__(message, details={
            'version': version,
            'expected_version': expected_version
        })


class ConfigEncryptionError(ConfigError):
    """配置加密错误"""

    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(message, details={'operation': operation})


class ConfigDecryptionError(ConfigError):
    """配置解密错误"""

    def __init__(self, message: str, operation: Optional[str] = None):
        super().__init__(message, details={'operation': operation})


class ConfigNetworkError(ConfigError):
    """配置网络错误"""

    def __init__(self, message: str, endpoint: Optional[str] = None, timeout: Optional[float] = None):
        super().__init__(message, details={
            'endpoint': endpoint,
            'timeout': timeout
        })


class ConfigTimeoutError(ConfigError):
    """配置超时错误"""

    def __init__(self, message: str, operation: Optional[str] = None, timeout_seconds: Optional[float] = None):
        super().__init__(message, details={
            'operation': operation,
            'timeout_seconds': timeout_seconds
        })


class ConfigQuotaExceededError(ConfigError):
    """配置配额超限错误"""

    def __init__(self, message: str, quota_type: Optional[str] = None, current_usage: Optional[int] = None, limit: Optional[int] = None):
        super().__init__(message, details={
            'quota_type': quota_type,
            'current_usage': current_usage,
            'limit': limit
        })


def raise_config_error(error_type: str, message: str, **kwargs):
    """根据错误类型抛出相应的异常"""
    error_classes = {
        'validation': ConfigValidationError,
        'load': ConfigLoadError,
        'not_found': ConfigNotFoundError,
        'type': ConfigTypeError,
        'access': ConfigAccessError,
        'security': ConfigSecurityError,
        'format': ConfigFormatError,
        'merge': ConfigMergeError,
        'backup': ConfigBackupError,
        'restore': ConfigRestoreError,
        'version': ConfigVersionError,
        'encryption': ConfigEncryptionError,
        'decryption': ConfigDecryptionError,
        'network': ConfigNetworkError,
        'timeout': ConfigTimeoutError,
        'quota': ConfigQuotaExceededError
    }

    error_class = error_classes.get(error_type, ConfigError)
    raise error_class(message, **kwargs)

# ==================== 向后兼容类 ====================


class ConfigTypeErrorOld(ConfigError):
    """配置类型错误（向后兼容简单接口）"""

    def __init__(self, message: str, expected_type: Optional[str] = None, actual_type: Optional[str] = None):
        super().__init__(message, details={
            'expected_type': expected_type,
            'actual_type': actual_type
        })

# 为向后兼容性创建别名
# ConfigTypeError = ConfigTypeErrorOld




