
import copy

from enum import Enum
from typing import Dict, Any, Optional
"""
RQA2025 基础设施层 - 统一异常定义

集中管理所有基础设施层的异常定义，避免重复和不一致。
"""


class ErrorCode(Enum):
    """统一错误代码枚举"""
    # 数据相关错误 (1000-1999)
    DATA_NOT_FOUND = 1001
    DATA_INVALID = 1002
    DATA_PROCESSING_ERROR = 1003
    DATA_FETCH_ERROR = 1004
    DATA_VALIDATION_ERROR = 1005
    DATALOADER_FILE_NOT_FOUND = 1006
    DATALOADER_INVALID_FORMAT = 1007

    # 配置相关错误 (2000-2999)
    CONFIG_NOT_FOUND = 2001
    CONFIG_INVALID = 2002
    CONFIG_LOAD_ERROR = 2003
    CONFIG_SAVE_ERROR = 2004
    CONFIG_FILE_NOT_FOUND = 2005
    CONFIG_INVALID_FORMAT = 2006

    # 网络相关错误 (3000-3999)
    NETWORK_ERROR = 3001
    CONNECTION_TIMEOUT = 3002
    API_ERROR = 3003
    NETWORK_UNAVAILABLE = 3004
    NETWORK_CONNECTION_FAILED = 3005
    NETWORK_TIMEOUT = 3006

    # 数据库相关错误 (4000-4999)
    DATABASE_ERROR = 4001
    DATABASE_CONNECTION_ERROR = 4002
    DATABASE_QUERY_ERROR = 4003
    DATABASE_TRANSACTION_ERROR = 4004
    DATABASE_CONNECTION_FAILED = 4005
    DATABASE_QUERY_FAILED = 4006

    # 缓存相关错误 (5000-5999)
    CACHE_ERROR = 5001
    CACHE_MISS = 5002
    CACHE_INVALID = 5003
    CACHE_CONNECTION_ERROR = 5004
    CACHE_CONNECTION_FAILED = 5005
    CACHE_OPERATION_FAILED = 5006

    # 安全相关错误 (6000-6999)
    SECURITY_ERROR = 6001
    AUTHENTICATION_ERROR = 6002
    AUTHORIZATION_ERROR = 6003
    PERMISSION_DENIED = 6004
    SECURITY_ACCESS_DENIED = 6005
    SECURITY_INVALID_CREDENTIALS = 6006

    # 系统相关错误 (9000-9999)
    SYSTEM_ERROR = 9000
    RESOURCE_UNAVAILABLE = 9001
    PERFORMANCE_THRESHOLD_EXCEEDED = 9002
    SYSTEM_RESOURCE_EXHAUSTED = 9003
    SYSTEM_OPERATION_FAILED = 9004
    UNKNOWN_ERROR = 9999


class InfrastructureError(Exception):
    """基础设施层异常基类"""

    def __init__(self, message: str, error_code: Optional[ErrorCode] = None,
                 details: Optional[Dict[str, Any]] = None, context: Optional[Dict[str, Any]] = None,
                 retryable: bool = False):
        self.message = message
        self.error_code = error_code
        # 深拷贝details以确保不可变性
        self.details = copy.deepcopy(details) if details is not None else None
        self.context = copy.deepcopy(context) if context is not None else None
        self.retryable = retryable
        super().__init__(self.message)

# 数据相关异常


class DataLoaderError(InfrastructureError):
    """数据加载器异常"""

    @classmethod
    def file_not_found(cls, message: str) -> 'DataLoaderError':
        """创建文件未找到错误实例"""
        return cls(message, ErrorCode.DATALOADER_FILE_NOT_FOUND)

    @classmethod
    def invalid_format(cls, message: str) -> 'DataLoaderError':
        """创建格式错误实例"""
        return cls(message, ErrorCode.DATALOADER_INVALID_FORMAT)

    @classmethod
    def encoding_error(cls, message: str) -> 'DataLoaderError':
        """创建编码错误实例"""
        return cls(message, ErrorCode.DATA_INVALID)


class DataProcessingError(InfrastructureError):
    """数据处理异常"""


class DataValidationError(InfrastructureError):
    """数据验证异常"""

# 配置相关异常


class ConfigurationError(InfrastructureError):
    """配置异常"""

    @classmethod
    def file_not_found(cls, message: str) -> 'ConfigurationError':
        """创建文件未找到错误实例"""
        return cls(message, ErrorCode.CONFIG_FILE_NOT_FOUND)

    @classmethod
    def invalid_format(cls, message: str) -> 'ConfigurationError':
        """创建格式错误实例"""
        return cls(message, ErrorCode.CONFIG_INVALID_FORMAT)

    @classmethod
    def missing_config(cls, message: str) -> 'ConfigurationError':
        """创建缺失配置错误实例"""
        return cls(message, ErrorCode.CONFIG_NOT_FOUND)


class ConfigNotFoundError(ConfigurationError):
    """配置未找到异常"""


class ConfigInvalidError(ConfigurationError):
    """配置无效异常"""

# 网络相关异常


class NetworkError(InfrastructureError):
    """网络异常"""

    @classmethod
    def connection_error(cls, message: str) -> 'NetworkError':
        """创建连接错误实例"""
        return cls(message, ErrorCode.NETWORK_CONNECTION_FAILED)

    @classmethod
    def timeout_error(cls, message: str) -> 'NetworkError':
        """创建超时错误实例"""
        return cls(message, ErrorCode.NETWORK_TIMEOUT)

    @classmethod
    def dns_error(cls, message: str) -> 'NetworkError':
        """创建DNS错误实例"""
        return cls(message, ErrorCode.NETWORK_ERROR)


class ConnectionTimeoutError(NetworkError):
    """连接超时异常"""


class NetworkUnavailableError(NetworkError):
    """网络不可用异常"""

# 数据库相关异常


class DatabaseError(InfrastructureError):
    """数据库异常"""

    @classmethod
    def connection_error(cls, message: str) -> 'DatabaseError':
        """创建连接错误实例"""
        return cls(message, ErrorCode.DATABASE_CONNECTION_FAILED)

    @classmethod
    def query_error(cls, message: str) -> 'DatabaseError':
        """创建查询错误实例"""
        return cls(message, ErrorCode.DATABASE_QUERY_FAILED)

    @classmethod
    def transaction_error(cls, message: str) -> 'DatabaseError':
        """创建事务错误实例"""
        return cls(message, ErrorCode.DATABASE_TRANSACTION_ERROR)


class DatabaseConnectionError(DatabaseError):
    """数据库连接异常"""


class DatabaseQueryError(DatabaseError):
    """数据库查询异常"""


class DatabaseTransactionError(DatabaseError):
    """数据库事务异常"""

# 缓存相关异常


class CacheError(InfrastructureError):
    """缓存异常"""

    @classmethod
    def connection_error(cls, message: str) -> 'CacheError':
        """创建连接错误实例"""
        return cls(message, ErrorCode.CACHE_CONNECTION_FAILED)

    @classmethod
    def operation_error(cls, message: str) -> 'CacheError':
        """创建操作错误实例"""
        return cls(message, ErrorCode.CACHE_OPERATION_FAILED)

    @classmethod
    def serialization_error(cls, message: str) -> 'CacheError':
        """创建序列化错误实例"""
        return cls(message, ErrorCode.CACHE_INVALID)


class CacheMemoryError(CacheError):
    """缓存内存异常"""


class CacheConnectionError(CacheError):
    """缓存连接异常"""

# 安全相关异常


class SecurityError(InfrastructureError):
    """安全异常"""

    @classmethod
    def access_denied(cls, message: str) -> 'SecurityError':
        """创建访问拒绝错误实例"""
        return cls(message, ErrorCode.SECURITY_ACCESS_DENIED)

    @classmethod
    def invalid_credentials(cls, message: str) -> 'SecurityError':
        """创建无效凭据错误实例"""
        return cls(message, ErrorCode.SECURITY_INVALID_CREDENTIALS)

    @classmethod
    def insufficient_permissions(cls, message: str) -> 'SecurityError':
        """创建权限不足错误实例"""
        return cls(message, ErrorCode.PERMISSION_DENIED)


class AuthenticationError(SecurityError):
    """认证异常"""


class AuthorizationError(SecurityError):
    """授权异常"""


class PermissionDeniedError(SecurityError):
    """权限拒绝异常"""

# 系统相关异常


class SystemError(InfrastructureError):
    """系统异常"""

    @classmethod
    def resource_exhausted(cls, message: str) -> 'SystemError':
        """创建资源耗尽错误实例"""
        return cls(message, ErrorCode.SYSTEM_RESOURCE_EXHAUSTED)

    @classmethod
    def operation_failed(cls, message: str) -> 'SystemError':
        """创建操作失败错误实例"""
        return cls(message, ErrorCode.SYSTEM_OPERATION_FAILED)

    @classmethod
    def hardware_failure(cls, message: str) -> 'SystemError':
        """创建硬件故障错误实例"""
        return cls(message, ErrorCode.SYSTEM_ERROR)


class ResourceUnavailableError(SystemError):
    """资源不可用异常"""


class PerformanceThresholdExceededError(SystemError):
    """性能阈值超限异常"""

# 严重程度分类的异常


class CriticalError(InfrastructureError):
    """严重错误，需要立即处理"""


class WarningError(InfrastructureError):
    """警告级别错误，可以继续运行但需要注意"""


class InfoLevelError(InfrastructureError):
    """信息级别错误，仅用于记录"""


class RetryableError(InfrastructureError):
    """可重试错误"""

    def __init__(self, message: str, error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
                 details: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code, details, retryable=True)


class RetryError(RetryableError):
    """重试异常"""

# 业务特定异常


class TradingError(InfrastructureError):
    """交易异常"""


class OrderRejectedError(TradingError):
    """订单拒绝异常"""


class InvalidPriceError(TradingError):
    """价格无效异常"""


class TradeError(TradingError):
    """交易执行异常"""


class CircuitBreakerOpenError(RetryableError):
    """熔断器开启异常"""


# 兼容性别名
# 保持向后兼容性
ValidationError = DataValidationError
ConfigError = ConfigInvalidError
ConnectionError = DatabaseConnectionError
TimeoutError = ConnectionTimeoutError

# 便捷函数


def create_error(error_code: ErrorCode, message: str, details: Optional[Dict[str, Any]] = None) -> InfrastructureError:
    """创建指定错误代码的异常"""
    return InfrastructureError(message, error_code, details)


def is_retryable_error(error: Exception) -> bool:
    """检查错误是否可重试"""
    return isinstance(error, RetryableError) or getattr(error, 'retryable', False)


def get_error_code(error: Exception) -> ErrorCode:
    """获取错误的错误代码"""
    if hasattr(error, 'error_code'):
        error_code = getattr(error, 'error_code')
        if isinstance(error_code, ErrorCode):
            return error_code
    return ErrorCode.UNKNOWN_ERROR


def get_error_details(error: Exception) -> Dict[str, Any]:
    """获取错误的详细信息"""
    if hasattr(error, 'details'):
        details = getattr(error, 'details')
        if isinstance(details, dict):
            return details
    return {}


# 异常分类映射
ERROR_SEVERITY_MAP = {
    CriticalError: "CRITICAL",
    WarningError: "WARNING",
    InfoLevelError: "INFO",
    InfrastructureError: "ERROR",
    Exception: "UNKNOWN"
}


def get_error_severity(error: Exception) -> str:
    """获取错误严重程度"""
    for error_type, severity in ERROR_SEVERITY_MAP.items():
        if isinstance(error, error_type):
            return severity
    return "UNKNOWN"
