"""
cache_exceptions 模块

提供 cache_exceptions 相关功能和接口。
"""

import logging

from datetime import datetime
from typing import Dict, Any, Optional, List
"""
缓存系统异常定义

提供完整的异常层次结构，支持：
- 详细的错误上下文信息
- 错误分类和错误码
- 错误恢复建议
- 调试信息记录
"""

logger = logging.getLogger(__name__)


class CacheError(Exception):
    """
    缓存基础异常类

    所有缓存相关异常的基类，提供统一的错误处理接口。
    """

    def __init__(self,
                 message: str,
                 error_code: str = "CACHE_ERROR",
                 details: Optional[Dict[str, Any]] = None,
                 recovery_suggestion: str = "",
                 cause: Optional[Exception] = None,
                 **kwargs):
        """
        初始化异常

        Args:
            message: 错误描述信息
            error_code: 错误码，用于错误分类和处理
            details: 详细错误信息字典
            recovery_suggestion: 恢复建议
            cause: 引起此异常的原因异常
            **kwargs: 其他自定义属性
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.recovery_suggestion = recovery_suggestion
        self.timestamp = datetime.now()
        self.cause = cause

        # 处理额外的关键字参数作为属性和details
        for key, value in kwargs.items():
            setattr(self, key, value)
            self.details[key] = value

        # 记录错误信息
        self._log_error()

    def _log_error(self):
        """记录错误信息到日志"""
        logger.error(f"[{self.error_code}] {self.message}")
        if self.details:
            logger.error(f"错误详情: {self.details}")
        if self.recovery_suggestion:
            logger.info(f"恢复建议: {self.recovery_suggestion}")

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'error_code': self.error_code,
            'message': self.message,
            'details': self.details,
            'recovery_suggestion': self.recovery_suggestion,
            'timestamp': self.timestamp.isoformat()
        }

    @classmethod
    def create_error(cls,
                     message: str,
                     error_code: str,
                     recovery_suggestion: str,
                     details: Optional[Dict[str, Any]] = None,
                     **extra_details) -> 'CacheError':
        """
        创建异常实例的通用方法

        Args:
            message: 错误信息
            error_code: 错误码
            recovery_suggestion: 恢复建议
            details: 基础详情字典
            **extra_details: 额外的详情信息

        Returns:
            CacheError: 创建的异常实例
        """
        if not details:
            details = {}
        details.update(extra_details)

        return cls(message, error_code, details, recovery_suggestion)


class CacheConnectionError(CacheError):
    """
    缓存连接异常

    当无法建立或维护缓存连接时抛出。
    """

    def __init__(self,
                 message: str = "无法连接到缓存服务器",
                 host: str = "",
                 port: int = 0,
                 details: Optional[Dict[str, Any]] = None,
                 retry_count: int = 0,
                 should_retry: bool = True,
                 **kwargs):
        # 将额外参数添加到details中
        if details is None:
            details = {}
        details.update({
            'host': host,
            'port': port,
            'retry_count': retry_count,
            'should_retry': should_retry,
            'connection_type': 'cache_server'
        })

        # 设置额外属性
        self.host = host
        self.port = port
        self.retry_count = retry_count
        self.should_retry = should_retry

        super().__init__(
            message=message,
            error_code="CONNECTION_ERROR",
            recovery_suggestion="请检查网络连接、服务器状态和认证信息",
            details=details,
            **kwargs
        )


class CacheSerializationError(CacheError):
    """
    缓存序列化异常

    当数据序列化或反序列化失败时抛出。
    """

    def __init__(self,
                 message: str = "数据序列化失败",
                 operation: str = "",
                 data_type: str = "",
                 data: Any = None,
                 format: str = "",
                 details: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # 将额外参数添加到details中
        if details is None:
            details = {}
        details.update({
            'operation': operation,
            'data_type': data_type,
            'data': str(data) if data is not None else None,
            'format': format
        })

        # 设置额外属性
        self.operation = operation
        self.data_type = data_type
        self.data = data
        self.format = format

        super().__init__(
            message=message,
            error_code="SERIALIZATION_ERROR",
            recovery_suggestion="请检查数据格式和序列化配置",
            details=details,
            **kwargs
        )


class CacheKeyError(CacheError):
    """
    缓存键异常

    当缓存键无效或操作失败时抛出。
    """

    def __init__(self,
                 message: str = "缓存键操作失败",
                 key: str = "",
                 operation: str = "",
                 key_type: str = "",
                 details: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # 将额外参数添加到details中
        if details is None:
            details = {}
        details.update({
            'key': key,
            'operation': operation,
            'key_type': key_type
        })

        # 设置额外属性
        self.key = key
        self.operation = operation
        self.key_type = key_type

        super().__init__(
            message=message,
            error_code="KEY_ERROR",
            recovery_suggestion="请检查键名格式和权限设置",
            details=details,
            **kwargs
        )


class CacheTimeoutError(CacheError):
    """
    缓存超时异常

    当缓存操作超时或响应时间过长时抛出。
    """

    def __init__(self,
                 message: str = "缓存操作超时",
                 operation: str = "",
                 timeout_ms: int = 0,
                 timeout_seconds: float = 0.0,
                 retryable: bool = False,
                 details: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # 将额外参数添加到details中
        if details is None:
            details = {}
        details.update({
            'operation': operation,
            'timeout_ms': timeout_ms,
            'timeout_seconds': timeout_seconds,
            'retryable': retryable
        })

        # 设置额外属性
        self.operation = operation
        self.timeout_ms = timeout_ms
        self.timeout_seconds = timeout_seconds
        self.retryable = retryable

        super().__init__(
            message=message,
            error_code="TIMEOUT_ERROR",
            recovery_suggestion="请检查网络延迟、服务器负载或增加超时时间",
            details=details,
            **kwargs
        )


class CacheConsistencyError(CacheError):
    """
    缓存一致性异常

    当缓存数据一致性出现问题时抛出。
    """

    def __init__(self,
                 message: str = "缓存数据一致性错误",
                 key: str = "",
                 expected_version: str = "",
                 actual_version: str = "",
                 keys: Optional[List[str]] = None,
                 details: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # 将额外参数添加到details中
        if details is None:
            details = {}
        details.update({
            'key': key,
            'expected_version': expected_version,
            'actual_version': actual_version,
            'keys': keys or []
        })

        # 设置额外属性
        self.key = key
        self.expected_version = expected_version
        self.actual_version = actual_version
        self.keys = keys or []

        super().__init__(
            message=message,
            error_code="CONSISTENCY_ERROR",
            recovery_suggestion="请检查并发访问控制和数据同步机制",
            details=details,
            **kwargs
        )


class CacheQuotaError(CacheError):
    """
    缓存配额异常

    当超出缓存容量或配额限制时抛出。
    """

    def __init__(self,
                 message: str = "超出缓存配额限制",
                 current_usage: int = 0,
                 max_quota: int = 0,
                 details: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # 将额外参数添加到details中
        if details is None:
            details = {}
        details.update({
            'current_usage': current_usage,
            'max_quota': max_quota
        })

        # 设置额外属性
        self.current_usage = current_usage
        self.max_quota = max_quota

        super().__init__(
            message=message,
            error_code="QUOTA_ERROR",
            recovery_suggestion="请清理过期数据或增加配额限制",
            details=details,
            **kwargs
        )


class CacheCapacityError(CacheError):
    """
    缓存容量异常

    当缓存容量设置无效或超出限制时抛出。
    """

    def __init__(self,
                 message: str = "缓存容量设置无效",
                 capacity: int = 0,
                 min_capacity: int = 0,
                 max_capacity: int = 0,
                 details: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # 将额外参数添加到details中
        if details is None:
            details = {}
        details.update({
            'capacity': capacity,
            'min_capacity': min_capacity,
            'max_capacity': max_capacity
        })

        # 设置额外属性
        self.capacity = capacity
        self.min_capacity = min_capacity
        self.max_capacity = max_capacity

        super().__init__(
            message=message,
            error_code="CAPACITY_ERROR",
            recovery_suggestion="请检查容量设置是否在有效范围内",
            details=details,
            **kwargs
        )


class CacheConfigurationError(CacheError):
    """
    缓存配置异常

    当缓存配置无效或不完整时抛出。
    """

    def __init__(self,
                 message: str = "缓存配置无效",
                 config_key: str = "",
                 invalid_value: Any = None,
                 config_value: Any = None,
                 details: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # 将额外参数添加到details中
        if details is None:
            details = {}
        details.update({
            'config_key': config_key,
            'invalid_value': str(invalid_value) if invalid_value is not None else None,
            'config_value': str(config_value) if config_value is not None else None
        })

        # 设置额外属性
        self.config_key = config_key
        self.invalid_value = invalid_value
        self.config_value = config_value

        super().__init__(
            message=message,
            error_code="CONFIGURATION_ERROR",
            recovery_suggestion="请检查配置文件格式和参数值",
            details=details,
            **kwargs
        )


class CacheValueError(CacheError):
    """
    缓存值异常

    当缓存值无效或不符合要求时抛出。
    """

    def __init__(self,
                 message: str = "缓存值无效",
                 value: Any = None,
                 value_type: str = "",
                 size_limit: int = 0,
                 details: Optional[Dict[str, Any]] = None,
                 **kwargs):
        # 将额外参数添加到details中
        if details is None:
            details = {}
        details.update({
            'value': str(value) if value is not None else None,
            'value_type': value_type,
            'size_limit': size_limit
        })

        # 设置额外属性
        self.value = value
        self.value_type = value_type
        self.size_limit = size_limit

        super().__init__(
            message=message,
            error_code="VALUE_ERROR",
            recovery_suggestion="请检查值的类型、大小和格式",
            details=details,
            **kwargs
        )
