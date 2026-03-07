
from typing import Any, Dict, Optional
"""
缓存管理模块异常处理
Cache Management Module Exception Handling

定义缓存管理相关的异常类和错误处理机制
"""


class CacheException(Exception):
    """缓存基础异常类"""

    def __init__(self, message: str, cache_key: Optional[str] = None,
                 operation: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.cache_key = cache_key
        self.operation = operation
        self.details = details or {}
        self.message = message


class CacheNotFoundError(CacheException):
    """缓存未找到错误"""

    def __init__(self, message: str, cache_key: Optional[str] = None, **kwargs):
        super().__init__(message, cache_key=cache_key, **kwargs)


class CacheExpiredError(CacheException):
    """缓存过期错误"""

    def __init__(self, message: str, cache_key: Optional[str] = None,
                 ttl: Optional[int] = None, **kwargs):
        super().__init__(message, cache_key=cache_key, **kwargs)
        self.ttl = ttl


class CacheFullError(CacheException):
    """缓存满错误"""

    def __init__(self, message: str, current_size: Optional[int] = None,
                 max_size: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.current_size = current_size
        self.max_size = max_size


class CacheSerializationError(CacheException):
    """缓存序列化错误"""

    def __init__(self, message: str, data_type: Optional[str] = None,
                 serialization_format: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.data_type = data_type
        self.serialization_format = serialization_format


class CacheConnectionError(CacheException):
    """缓存连接错误"""

    def __init__(self, message: str, host: Optional[str] = None,
                 port: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.host = host
        self.port = port


class CacheTimeoutError(CacheException):
    """缓存超时错误"""

    def __init__(self, message: str, timeout: Optional[float] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, operation=operation, **kwargs)
        self.timeout = timeout


class CacheConsistencyError(CacheException):
    """缓存一致性错误"""

    def __init__(self, message: str, expected_value: Any = None,
                 actual_value: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.expected_value = expected_value
        self.actual_value = actual_value


class CacheConfigurationError(CacheException):
    """缓存配置错误"""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 expected_value: Any = None, actual_value: Any = None, **kwargs):
        super().__init__(message, cache_key=config_key, **kwargs)
        self.config_key = config_key
        self.expected_value = expected_value
        self.actual_value = actual_value


class CachePerformanceError(CacheException):
    """缓存性能错误"""

    def __init__(self, message: str, threshold: Optional[float] = None,
                 actual_value: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.threshold = threshold
        self.actual_value = actual_value


class CacheCorruptionError(CacheException):
    """缓存损坏错误"""

    def __init__(self, message: str, corruption_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.corruption_type = corruption_type


class CacheQuotaExceededError(CacheException):
    """缓存配额超限错误"""

    def __init__(self, message: str, quota_type: Optional[str] = None,
                 current_usage: Optional[int] = None,
                 limit: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.quota_type = quota_type
        self.current_usage = current_usage
        self.limit = limit


class DistributedCacheError(CacheException):
    """分布式缓存错误"""

    def __init__(self, message: str, node_id: Optional[str] = None,
                 cluster_info: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.node_id = node_id
        self.cluster_info = cluster_info or {}


class CacheMigrationError(CacheException):
    """缓存迁移错误"""

    def __init__(self, message: str, source_node: Optional[str] = None,
                 target_node: Optional[str] = None,
                 migration_phase: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.source_node = source_node
        self.target_node = target_node
        self.migration_phase = migration_phase


class CacheBackupError(CacheException):
    """缓存备份错误"""

    def __init__(self, message: str, backup_type: Optional[str] = None,
                 backup_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.backup_type = backup_type
        self.backup_path = backup_path


class CacheRestoreError(CacheException):
    """缓存恢复错误"""

    def __init__(self, message: str, restore_type: Optional[str] = None,
                 restore_path: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.restore_type = restore_type
        self.restore_path = restore_path

# ============================================================================
# 异常处理装饰器
# ============================================================================


def handle_cache_exception(operation: str = "cache_operation"):
    """
    缓存异常处理装饰器

    Args:
        operation: 操作名称，用于日志记录

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CacheException:
                # 重新抛出缓存异常，不做额外处理
                raise
            except Exception as e:
                # 将其他异常包装为缓存异常
                error_msg = f"{operation} 失败: {str(e)}"
                raise CacheException(error_msg, operation=operation,
                                     details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_cache_connection_exception(host: str = "unknown", port: int = 0):
    """
    缓存连接异常处理装饰器

    Args:
        host: 主机地址
        port: 端口号

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CacheConnectionError:
                raise
            except Exception as e:
                error_msg = f"连接缓存服务失败 {host}:{port}: {str(e)}"
                raise CacheConnectionError(error_msg, host=host, port=port,
                                           details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_cache_timeout_exception(timeout: float = 30.0, operation: str = "unknown"):
    """
    缓存超时异常处理装饰器

    Args:
        timeout: 超时时间（秒）
        operation: 操作名称

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CacheTimeoutError:
                raise
            except Exception as e:
                error_msg = f"缓存操作超时 {operation}: {str(e)}"
                raise CacheTimeoutError(error_msg, timeout=timeout, operation=operation,
                                        details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_cache_performance_exception(threshold: float = 0.0, metric: str = "unknown"):
    """
    缓存性能异常处理装饰器

    Args:
        threshold: 性能阈值
        metric: 性能指标名称

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except CachePerformanceError:
                raise
            except Exception as e:
                error_msg = f"缓存性能异常 {metric}: {str(e)}"
                raise CachePerformanceError(error_msg, threshold=threshold,
                                            details={"original_error": str(e), "metric": metric}) from e
        return wrapper
    return decorator
