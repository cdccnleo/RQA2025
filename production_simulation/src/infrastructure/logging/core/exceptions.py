
import time

from typing import Any, Dict, Optional
"""
日志管理模块异常处理
Logging Management Module Exception Handling

定义日志管理相关的异常类和错误处理机制
"""

# HTTP状态码常量
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_INTERNAL_ERROR = 500

# 分页常量
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100


class LoggingException(Exception):
    """日志基础异常类"""

    def __init__(self, message: str, logger_name: Optional[str] = None,
                 log_level: Optional[str] = None, details: Optional[Dict[str, Any]] = None, **kwargs):
        super().__init__(message)
        self.logger_name = logger_name
        self.log_level = log_level
        self.details = details or {}
        # 将额外的kwargs合并到details中
        self.details.update(kwargs)
        self.message = message


class LogConfigurationError(LoggingException):
    """日志配置错误"""

    def __init__(self, message: str, config_key: Optional[str] = None,
                 expected_value: Any = None, actual_value: Any = None, **kwargs):
        super().__init__(message, **kwargs)
        self.config_key = config_key
        self.expected_value = expected_value
        self.actual_value = actual_value


class LogHandlerError(LoggingException):
    """日志处理器错误"""

    def __init__(self, message: str, handler_type: Optional[str] = None,
                 handler_name: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.handler_type = handler_type
        self.handler_name = handler_name


class LogFormatterError(LoggingException):
    """日志格式化错误"""

    def __init__(self, message: str, formatter_type: Optional[str] = None,
                 original_message: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.formatter_type = formatter_type
        self.original_message = original_message


class LogFileError(LoggingException):
    """日志文件错误"""

    def __init__(self, message: str, file_path: Optional[str] = None,
                 operation: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.file_path = file_path
        self.operation = operation


class LogRotationError(LogFileError):
    """日志轮转错误"""

    def __init__(self, message: str, rotation_type: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.rotation_type = rotation_type


class LogCompressionError(LogFileError):
    """日志压缩错误"""

    def __init__(self, message: str, compression_algorithm: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.compression_algorithm = compression_algorithm


class LogNetworkError(LoggingException):
    """日志网络错误"""

    def __init__(self, message: str, host: Optional[str] = None,
                 port: Optional[int] = None, protocol: Optional[str] = None, **kwargs):
        self.host = host
        self.port = port
        self.protocol = protocol
        super().__init__(message, **kwargs)


class LogQueueError(LoggingException):
    """日志队列错误"""

    def __init__(self, message: str, queue_size: Optional[int] = None,
                 current_size: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.queue_size = queue_size
        self.current_size = current_size


class LogFilterError(LoggingException):
    """日志过滤错误"""

    def __init__(self, message: str, filter_type: Optional[str] = None,
                 filter_rule: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.filter_type = filter_type
        self.filter_rule = filter_rule


class LogSecurityError(LoggingException):
    """日志安全错误"""

    def __init__(self, message: str, security_issue: Optional[str] = None,
                 sensitive_data: Optional[str] = None, **kwargs):
        self.security_issue = security_issue
        self.sensitive_data = sensitive_data
        super().__init__(message, **kwargs)


class LogMonitorError(LoggingException):
    """日志监控错误"""

    def __init__(self, message: str, metric_name: Optional[str] = None,
                 threshold: Optional[float] = None, actual_value: Optional[float] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.metric_name = metric_name
        self.threshold = threshold
        self.actual_value = actual_value


class LogPerformanceError(LoggingException):
    """日志性能错误"""

    def __init__(self, message: str, operation: Optional[str] = None,
                 duration: Optional[float] = None, threshold: Optional[float] = None, **kwargs):
        self.operation = operation
        self.duration = duration
        self.threshold = threshold
        super().__init__(message, **kwargs)


class LogStorageError(LoggingException):
    """日志存储错误"""

    def __init__(self, message: str, storage_type: Optional[str] = None,
                 storage_path: Optional[str] = None, **kwargs):
        self.storage_type = storage_type
        self.storage_path = storage_path
        super().__init__(message, **kwargs)


class LogAsyncError(LoggingException):
    """日志异步处理错误"""

    def __init__(self, message: str, task_id: Optional[str] = None,
                 worker_id: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.task_id = task_id
        self.worker_id = worker_id


class LogBatchError(LoggingException):
    """日志批量处理错误"""

    def __init__(self, message: str, batch_size: Optional[int] = None,
                 processed_count: Optional[int] = None,
                 failed_count: Optional[int] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.batch_size = batch_size
        self.processed_count = processed_count
        self.failed_count = failed_count


class LogValidationError(LoggingException):
    """日志验证错误"""

    def __init__(self, message: str, validation_rule: Optional[str] = None,
                 invalid_value: Any = None, **kwargs):
        self.validation_rule = validation_rule
        self.invalid_value = invalid_value
        super().__init__(message, **kwargs)


class LogTimeoutError(LoggingException):
    """日志超时错误"""

    def __init__(self, message: str, timeout: Optional[float] = None,
                 operation: Optional[str] = None, **kwargs):
        self.timeout = timeout
        self.operation = operation
        super().__init__(message, **kwargs)


class ResourceError(LoggingException):
    """资源错误"""

    def __init__(self, message: str, resource_type: Optional[str] = None,
                 resource_id: Optional[str] = None, **kwargs):
        self.resource_type = resource_type
        self.resource_id = resource_id
        super().__init__(message, **kwargs)

# ============================================================================
# 异常处理装饰器
# ============================================================================


def handle_logging_exception(operation: str = "logging_operation"):
    """
    日志异常处理装饰器

    Args:
        operation: 操作名称，用于日志记录

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except LoggingException:
                # 重新抛出日志异常，不做额外处理
                raise
            except Exception as e:
                # 将其他异常包装为日志异常
                error_msg = f"{operation} 失败: {str(e)}"
                raise LoggingException(error_msg, operation=operation,
                                       details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_log_file_exception(file_path: str = "unknown", operation: str = "file_operation"):
    """
    日志文件异常处理装饰器

    Args:
        file_path: 文件路径
        operation: 操作名称

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except LogFileError:
                raise
            except Exception as e:
                error_msg = f"文件操作失败 {operation}: {file_path}"
                raise LogFileError(error_msg, file_path=file_path, operation=operation,
                                   details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_log_network_exception(host: str = "unknown", port: int = 0, protocol: str = "unknown"):
    """
    日志网络异常处理装饰器

    Args:
        host: 主机地址
        port: 端口号
        protocol: 协议类型

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except LogNetworkError:
                raise
            except Exception as e:
                error_msg = f"网络连接失败 {protocol}://{host}:{port}: {str(e)}"
                raise LogNetworkError(error_msg, host=host, port=port, protocol=protocol,
                                      details={"original_error": str(e)}) from e
        return wrapper
    return decorator


def handle_log_performance_exception(operation: str = "unknown", threshold: float = 1.0):
    """
    日志性能异常处理装饰器

    Args:
        operation: 操作名称
        threshold: 性能阈值（秒）

    Returns:
        装饰器函数
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()

            try:
                result = func(*args, **kwargs)

                duration = time.time() - start_time
                if duration > threshold:
                    error_msg = f"操作性能超出阈值 {operation}: {duration:.3f}s > {threshold}s"
                    raise LogPerformanceError(error_msg, operation=operation,
                                              duration=duration, threshold=threshold)

                return result

            except LogPerformanceError:
                raise
            except Exception as e:
                duration = time.time() - start_time
                error_msg = f"操作执行失败 {operation}: {str(e)}"
                raise LoggingException(error_msg, operation=operation,
                                       details={"duration": duration, "original_error": str(e)}) from e
        return wrapper
    return decorator
