"""
异常工具模块
提供基础设施层工具使用的异常类

本模块从core.exceptions重新导出异常类，以保持向后兼容
"""

# 从core.exceptions导入所有异常类
from .core.exceptions import (
    InfrastructureError,
    ConfigurationError,
    DataProcessingError,
    ConnectionError,
    ServiceDiscoveryError,
    HealthCheckError,
    EventBusError,
    MonitoringError,
    LoggingError,
    DataLoaderError,  # datetime_parser需要
    CacheError,
    SecurityError,
    ResourceLimitError,
    ValidationError,
    TimeoutError,
    DataVersionError,
    DependencyError,
)

__all__ = [
    'InfrastructureError',
    'ConfigurationError',
    'DataProcessingError',
    'ConnectionError',
    'ServiceDiscoveryError',
    'HealthCheckError',
    'EventBusError',
    'MonitoringError',
    'LoggingError',
    'DataLoaderError',
    'CacheError',
    'SecurityError',
    'ResourceLimitError',
    'ValidationError',
    'TimeoutError',
    'DataVersionError',
    'DependencyError',
]

