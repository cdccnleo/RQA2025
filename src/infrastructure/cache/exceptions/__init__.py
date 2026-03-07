"""
__init__ 模块

提供 __init__ 相关功能和接口。
"""

"""
缓存异常定义

定义缓存系统相关的异常类：
- 缓存异常
- 配置异常
- 连接异常
"""


from .cache_exceptions import (
    CacheError, CacheConnectionError, CacheConsistencyError,
    CacheConfigurationError, CacheTimeoutError, CacheSerializationError,
    CacheKeyError, CacheQuotaError, CacheValueError
)

__all__ = [
    'CacheError', 'CacheConnectionError', 'CacheConsistencyError',
    'CacheConfigurationError', 'CacheTimeoutError', 'CacheSerializationError',
    'CacheKeyError', 'CacheQuotaError', 'CacheValueError'
]
