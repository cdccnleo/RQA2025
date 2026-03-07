"""缓存模块"""

# redis_cache模块已被重构到distributed中，这里保留向后兼容的导入
try:
    from .distributed.distributed_cache_manager import DistributedCacheManager as RedisCache
    from .distributed.distributed_cache_manager import DistributedCacheManager
    redis_cache = None  # 占位符，保持向后兼容
except ImportError:
    # 如果导入失败，提供空实现
    class RedisCache:
        pass
    class DistributedCacheManager:
        pass
    redis_cache = None

try:
    from .core.cache_manager import UnifiedCacheManager as CacheManager
    from .core.cache_manager import UnifiedCacheManager
    from .monitoring.performance_monitor import PerformanceMonitor as CacheMonitor
    ThreadSafeCache = UnifiedCacheManager
    ThreadSafeTTLCache = UnifiedCacheManager
except ImportError:
    # 提供空实现
    class CacheManager:
        pass
    class ThreadSafeTTLCache:
        pass
    class UnifiedCacheManager:
        pass
    class CacheMonitor:
        pass
    ThreadSafeCache = ThreadSafeTTLCache

# 导入多级缓存组件
try:
    from .multi_tier_cache import (
        MultiTierCache,
        MultiTierCacheConfig,
        CacheStats,
        CacheDecorator,
        CacheTierLevel,
        get_global_cache,
        clear_global_cache,
    )
except ImportError:
    # 提供空实现
    class MultiTierCache:
        pass
    class MultiTierCacheConfig:
        pass
    class CacheStats:
        pass
    class CacheDecorator:
        pass
    class CacheTierLevel:
        pass
    def get_global_cache(config=None):
        pass
    def clear_global_cache():
        pass

__all__ = [
    'RedisCache',
    'redis_cache',
    'ThreadSafeTTLCache',
    'ThreadSafeCache',
    'CacheManager',  # 兼容性别名
    'CacheMonitor',
    'UnifiedCacheManager',  # 统一缓存管理器
    'DistributedCacheManager',  # 分布式缓存管理器
    # 多级缓存组件
    'MultiTierCache',
    'MultiTierCacheConfig',
    'CacheStats',
    'CacheDecorator',
    'CacheTierLevel',
    'get_global_cache',
    'clear_global_cache',
]