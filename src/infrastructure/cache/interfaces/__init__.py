
from .base_component_interface import IBaseComponent
from .cache_interfaces import ICacheComponent
from .consistency_checker import InterfaceConsistencyChecker
from .data_structures import CacheEntry, CacheStats, PerformanceMetrics, CacheEvictionStrategy, AccessPattern, ConsistencyLevel
from .global_interfaces import ICacheStrategy
"""
缓存系统接口定义

统一接口规范和数据结构定义：
- 组件接口
- 缓存接口
- 数据结构
- 一致性检查工具
"""

# CacheEntry, CacheStats, PerformanceMetrics,
# CacheEvictionStrategy, AccessPattern, ConsistencyLevel
__all__ = [
    # 基础接口
    'IBaseComponent',

    # 缓存接口
    'ICacheComponent', 'IAdvancedCacheComponent', 'ICacheStrategy',
    'ICacheManager', 'IConsistencyManager',

    # 数据结构
    'CacheEntry', 'CacheStats', 'PerformanceMetrics',
    'CacheEvictionStrategy', 'AccessPattern', 'ConsistencyLevel',

    # 工具类
    'InterfaceConsistencyChecker'
]
