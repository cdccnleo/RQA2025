"""
分布式缓存管理器模块（别名模块）
提供向后兼容的导入路径

实际实现在 distributed/distributed_cache_manager.py 中
"""

try:
    from .distributed.distributed_cache_manager import (
        DistributedCacheManager,
        DistributedConfig,
        ClusterNode,
        SyncStrategy,
        SyncMode
    )
    from ..interfaces import ConsistencyLevel
except ImportError:
    # 提供基础实现
    class DistributedCacheManager:
        pass
    
    class DistributedConfig:
        pass
    
    class ClusterNode:
        pass
    
    class ConsistencyLevel:
        pass
    
    class SyncStrategy:
        pass
    
    class SyncMode:
        pass

__all__ = [
    'DistributedCacheManager',
    'DistributedConfig',
    'ClusterNode',
    'ConsistencyLevel',
    'SyncStrategy',
    'SyncMode'
]

