"""数据库模块

提供数据库连接池管理和查询优化功能。
"""

# 导入连接池管理器
try:
    from .connection_pool_manager import (
        ConnectionPoolManager,
        PoolConfig,
        PoolStats,
        ConnectionInfo,
        PoolStatus,
    )
except ImportError:
    class ConnectionPoolManager:
        pass
    class PoolConfig:
        pass
    class PoolStats:
        pass
    class ConnectionInfo:
        pass
    class PoolStatus:
        pass

# 导入查询优化器
try:
    from .query_optimizer import (
        QueryOptimizer,
        QueryOptimizerConfig,
        QueryCache,
        QueryAnalyzer,
        QueryPlan,
        QueryStats,
        IndexSuggestion,
        QueryType,
        OptimizationStrategy,
        get_global_optimizer,
        clear_global_optimizer,
    )
except ImportError:
    class QueryOptimizer:
        pass
    class QueryOptimizerConfig:
        pass
    class QueryCache:
        pass
    class QueryAnalyzer:
        pass
    class QueryPlan:
        pass
    class QueryStats:
        pass
    class IndexSuggestion:
        pass
    class QueryType:
        pass
    class OptimizationStrategy:
        pass
    
    async def get_global_optimizer(config=None):
        pass
    
    async def clear_global_optimizer():
        pass

__all__ = [
    # 连接池管理
    'ConnectionPoolManager',
    'PoolConfig',
    'PoolStats',
    'ConnectionInfo',
    'PoolStatus',
    # 查询优化器
    'QueryOptimizer',
    'QueryOptimizerConfig',
    'QueryCache',
    'QueryAnalyzer',
    'QueryPlan',
    'QueryStats',
    'IndexSuggestion',
    'QueryType',
    'OptimizationStrategy',
    'get_global_optimizer',
    'clear_global_optimizer',
]
