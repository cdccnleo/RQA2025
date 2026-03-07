"""
Enhanced Data Integration Modules

模块化的增强版数据集成组件。
"""

from .config import IntegrationConfig
from .components import (
    TaskPriority,
    LoadTask,
    EnhancedParallelLoadingManager,
    DynamicThreadPoolManager,
    ConnectionPoolManager,
    MemoryOptimizer,
    FinancialDataOptimizer,
    create_enhanced_loader,
)
from .cache_utils import (
    check_cache_for_symbols,
    check_cache_for_indices,
    check_cache_for_financial,
    cache_data,
    cache_index_data,
    cache_financial_data,
)
from .performance_utils import (
    check_data_quality,
    update_avg_response_time,
    monitor_performance,
    get_integration_stats,
    shutdown,
)
from .integration_manager import EnhancedDataIntegration

__all__ = [
    # 主类
    "EnhancedDataIntegration",
    # 配置
    "IntegrationConfig",
    # 组件
    "TaskPriority",
    "LoadTask",
    "EnhancedParallelLoadingManager",
    "DynamicThreadPoolManager",
    "ConnectionPoolManager",
    "MemoryOptimizer",
    "FinancialDataOptimizer",
    "create_enhanced_loader",
    # 缓存工具
    "check_cache_for_symbols",
    "check_cache_for_indices",
    "check_cache_for_financial",
    "cache_data",
    "cache_index_data",
    "cache_financial_data",
    # 性能工具
    "check_data_quality",
    "update_avg_response_time",
    "monitor_performance",
    "get_integration_stats",
    "shutdown",
]

