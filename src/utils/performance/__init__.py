"""
性能优化工具包

提供全面的性能优化功能，包括：
- 内存管理优化
- 多级缓存系统
- 异步I/O优化
- 自适应批处理
- ML模型缓存和批推理
- Prometheus性能监控

使用示例:
    from src.utils.performance import (
        DynamicMemoryManager,
        MultiLevelCache,
        AsyncDatabasePool,
        AsyncHTTPClient,
        BatchProcessor,
        ModelServingService,
        monitor
    )
"""

# 内存管理
from .memory_manager import (
    DynamicMemoryManager,
    MemoryPool,
    MemoryConfig,
    MemorySnapshot,
)

# 多级缓存
from .multi_level_cache import (
    MultiLevelCache,
    CacheConfig,
    CacheLevel,
    LRUCache,
    cached,
)

# 异步数据库
from .async_database import (
    AsyncDatabasePool,
    PoolConfig,
    Transaction,
)

# 异步HTTP
from .async_http import (
    AsyncHTTPClient,
    HTTPConfig,
    Response,
    cached_http,
)

# 批处理
from .batch_processor import (
    BatchProcessor,
    ParallelBatchProcessor,
    BatchConfig,
    BatchStrategy,
    BatchMetrics,
    batch_process,
    create_batch_processor,
)

# ML推理
from .ml_inference import (
    ModelServingService,
    BatchInferenceEngine,
    ModelConfig,
    ModelCache,
    InferenceResult,
    ModelMetrics,
    ModelFramework,
)

# Prometheus监控
from .prometheus_metrics import (
    MetricsCollector,
    PerformanceMonitor,
    MetricConfig,
    MetricType,
    monitor,
    timed,
    counted,
    timed_block,
    get_metrics_endpoint,
    start_metrics_server,
    create_fastapi_middleware,
)

__version__ = "1.0.0"

__all__ = [
    # 内存管理
    "DynamicMemoryManager",
    "MemoryPool",
    "MemoryConfig",
    "MemorySnapshot",

    # 多级缓存
    "MultiLevelCache",
    "CacheConfig",
    "CacheLevel",
    "LRUCache",
    "cached",

    # 异步数据库
    "AsyncDatabasePool",
    "PoolConfig",
    "Transaction",

    # 异步HTTP
    "AsyncHTTPClient",
    "HTTPConfig",
    "Response",
    "cached_http",

    # 批处理
    "BatchProcessor",
    "ParallelBatchProcessor",
    "BatchConfig",
    "BatchStrategy",
    "BatchMetrics",
    "batch_process",
    "create_batch_processor",

    # ML推理
    "ModelServingService",
    "BatchInferenceEngine",
    "ModelConfig",
    "ModelCache",
    "InferenceResult",
    "ModelMetrics",
    "ModelFramework",

    # Prometheus监控
    "MetricsCollector",
    "PerformanceMonitor",
    "MetricConfig",
    "MetricType",
    "monitor",
    "timed",
    "counted",
    "timed_block",
    "get_metrics_endpoint",
    "start_metrics_server",
    "create_fastapi_middleware",
]
