
# ============================================================================
# 缓存大小和容量相关常量
# 默认缓存大小

from typing import Final
from enum import Enum
"""
缓存管理模块常量定义
Cache Management Module Constants

定义缓存管理相关的常量，避免魔法数字
"""

DEFAULT_CACHE_SIZE: Final[int] = 1000
MAX_CACHE_SIZE: Final[int] = 100000
MIN_CACHE_SIZE: Final[int] = 100

# 内存缓存大小
DEFAULT_MEMORY_CACHE_SIZE: Final[int] = 10000
MAX_MEMORY_CACHE_SIZE: Final[int] = 1000000
MIN_MEMORY_CACHE_SIZE: Final[int] = 100

# 分布式缓存大小
DEFAULT_DISTRIBUTED_CACHE_SIZE: Final[int] = 100000
MAX_DISTRIBUTED_CACHE_SIZE: Final[int] = 10000000
MIN_DISTRIBUTED_CACHE_SIZE: Final[int] = 1000

# ============================================================================
# 缓存TTL（生存时间）相关常量（秒）
# ============================================================================

# 默认TTL
DEFAULT_CACHE_TTL: Final[int] = 3600  # 1小时
MAX_CACHE_TTL: Final[int] = 2592000   # 30天
MIN_CACHE_TTL: Final[int] = 60        # 1分钟

# 短TTL
SHORT_CACHE_TTL: Final[int] = 300     # 5分钟
MEDIUM_CACHE_TTL: Final[int] = 1800   # 30分钟
LONG_CACHE_TTL: Final[int] = 86400    # 24小时

# ============================================================================
# 多级缓存相关常量
# ============================================================================

# 缓存层级数量
MAX_CACHE_LEVELS: Final[int] = 5
DEFAULT_CACHE_LEVELS: Final[int] = 3

# 缓存命中率阈值
CACHE_HIT_RATE_WARNING: Final[float] = 0.7
CACHE_HIT_RATE_CRITICAL: Final[float] = 0.5
CACHE_HIT_RATE_EXCELLENT: Final[float] = 0.9

# ============================================================================
# 性能监控相关常量
# ============================================================================

# 响应时间阈值（毫秒）
CACHE_RESPONSE_TIME_WARNING: Final[int] = 100
CACHE_RESPONSE_TIME_CRITICAL: Final[int] = 500
CACHE_RESPONSE_TIME_EXCELLENT: Final[int] = 10

# 性能基准线
PERFORMANCE_BASELINE_PERIOD: Final[int] = 3600  # 1小时
ANOMALY_DETECTION_WINDOW: Final[int] = 300      # 5分钟

# ============================================================================
# 清理和维护相关常量
# ============================================================================

# 清理间隔（秒）
DEFAULT_CLEANUP_INTERVAL: Final[int] = 300      # 5分钟
MIN_CLEANUP_INTERVAL: Final[int] = 60           # 1分钟
MAX_CLEANUP_INTERVAL: Final[int] = 3600         # 1小时

# 清理阈值
CLEANUP_THRESHOLD_RATIO: Final[float] = 0.8     # 80%容量时触发清理
FORCE_CLEANUP_RATIO: Final[float] = 0.95        # 95%容量时强制清理

# ============================================================================
# 连接和网络相关常量
# ============================================================================

# Redis连接池
DEFAULT_REDIS_POOL_SIZE: Final[int] = 10
MAX_REDIS_POOL_SIZE: Final[int] = 100
MIN_REDIS_POOL_SIZE: Final[int] = 1

# 连接超时
DEFAULT_CONNECTION_TIMEOUT: Final[int] = 5
MAX_CONNECTION_TIMEOUT: Final[int] = 30
MIN_CONNECTION_TIMEOUT: Final[int] = 1

# 重试配置
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[float] = 0.1
MAX_RETRY_DELAY: Final[float] = 1.0

# ============================================================================
# 序列化和压缩相关常量
# ============================================================================

# 压缩阈值（字节）
COMPRESSION_THRESHOLD: Final[int] = 1024        # 1KB以上压缩
MAX_COMPRESSION_SIZE: Final[int] = 10485760     # 10MB

# 序列化格式
DEFAULT_SERIALIZATION_FORMAT: Final[str] = "json"
SUPPORTED_SERIALIZATION_FORMATS: Final[list] = ["json", "pickle", "msgpack"]

# ============================================================================
# 监控和告警相关常量
# ============================================================================

# 监控间隔（秒）
DEFAULT_MONITOR_INTERVAL: Final[int] = 60
MIN_MONITOR_INTERVAL: Final[int] = 10
MAX_MONITOR_INTERVAL: Final[int] = 300

# 告警阈值
ALERT_HIGH_THRESHOLD: Final[float] = 0.9
ALERT_MEDIUM_THRESHOLD: Final[float] = 0.7
ALERT_LOW_THRESHOLD: Final[float] = 0.5

# ============================================================================
# 统计和度量相关常量
# ============================================================================

# 统计窗口大小
STATISTICS_WINDOW_SIZE: Final[int] = 1000
STATISTICS_ROLLING_WINDOW: Final[int] = 100

# 百分位数计算
PERCENTILE_50: Final[float] = 50.0
PERCENTILE_95: Final[float] = 95.0
PERCENTILE_99: Final[float] = 99.0

# ============================================================================
# 一致性检查相关常量
# ============================================================================

# 一致性检查间隔
CONSISTENCY_CHECK_INTERVAL: Final[int] = 300    # 5分钟
CONSISTENCY_TIMEOUT: Final[int] = 30            # 30秒

# 一致性阈值
CONSISTENCY_THRESHOLD_HIGH: Final[float] = 0.99
CONSISTENCY_THRESHOLD_MEDIUM: Final[float] = 0.95
CONSISTENCY_THRESHOLD_LOW: Final[float] = 0.90

# ============================================================================
# 其他通用常量
# ============================================================================

# 分页和批量操作
DEFAULT_BATCH_SIZE: Final[int] = 100
MAX_BATCH_SIZE: Final[int] = 1000
MIN_BATCH_SIZE: Final[int] = 10

# 并发控制
DEFAULT_MAX_CONCURRENT: Final[int] = 10
MAX_CONCURRENT_OPERATIONS: Final[int] = 100

# 版本控制
CACHE_VERSION_PREFIX: Final[str] = "v"
DEFAULT_CACHE_VERSION: Final[str] = "1.0.0"

# 键前缀
CACHE_KEY_SEPARATOR: Final[str] = ":"
DEFAULT_CACHE_PREFIX: Final[str] = "cache"

# ============================================================================
# 缓存策略相关常量
# ============================================================================

# LRU策略
LRU_DEFAULT_CAPACITY: Final[int] = 1000

# LFU策略
LFU_DEFAULT_CAPACITY: Final[int] = 1000
LFU_MIN_FREQUENCY: Final[int] = 1

# TTL策略
TTL_CLEANUP_BATCH_SIZE: Final[int] = 100

# ============================================================================
# 错误码和状态码
# ============================================================================

# 缓存操作状态码
CACHE_STATUS_SUCCESS: Final[int] = 200
CACHE_STATUS_NOT_FOUND: Final[int] = 404
CACHE_STATUS_ERROR: Final[int] = 500
CACHE_STATUS_TIMEOUT: Final[int] = 408
CACHE_STATUS_CONFLICT: Final[int] = 409

# 缓存操作错误码
ERROR_CACHE_FULL: Final[int] = 1001
ERROR_CACHE_EXPIRED: Final[int] = 1002
ERROR_CACHE_CORRUPTED: Final[int] = 1003
ERROR_CACHE_SERIALIZATION: Final[int] = 1004
ERROR_CACHE_DESERIALIZATION: Final[int] = 1005

class CacheLevel(Enum):
    """缓存层级枚举"""
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    HYBRID = "hybrid"
