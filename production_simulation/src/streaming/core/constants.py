"""
流处理层常量定义
Streaming Layer Constants

定义流数据处理相关的常量，避免魔法数字
"""

# 数据处理参数
DEFAULT_BATCH_SIZE = 1000          # 默认批处理大小
MAX_BATCH_SIZE = 10000            # 最大批处理大小
MIN_BATCH_SIZE = 10               # 最小批处理大小

# 缓冲区参数
DEFAULT_BUFFER_SIZE = 10000       # 默认缓冲区大小
MAX_BUFFER_SIZE = 100000         # 最大缓冲区大小
MIN_BUFFER_SIZE = 100            # 最小缓冲区大小

# 时间窗口参数
DEFAULT_WINDOW_SIZE_SECONDS = 60  # 默认时间窗口大小(秒)
SLIDING_WINDOW_STEP = 10          # 滑动窗口步长(秒)
MAX_WINDOW_SIZE_MINUTES = 60      # 最大窗口大小(分钟)

# 处理延迟参数
TARGET_PROCESSING_LATENCY_MS = 100  # 目标处理延迟(毫秒)
MAX_PROCESSING_LATENCY_MS = 1000   # 最大处理延迟(毫秒)
LATENCY_CHECK_INTERVAL_MS = 10     # 延迟检查间隔(毫秒)

# 吞吐量参数
TARGET_THROUGHPUT_EVENTS_PER_SEC = 10000  # 目标吞吐量(事件/秒)
MAX_THROUGHPUT_EVENTS_PER_SEC = 50000     # 最大吞吐量(事件/秒)

# 内存管理参数
MEMORY_CHECK_INTERVAL_MS = 1000   # 内存检查间隔(毫秒)
MEMORY_USAGE_THRESHOLD_PCT = 80   # 内存使用率阈值(%)
MAX_MEMORY_USAGE_MB = 1024        # 最大内存使用(MB)

# 连接参数
CONNECTION_TIMEOUT_MS = 5000      # 连接超时时间(毫秒)
RECONNECT_ATTEMPTS = 5            # 重连尝试次数
HEARTBEAT_INTERVAL_SECONDS = 30   # 心跳间隔(秒)

# 数据质量参数
DATA_VALIDATION_TIMEOUT_MS = 100  # 数据验证超时(毫秒)
MAX_DUPLICATE_EVENTS = 1000       # 最大重复事件数
OUTLIER_THRESHOLD_SIGMA = 3.0     # 异常值阈值(标准差倍数)

# 状态管理参数
STATE_CHECKPOINT_INTERVAL_SECONDS = 300  # 状态检查点间隔(秒)
MAX_STATE_SIZE_MB = 512          # 最大状态大小(MB)
STATE_RETENTION_HOURS = 24       # 状态保留时间(小时)

# 事件处理参数
MAX_EVENTS_PER_SECOND = 10000    # 每秒最大事件数
EVENT_PROCESSING_TIMEOUT_MS = 500  # 事件处理超时(毫秒)
EVENT_QUEUE_SIZE = 100000        # 事件队列大小

# 聚合参数
AGGREGATION_WINDOW_SECONDS = 60   # 聚合窗口大小(秒)
MAX_AGGREGATION_KEYS = 1000      # 最大聚合键数
AGGREGATION_UPDATE_INTERVAL_MS = 1000  # 聚合更新间隔(毫秒)

# 监控参数
METRICS_UPDATE_INTERVAL_SECONDS = 10  # 指标更新间隔(秒)
ALERT_THRESHOLD_HIGH = 0.9       # 高风险告警阈值
ALERT_THRESHOLD_MEDIUM = 0.7     # 中风险告警阈值

# 缓存参数
CACHE_SIZE_EVENTS = 10000        # 缓存事件数量
CACHE_TTL_SECONDS = 3600         # 缓存过期时间(秒)
CACHE_CLEANUP_INTERVAL_SECONDS = 300  # 缓存清理间隔(秒)

# 序列化参数
MAX_MESSAGE_SIZE_BYTES = 1048576  # 最大消息大小(1MB)
COMPRESSION_THRESHOLD_BYTES = 1024  # 压缩阈值(1KB)
SERIALIZATION_TIMEOUT_MS = 100   # 序列化超时(毫秒)
