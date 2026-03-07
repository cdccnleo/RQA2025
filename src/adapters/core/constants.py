"""
适配器层常量定义
Adapters Layer Constants

定义数据适配器相关的常量，避免魔法数字
"""

# 连接参数
DEFAULT_CONNECTION_TIMEOUT = 30    # 默认连接超时时间(秒)
DEFAULT_RETRY_ATTEMPTS = 3        # 默认重试次数
DEFAULT_RETRY_DELAY = 5           # 默认重试延迟(秒)

# 数据适配参数
DEFAULT_BATCH_SIZE = 1000         # 默认批处理大小
MAX_BATCH_SIZE = 10000           # 最大批处理大小
DATA_VALIDATION_TIMEOUT = 60      # 数据验证超时时间(秒)

# 缓存参数
ADAPTER_CACHE_SIZE = 1000         # 适配器缓存大小
CACHE_TTL_SECONDS = 3600          # 缓存过期时间(秒)
CACHE_CLEANUP_INTERVAL = 300      # 缓存清理间隔(秒)

# 监控参数
HEALTH_CHECK_INTERVAL = 60        # 健康检查间隔(秒)
METRICS_UPDATE_INTERVAL = 30      # 指标更新间隔(秒)
ALERT_THRESHOLD_HIGH = 0.9        # 高风险告警阈值

# 市场数据参数
MARKET_DATA_TIMEOUT = 10          # 市场数据超时时间(秒)
PRICE_PRECISION = 4               # 价格精度
VOLUME_PRECISION = 0              # 成交量精度
TIMESTAMP_PRECISION = 6           # 时间戳精度

# API参数
API_REQUEST_TIMEOUT = 15          # API请求超时时间(秒)
API_RATE_LIMIT = 100              # API速率限制(请求/分钟)
API_MAX_CONNECTIONS = 50          # API最大连接数

# 数据转换参数
DATA_TRANSFORM_TIMEOUT = 30       # 数据转换超时时间(秒)
MAX_DATA_FIELDS = 100             # 最大数据字段数
DATA_QUALITY_THRESHOLD = 0.95     # 数据质量阈值

# 同步参数
SYNC_INTERVAL_SECONDS = 60        # 同步间隔(秒)
SYNC_TIMEOUT_SECONDS = 300        # 同步超时时间(秒)
SYNC_BATCH_SIZE = 5000            # 同步批处理大小

# 错误处理参数
MAX_ERROR_COUNT = 100             # 最大错误计数
ERROR_RESET_INTERVAL = 3600       # 错误重置间隔(秒)
CIRCUIT_BREAKER_THRESHOLD = 10    # 熔断器阈值

# 日志参数
LOG_LEVEL_DEFAULT = "INFO"        # 默认日志级别
LOG_MAX_FILE_SIZE_MB = 100        # 最大日志文件大小(MB)
LOG_BACKUP_COUNT = 10             # 日志备份数量

# 性能参数
THROUGHPUT_TARGET_RPS = 1000      # 吞吐量目标(RPS)
LATENCY_TARGET_MS = 100           # 延迟目标(毫秒)
MEMORY_USAGE_THRESHOLD_MB = 512   # 内存使用阈值(MB)

# 配置参数
CONFIG_UPDATE_INTERVAL = 300      # 配置更新间隔(秒)
CONFIG_VALIDATION_TIMEOUT = 10    # 配置验证超时时间(秒)

# 版本参数
ADAPTER_VERSION = "1.0.0"         # 适配器版本
API_VERSION = "v1"                # API版本
PROTOCOL_VERSION = "1.0"          # 协议版本
