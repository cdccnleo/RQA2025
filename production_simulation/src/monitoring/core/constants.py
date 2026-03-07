"""
监控层常量定义
Monitoring Layer Constants

定义监控相关的常量，避免魔法数字
"""

# 时间相关常量
DEFAULT_MONITORING_INTERVAL = 60  # 默认监控间隔(秒)
DEFAULT_RETENTION_HOURS = 24      # 默认数据保留时间(小时)
ALERT_CHECK_INTERVAL = 30         # 告警检查间隔(秒)

# 性能阈值常量
CPU_THRESHOLD_HIGH = 80.0         # CPU使用率高阈值(%)
CPU_THRESHOLD_CRITICAL = 95.0     # CPU使用率临界阈值(%)
MEMORY_THRESHOLD_HIGH = 85.0      # 内存使用率高阈值(%)
MEMORY_THRESHOLD_CRITICAL = 95.0  # 内存使用率临界阈值(%)

# 响应时间阈值常量
RESPONSE_TIME_HIGH = 1000         # 响应时间高阈值(ms)
RESPONSE_TIME_CRITICAL = 5000     # 响应时间临界阈值(ms)

# 错误率阈值常量
ERROR_RATE_HIGH = 5.0             # 错误率高阈值(%)
ERROR_RATE_CRITICAL = 10.0        # 错误率临界阈值(%)

# 容量阈值常量
MAX_METRICS_BUFFER = 10000        # 最大指标缓冲区大小
MAX_ALERT_BUFFER = 1000           # 最大告警缓冲区大小
MAX_LOG_ENTRIES = 50000           # 最大日志条目数

# 重试和超时常量
MAX_RETRY_ATTEMPTS = 3            # 最大重试次数
RETRY_DELAY_SECONDS = 5           # 重试延迟(秒)
OPERATION_TIMEOUT = 30            # 操作超时时间(秒)

# 健康检查常量
HEALTH_CHECK_TIMEOUT = 10         # 健康检查超时(秒)
HEALTH_SCORE_THRESHOLD = 70       # 健康评分阈值

# 监控端口常量
DEFAULT_MONITORING_PORT = 8080    # 默认监控端口
DEFAULT_METRICS_PORT = 9090       # 默认指标端口

# 批处理常量
DEFAULT_BATCH_SIZE = 100          # 默认批处理大小
MAX_BATCH_SIZE = 1000             # 最大批处理大小

# 缓存常量
CACHE_TTL_DEFAULT = 300           # 默认缓存过期时间(秒)
CACHE_MAX_SIZE = 1000             # 缓存最大大小

# 告警常量
ALERT_COOLDOWN_MINUTES = 5        # 告警冷却时间(分钟)
MAX_CONSECUTIVE_ALERTS = 10       # 最大连续告警次数
