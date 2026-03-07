"""
constants 模块

提供 constants 相关功能和接口。
"""

#!/usr/bin/env python3
"""
RQA2025 基础设施层常量定义

治理魔法数字，提供统一的常量管理
"""

# ==================== 缓存相关常量 ====================


class CacheConstants:
    """缓存相关常量"""

    # 缓存大小 (语义化命名)
    DEFAULT_CACHE_SIZE = 1024  # 1KB
    ONE_KB = 1024
    ONE_MB = 1048576
    MAX_CACHE_SIZE = ONE_MB  # 1MB
    MIN_CACHE_SIZE = 64  # 64字节

    # TTL时间 (秒) - 语义化时间常量
    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    ONE_HOUR = 3600
    ONE_DAY = 86400
    
    DEFAULT_TTL = ONE_HOUR  # 1小时
    MAX_TTL = ONE_DAY  # 24小时
    MIN_TTL = ONE_MINUTE  # 1分钟

    # 清理间隔 (语义化)
    CLEANUP_INTERVAL = FIVE_MINUTES  # 5分钟
    EVICTION_CHECK_INTERVAL = ONE_MINUTE  # 1分钟

    # 性能阈值
    HIT_RATIO_THRESHOLD = 0.8
    EVICTION_RATIO_THRESHOLD = 0.1

    # 并发控制
    MAX_CONCURRENT_REQUESTS = 100
    REQUEST_TIMEOUT_SECONDS = 30

# ==================== 配置相关常量 ====================


class ConfigConstants:
    """配置相关常量"""

    # 大小单位常量
    TEN_MB = 10 * 1024 * 1024  # 10MB
    
    # 文件大小限制 (语义化)
    MAX_CONFIG_FILE_SIZE = TEN_MB  # 10MB
    MAX_CONFIG_DEPTH = 10  # 最大嵌套深度

    # 时间间隔 (语义化)
    ONE_MINUTE = 60
    THIRTY_SECONDS = 30
    
    CONFIG_REFRESH_INTERVAL = ONE_MINUTE  # 刷新间隔
    CONFIG_WATCH_TIMEOUT = THIRTY_SECONDS  # 监听超时

    # 重试次数
    CONFIG_LOAD_MAX_RETRIES = 3
    CONFIG_SAVE_MAX_RETRIES = 3

    # 缓存大小
    CONFIG_CACHE_SIZE = 1000

# ==================== 监控相关常量 ====================


class MonitoringConstants:
    """监控相关常量"""

    # 时间间隔 (语义化)
    THIRTY_SECONDS = 30
    ONE_MINUTE = 60
    
    DEFAULT_MONITOR_INTERVAL = THIRTY_SECONDS  # 默认监控间隔
    HEALTH_CHECK_INTERVAL = THIRTY_SECONDS  # 健康检查间隔
    METRICS_COLLECTION_INTERVAL = ONE_MINUTE  # 指标收集间隔

    # 阈值 (百分比)
    CPU_USAGE_THRESHOLD_PERCENT = 80.0  # CPU使用率阈值
    MEMORY_USAGE_THRESHOLD_PERCENT = 85.0  # 内存使用率阈值
    DISK_USAGE_THRESHOLD_PERCENT = 90.0  # 磁盘使用率阈值

    # 队列大小 (语义化)
    TEN_THOUSAND = 10000
    ONE_THOUSAND = 1000
    
    MAX_METRICS_QUEUE_SIZE = TEN_THOUSAND  # 最大指标队列大小
    MAX_ALERT_QUEUE_SIZE = ONE_THOUSAND  # 最大告警队列大小

    # 保留时间 (天数)
    THIRTY_DAYS = 30
    NINETY_DAYS = 90
    
    METRICS_RETENTION_DAYS = THIRTY_DAYS  # 指标保留时间
    LOG_RETENTION_DAYS = NINETY_DAYS  # 日志保留时间

# ==================== 资源管理常量 ====================


class ResourceConstants:
    """资源管理常量"""

    # 时间常量
    THIRTY_SECONDS = 30
    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    
    # 大小常量
    ONE_HUNDRED_THOUSAND = 100000

    # 连接池 (语义化)
    DEFAULT_POOL_SIZE = 10  # 默认连接池大小
    MAX_POOL_SIZE = 100  # 最大连接池大小
    MIN_POOL_SIZE = 1  # 最小连接池大小
    POOL_TIMEOUT = THIRTY_SECONDS  # 连接池超时

    # 队列 (语义化)
    DEFAULT_QUEUE_SIZE = 1000  # 默认队列大小
    MAX_QUEUE_SIZE = ONE_HUNDRED_THOUSAND  # 最大队列大小
    QUEUE_TIMEOUT = ONE_MINUTE  # 队列超时

    # 线程池 (语义化)
    DEFAULT_THREAD_POOL_SIZE = 4  # 默认线程池大小
    MAX_THREAD_POOL_SIZE = 32  # 最大线程池大小
    THREAD_TIMEOUT = FIVE_MINUTES  # 线程超时

# ==================== 网络相关常量 ====================


class NetworkConstants:
    """网络相关常量"""

    # 时间常量
    THIRTY_SECONDS = 30
    ONE_MINUTE = 60
    FIVE_MINUTES = 300
    
    # 大小常量
    EIGHT_KB = 8192
    ONE_MB = 1048576

    # 端口范围
    MIN_PORT = 1024  # 最小端口号
    MAX_PORT = 65535  # 最大端口号
    DEFAULT_PORT = 8080  # 默认端口

    # 超时时间 (语义化)
    CONNECTION_TIMEOUT = THIRTY_SECONDS  # 连接超时
    READ_TIMEOUT = ONE_MINUTE  # 读取超时
    WRITE_TIMEOUT = ONE_MINUTE  # 写入超时

    # 重试配置
    MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
    RETRY_BACKOFF_FACTOR = 2.0  # 重试退避因子
    RETRY_MAX_DELAY = FIVE_MINUTES  # 最大重试延迟

    # 缓冲区 (语义化)
    DEFAULT_BUFFER_SIZE = EIGHT_KB  # 默认缓冲区大小
    MAX_BUFFER_SIZE = ONE_MB  # 最大缓冲区大小

# ==================== 安全相关常量 ====================


class SecurityConstants:
    """安全相关常量"""

    # 时间常量
    ONE_MINUTE = 60
    ONE_HOUR = 3600
    NINETY_DAYS = 90

    # 密码策略 (语义化)
    MIN_PASSWORD_LENGTH = 8  # 最小密码长度
    MAX_PASSWORD_LENGTH = 128  # 最大密码长度
    PASSWORD_EXPIRY_DAYS = NINETY_DAYS  # 密码过期天数

    # 会话管理 (语义化)
    SESSION_TIMEOUT = ONE_HOUR  # 会话超时时间
    MAX_SESSIONS_PER_USER = 5  # 每用户最大会话数

    # 加密配置 (位数)
    ENCRYPTION_KEY_SIZE = 256  # 加密密钥大小(位)
    SALT_SIZE = 32  # 盐值大小(字节)
    IV_SIZE = 16  # 初始化向量大小(字节)

    # 速率限制 (语义化)
    RATE_LIMIT_REQUESTS = 100  # 速率限制请求数
    RATE_LIMIT_WINDOW = ONE_MINUTE  # 速率限制时间窗口

# ==================== 数据库相关常量 ====================


class DatabaseConstants:
    """数据库相关常量"""

    # 时间常量
    THIRTY_SECONDS = 30
    FIVE_MINUTES = 300
    
    # 大小常量
    TEN_THOUSAND = 10000

    # 连接池 (语义化)
    DB_POOL_SIZE = 10  # 数据库连接池大小
    DB_MAX_POOL_SIZE = 50  # 最大数据库连接池大小
    DB_POOL_TIMEOUT = THIRTY_SECONDS  # 连接池超时

    # 查询配置 (语义化)
    QUERY_TIMEOUT = FIVE_MINUTES  # 查询超时时间
    MAX_QUERY_ROWS = TEN_THOUSAND  # 最大查询行数

    # 事务配置 (语义化)
    TRANSACTION_TIMEOUT = FIVE_MINUTES  # 事务超时时间
    MAX_TRANSACTION_RETRIES = 3  # 最大事务重试次数

# ==================== 文件系统常量 ====================


class FileSystemConstants:
    """文件系统常量"""

    # 大小常量
    EIGHT_KB = 8192
    ONE_HUNDRED_MB = 100 * 1024 * 1024  # 100MB

    # 文件大小 (语义化)
    MAX_FILE_SIZE = ONE_HUNDRED_MB  # 最大文件大小
    CHUNK_SIZE = EIGHT_KB  # 块大小

    # 权限 (八进制)
    DEFAULT_FILE_PERMISSIONS = 0o644  # 默认文件权限(rw-r--r--)
    DEFAULT_DIR_PERMISSIONS = 0o755  # 默认目录权限(rwxr-xr-x)

    # 路径长度限制
    MAX_PATH_LENGTH = 4096  # 最大路径长度
    MAX_FILENAME_LENGTH = 255  # 最大文件名长度

# ==================== 时间相关常量 ====================


class TimeConstants:
    """时间相关常量"""

    # 基本单位
    MILLISECONDS_PER_SECOND = 1000
    MICROSECONDS_PER_SECOND = 1000000
    NANOSECONDS_PER_SECOND = 1000000000

    # 时间格式
    ISO_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"
    DATE_FORMAT = "%Y-%m-%d"
    TIME_FORMAT = "%H:%M:%S"

    # 时区
    UTC_OFFSET = 0

# ==================== 通用常量 ====================


class CommonConstants:
    """通用常量"""

    # 布尔值
    TRUE_VALUES = ['true', 'True', 'TRUE', '1', 'yes', 'Yes', 'YES']
    FALSE_VALUES = ['false', 'False', 'FALSE', '0', 'no', 'No', 'NO']

    # 编码
    DEFAULT_ENCODING = 'utf-8'
    FALLBACK_ENCODING = 'latin-1'

    # 分隔符
    PATH_SEPARATOR = '/'
    LINE_SEPARATOR = '\n'
    FIELD_SEPARATOR = ','

    # 空值
    EMPTY_STRING = ''
    EMPTY_LIST = []
    EMPTY_DICT = {}

# ==================== 日志相关常量 ====================


class LoggingConstants:
    """日志相关常量"""

    # 日志文件大小
    MAX_LOG_FILE_SIZE = 104857600  # 100MB
    LOG_ROTATION_COUNT = 10

    # 日志保留时间
    LOG_RETENTION_DAYS = 30
    ARCHIVE_RETENTION_DAYS = 365

    # 队列大小
    LOG_QUEUE_SIZE = 10000
    ASYNC_LOG_QUEUE_SIZE = 50000

    # 性能阈值
    LOG_THROUGHPUT_THRESHOLD = 1000  # logs/second
    LOG_LATENCY_THRESHOLD = 100  # milliseconds

# ==================== 监控相关常量 ====================


class HealthConstants:
    """健康检查相关常量"""

    # 健康状态
    HEALTH_CHECK_TIMEOUT = 30
    HEALTH_DEGRADED_THRESHOLD = 80.0  # 80%以下为降级
    HEALTH_UNHEALTHY_THRESHOLD = 50.0  # 50%以下为不健康

    # 检查间隔
    BASIC_CHECK_INTERVAL = 30
    DETAILED_CHECK_INTERVAL = 300  # 5分钟

    # 重试配置
    HEALTH_CHECK_RETRIES = 3
    HEALTH_RETRY_DELAY = 5

# ==================== 配置相关常量 ====================


class ConfigConstants:
    """配置管理相关常量"""

    # 配置大小限制
    MAX_CONFIG_SIZE = 10485760  # 10MB
    MAX_KEY_LENGTH = 1000
    MAX_VALUE_LENGTH = 1048576  # 1MB

    # 配置层级
    MAX_CONFIG_DEPTH = 10
    MAX_ARRAY_SIZE = 1000

    # 刷新和缓存
    CONFIG_CACHE_TTL = 300  # 5分钟
    CONFIG_REFRESH_INTERVAL = 60  # 1分钟

    # 验证规则
    MAX_VALIDATION_RULES = 100
    VALIDATION_TIMEOUT = 10  # 10秒

# ==================== 资源管理常量 ====================


class ResourceLimits:
    """资源限制常量"""

    # CPU使用率
    CPU_WARNING_THRESHOLD = 75.0
    CPU_CRITICAL_THRESHOLD = 90.0

    # 内存使用率
    MEMORY_WARNING_THRESHOLD = 80.0
    MEMORY_CRITICAL_THRESHOLD = 95.0

    # 磁盘使用率
    DISK_WARNING_THRESHOLD = 85.0
    DISK_CRITICAL_THRESHOLD = 95.0

    # 网络使用率
    NETWORK_WARNING_THRESHOLD = 80.0
    NETWORK_CRITICAL_THRESHOLD = 90.0

# ==================== 性能基准常量 ====================


class PerformanceBenchmarks:
    """性能基准常量"""

    # 响应时间基准 (毫秒)
    API_RESPONSE_FAST = 100
    API_RESPONSE_ACCEPTABLE = 500
    API_RESPONSE_SLOW = 2000

    # 吞吐量基准 (requests/second)
    HIGH_THROUGHPUT = 1000
    MEDIUM_THROUGHPUT = 500
    LOW_THROUGHPUT = 100

    # 内存使用基准 (MB)
    LOW_MEMORY_USAGE = 100
    MEDIUM_MEMORY_USAGE = 500
    HIGH_MEMORY_USAGE = 1000

    # CPU使用基准 (%)
    LOW_CPU_USAGE = 20
    MEDIUM_CPU_USAGE = 50
    HIGH_CPU_USAGE = 80

# ==================== 错误处理常量 ====================


class ErrorConstants:
    """错误处理相关常量"""

    # 时间常量
    FIVE_SECONDS = 5
    THIRTY_SECONDS = 30
    ONE_MINUTE = 60
    FIVE_MINUTES = 300

    # 重试配置 (语义化)
    MAX_RETRY_ATTEMPTS = 3  # 最大重试次数
    BASE_RETRY_DELAY_SECONDS = 1.0  # 基础重试延迟(秒)
    MAX_RETRY_DELAY_SECONDS = ONE_MINUTE  # 最大重试延迟(秒)

    # 超时配置 (语义化)
    DEFAULT_TIMEOUT = THIRTY_SECONDS  # 默认超时
    LONG_RUNNING_TIMEOUT = FIVE_MINUTES  # 长时运行超时
    SHORT_TIMEOUT = FIVE_SECONDS  # 短超时

    # 错误率阈值 (百分比)
    ERROR_RATE_WARNING_PERCENT = 5.0  # 警告阈值(5%)
    ERROR_RATE_CRITICAL_PERCENT = 10.0  # 危险阈值(10%)

# ==================== 消息和通知常量 ====================


class NotificationConstants:
    """消息和通知相关常量"""

    # 时间常量
    FIVE_MINUTES = 300

    # 告警级别 (字符串常量)
    ALERT_LEVEL_INFO = "info"
    ALERT_LEVEL_WARNING = "warning"
    ALERT_LEVEL_ERROR = "error"
    ALERT_LEVEL_CRITICAL = "critical"

    # 通知渠道 (字符串常量)
    CHANNEL_EMAIL = "email"
    CHANNEL_SMS = "sms"
    CHANNEL_SLACK = "slack"
    CHANNEL_WEBHOOK = "webhook"

    # 通知频率控制 (语义化)
    MAX_NOTIFICATIONS_PER_HOUR = 10  # 每小时最大通知数
    NOTIFICATION_COOLDOWN_SECONDS = FIVE_MINUTES  # 通知冷却时间(5分钟)

# ==================== 向后兼容性 ====================


# 为方便使用，提供常用常量的快捷访问
DEFAULT_TIMEOUT = NetworkConstants.CONNECTION_TIMEOUT
DEFAULT_CACHE_SIZE = CacheConstants.DEFAULT_CACHE_SIZE
DEFAULT_POOL_SIZE = ResourceConstants.DEFAULT_POOL_SIZE
DEFAULT_QUEUE_SIZE = ResourceConstants.DEFAULT_QUEUE_SIZE
MAX_RETRY_ATTEMPTS = NetworkConstants.MAX_RETRY_ATTEMPTS
RETRY_BACKOFF_FACTOR = NetworkConstants.RETRY_BACKOFF_FACTOR

# 新增快捷常量
DEFAULT_LOG_RETENTION = LoggingConstants.LOG_RETENTION_DAYS
HEALTH_CHECK_TIMEOUT = HealthConstants.HEALTH_CHECK_TIMEOUT
CONFIG_CACHE_TTL = ConfigConstants.CONFIG_CACHE_TTL
CPU_WARNING_THRESHOLD = ResourceLimits.CPU_WARNING_THRESHOLD
API_RESPONSE_ACCEPTABLE = PerformanceBenchmarks.API_RESPONSE_ACCEPTABLE

__all__ = [
    # 常量类
    'CacheConstants',
    'ConfigConstants',
    'MonitoringConstants',
    'ResourceConstants',
    'NetworkConstants',
    'SecurityConstants',
    'DatabaseConstants',
    'FileSystemConstants',
    'TimeConstants',
    'CommonConstants',
    'LoggingConstants',
    'HealthConstants',
    'ResourceLimits',
    'PerformanceBenchmarks',
    'ErrorConstants',
    'NotificationConstants',

    # 快捷常量
    'DEFAULT_TIMEOUT',
    'DEFAULT_CACHE_SIZE',
    'DEFAULT_POOL_SIZE',
    'DEFAULT_QUEUE_SIZE',
    'MAX_RETRY_ATTEMPTS',
    'RETRY_BACKOFF_FACTOR',
    'DEFAULT_LOG_RETENTION',
    'HEALTH_CHECK_TIMEOUT',
    'CONFIG_CACHE_TTL',
    'CPU_WARNING_THRESHOLD',
    'API_RESPONSE_ACCEPTABLE',
]
