"""
核心服务层常量定义

集中管理所有魔数，提高代码可维护性。

使用示例:
    from src.core.config.core_constants import MAX_RETRIES, DEFAULT_TIMEOUT
    
    max_retries = MAX_RETRIES
    timeout = DEFAULT_TIMEOUT
"""

# 时间相关常量
SECONDS_PER_MINUTE = 60
SECONDS_PER_HOUR = 3600
SECONDS_PER_DAY = 86400
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24

# 超时相关常量
DEFAULT_TIMEOUT = 30  # 默认超时时间（秒）
DEFAULT_TEST_TIMEOUT = 300  # 默认测试超时时间（秒）
DEFAULT_REQUEST_TIMEOUT = 30  # 默认请求超时时间（秒）
DEFAULT_CONNECTION_TIMEOUT = 10  # 默认连接超时时间（秒）

# 重试相关常量
MAX_RETRIES = 100  # 最大重试次数
DEFAULT_RETRY_DELAY = 1.0  # 默认重试延迟（秒）
MAX_RETRY_DELAY = 60  # 最大重试延迟（秒）

# 批处理相关常量
DEFAULT_BATCH_SIZE = 10  # 默认批处理大小
MAX_BATCH_SIZE = 1000  # 最大批处理大小
MIN_BATCH_SIZE = 1  # 最小批处理大小

# 队列相关常量
MAX_QUEUE_SIZE = 10000  # 最大队列大小
DEFAULT_QUEUE_SIZE = 1000  # 默认队列大小
MIN_QUEUE_SIZE = 10  # 最小队列大小

# 数据相关常量
MAX_RECORDS = 1000  # 最大记录数
MAX_DATA_SIZE_BYTES = 1000000  # 最大数据大小（1MB）
DEFAULT_PAGE_SIZE = 100  # 默认分页大小

# 工作线程相关常量
DEFAULT_MAX_WORKERS = 10  # 默认最大工作线程数
MAX_WORKERS = 100  # 最大工作线程数
MIN_WORKERS = 1  # 最小工作线程数

# 内存相关常量
DEFAULT_MEMORY_LIMIT_MB = 100  # 默认内存限制（MB）
MAX_MEMORY_LIMIT_MB = 1000  # 最大内存限制（MB）

# 性能相关常量
DEFAULT_PERFORMANCE_THRESHOLD = 1000  # 默认性能阈值（毫秒）
MAX_PROCESSING_TIME = 30  # 最大处理时间（秒）

# 监控相关常量
DEFAULT_MONITORING_INTERVAL = 60  # 默认监控间隔（秒）
DEFAULT_CLEANUP_INTERVAL = 3600  # 默认清理间隔（秒）

# 日志相关常量
DEFAULT_LOG_RETENTION_DAYS = 7  # 默认日志保留天数
MAX_LOG_SIZE_MB = 100  # 最大日志文件大小（MB）

# 缓存相关常量
DEFAULT_CACHE_TTL = 3600  # 默认缓存TTL（秒）
MAX_CACHE_SIZE = 1000  # 最大缓存大小

# 事件相关常量
DEFAULT_EVENT_HISTORY_SIZE = 1000  # 默认事件历史大小
MAX_EVENT_HISTORY_SIZE = 10000  # 最大事件历史大小

# 交易相关常量
MAX_ACTIVE_ORDERS = 100  # 最大活跃订单数
DEFAULT_EXECUTION_TIMEOUT = 30  # 默认执行超时时间（秒）
DEFAULT_ORDER_TIMEOUT = 60  # 默认订单超时时间（秒）

