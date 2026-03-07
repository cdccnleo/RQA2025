"""
配置相关常量定义
"""


class ConfigConstants:
    """配置管理相关常量"""
    
    # 缓存大小
    DEFAULT_CACHE_SIZE = 1024  # 1KB
    MAX_CACHE_SIZE = 1048576  # 1MB
    MIN_CACHE_SIZE = 64  # 64B
    
    # TTL (Time To Live) - 秒
    DEFAULT_TTL = 3600  # 1小时
    MIN_TTL = 60  # 1分钟
    MAX_TTL = 86400  # 24小时
    
    # 清理间隔
    CLEANUP_INTERVAL = 300  # 5分钟
    CLEANUP_BATCH_SIZE = 1000
    
    # 请求超时
    REQUEST_TIMEOUT = 30  # 30秒
    CONNECT_TIMEOUT = 10  # 10秒
    READ_TIMEOUT = 30  # 30秒
    
    # 配置文件
    MAX_CONFIG_FILE_SIZE = 10485760  # 10MB
    CONFIG_WATCH_TIMEOUT = 30  # 30秒
    CONFIG_RELOAD_DELAY = 1  # 1秒
    
    # 重试配置
    MAX_RETRIES = 3
    RETRY_DELAY = 1  # 秒
    RETRY_BACKOFF_FACTOR = 2
    
    # 队列大小
    MAX_QUEUE_SIZE = 100000
    MIN_QUEUE_SIZE = 100
    
    # 线程池
    MIN_THREAD_POOL_SIZE = 2
    MAX_THREAD_POOL_SIZE = 32
    DEFAULT_THREAD_POOL_SIZE = 10
    
    # 缓存TTL策略
    CACHE_TTL_SHORT = 300  # 5分钟
    CACHE_TTL_MEDIUM = 1800  # 30分钟
    CACHE_TTL_LONG = 3600  # 1小时
    CACHE_TTL_EXTENDED = 86400  # 1天
    
    # 版本保留
    VERSION_RETENTION_DAYS = 30
    VERSION_MAX_KEEP = 100

