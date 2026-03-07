"""
RQA2025 核心层统一常量定义
Core Layer Constants

为核心层提供统一的常量定义，从各子模块整合常用常量
"""

# 从配置模块导入常量
try:
    from .config.core_constants import *
except ImportError:
    pass

# 基础常量定义
class CoreConstants:
    """核心层基础常量"""
    
    # 时间常量
    DEFAULT_TIMEOUT = 30  # 默认超时时间（秒）
    CONNECTION_TIMEOUT = 30  # 连接超时时间
    QUERY_TIMEOUT = 300  # 查询超时时间
    
    # 重试常量
    MAX_RETRIES = 3  # 最大重试次数
    RETRY_DELAY = 1.0  # 重试延迟（秒）
    
    # 批量操作
    DEFAULT_BATCH_SIZE = 1000  # 默认批量大小
    MAX_BATCH_SIZE = 10000  # 最大批量大小
    
    # 连接池
    MIN_CONNECTIONS = 1  # 最小连接数
    MAX_CONNECTIONS = 10  # 最大连接数
    DEFAULT_POOL_SIZE = 5  # 默认连接池大小
    
    # 缓存
    DEFAULT_CACHE_SIZE = 1000  # 默认缓存大小
    DEFAULT_CACHE_TTL = 3600  # 默认缓存过期时间（秒）
    
    # 性能
    API_RESPONSE_ACCEPTABLE = 0.1  # API响应时间可接受值（秒）
    DEFAULT_QUEUE_SIZE = 1000  # 默认队列大小


# 快捷常量（向后兼容）
DEFAULT_TIMEOUT = CoreConstants.DEFAULT_TIMEOUT
MAX_RETRIES = CoreConstants.MAX_RETRIES
RETRY_DELAY = CoreConstants.RETRY_DELAY
DEFAULT_BATCH_SIZE = CoreConstants.DEFAULT_BATCH_SIZE
DEFAULT_CACHE_SIZE = CoreConstants.DEFAULT_CACHE_SIZE
DEFAULT_POOL_SIZE = CoreConstants.DEFAULT_POOL_SIZE
DEFAULT_QUEUE_SIZE = CoreConstants.DEFAULT_QUEUE_SIZE
CONNECTION_TIMEOUT = CoreConstants.CONNECTION_TIMEOUT
QUERY_TIMEOUT = CoreConstants.QUERY_TIMEOUT

# 导出类常量
MIN_CONNECTIONS = CoreConstants.MIN_CONNECTIONS
MAX_CONNECTIONS = CoreConstants.MAX_CONNECTIONS
MAX_BATCH_SIZE = CoreConstants.MAX_BATCH_SIZE
DEFAULT_CACHE_TTL = CoreConstants.DEFAULT_CACHE_TTL
API_RESPONSE_ACCEPTABLE = CoreConstants.API_RESPONSE_ACCEPTABLE

# 从core_constants导入时间常量（如果存在）
try:
    from .config.core_constants import SECONDS_PER_MINUTE, SECONDS_PER_HOUR, MINUTES_PER_HOUR
except ImportError:
    # 定义基础时间常量
    SECONDS_PER_MINUTE = 60
    SECONDS_PER_HOUR = 3600
    MINUTES_PER_HOUR = 60

__all__ = [
    'CoreConstants',
    'DEFAULT_TIMEOUT',
    'MAX_RETRIES',
    'RETRY_DELAY',
    'DEFAULT_BATCH_SIZE',
    'DEFAULT_CACHE_SIZE',
    'DEFAULT_POOL_SIZE',
    'DEFAULT_QUEUE_SIZE',
    'CONNECTION_TIMEOUT',
    'QUERY_TIMEOUT',
    'MIN_CONNECTIONS',
    'MAX_CONNECTIONS',
    'MAX_BATCH_SIZE',
    'DEFAULT_CACHE_TTL',
    'API_RESPONSE_ACCEPTABLE',
    'SECONDS_PER_MINUTE',
    'SECONDS_PER_HOUR',
    'MINUTES_PER_HOUR',
]

