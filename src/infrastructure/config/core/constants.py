
# ============================================================================
# 文件和路径相关常量
# 默认配置文件名

from typing import Final
"""
配置管理模块常量定义
Config Management Module Constants

定义配置管理相关的常量，避免魔法数字
"""

DEFAULT_CONFIG_FILENAME: Final[str] = "config.yaml"
DEFAULT_CONFIG_FILE_JSON: Final[str] = "config.json"
DEFAULT_CONFIG_FILE_TOML: Final[str] = "config.toml"
DEFAULT_CONFIG_FILE_ENV: Final[str] = ".env"

# 配置目录路径
DEFAULT_CONFIG_DIR: Final[str] = "config"
DEFAULT_CONFIG_SUBDIR: Final[str] = "conf.d"

# ============================================================================
# 时间和超时相关常量
# ============================================================================

# 默认超时时间（秒）
DEFAULT_CONFIG_TIMEOUT: Final[int] = 30
DEFAULT_LOAD_TIMEOUT: Final[int] = 10
DEFAULT_CACHE_TIMEOUT: Final[int] = 300  # 5分钟

# 重试相关
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[float] = 1.0
DEFAULT_RETRY_BACKOFF: Final[float] = 2.0

# ============================================================================
# 缓存相关常量
# ============================================================================

# 缓存大小限制
DEFAULT_CACHE_SIZE: Final[int] = 1000
MAX_CACHE_SIZE: Final[int] = 10000
MIN_CACHE_SIZE: Final[int] = 10

# 缓存TTL（秒）
DEFAULT_CACHE_TTL: Final[int] = 3600  # 1小时
MAX_CACHE_TTL: Final[int] = 86400    # 24小时
MIN_CACHE_TTL: Final[int] = 60       # 1分钟

# ============================================================================
# 验证相关常量
# ============================================================================

# 字符串长度限制
MIN_CONFIG_KEY_LENGTH: Final[int] = 1
MAX_CONFIG_KEY_LENGTH: Final[int] = 255
MIN_CONFIG_VALUE_LENGTH: Final[int] = 0
MAX_CONFIG_VALUE_LENGTH: Final[int] = 10000

# 数值范围限制
MIN_PORT_NUMBER: Final[int] = 1
MAX_PORT_NUMBER: Final[int] = 65535
MIN_TIMEOUT_VALUE: Final[int] = 1
MAX_TIMEOUT_VALUE: Final[int] = 3600

# ============================================================================
# 监控和性能相关常量
# ============================================================================

# 性能阈值（毫秒）
PERFORMANCE_WARNING_THRESHOLD: Final[int] = 100
PERFORMANCE_ERROR_THRESHOLD: Final[int] = 500
PERFORMANCE_CRITICAL_THRESHOLD: Final[int] = 2000

# 监控间隔（秒）
DEFAULT_MONITOR_INTERVAL: Final[int] = 60
MIN_MONITOR_INTERVAL: Final[int] = 5
MAX_MONITOR_INTERVAL: Final[int] = 3600

# ============================================================================
# 版本管理相关常量
# ============================================================================

# 版本号格式
VERSION_PATTERN: Final[str] = r'^\d+\.\d+\.\d+$'
MAX_VERSION_HISTORY: Final[int] = 100
DEFAULT_VERSION_RETENTION_DAYS: Final[int] = 30

# ============================================================================
# 安全相关常量
# ============================================================================

# 加密相关
DEFAULT_ENCRYPTION_ALGORITHM: Final[str] = "AES256"
DEFAULT_KEY_SIZE: Final[int] = 256
MIN_KEY_SIZE: Final[int] = 128
MAX_KEY_SIZE: Final[int] = 512

# 访问控制
DEFAULT_SESSION_TIMEOUT: Final[int] = 3600  # 1小时
MAX_CONCURRENT_SESSIONS: Final[int] = 100
RATE_LIMIT_REQUESTS_PER_MINUTE: Final[int] = 60

# ============================================================================
# 存储相关常量
# ============================================================================

# 文件大小限制（字节）
MAX_CONFIG_FILE_SIZE: Final[int] = 10485760  # 10MB
MAX_BACKUP_FILE_SIZE: Final[int] = 52428800  # 50MB

# 数据库连接池
DEFAULT_POOL_SIZE: Final[int] = 10
MIN_POOL_SIZE: Final[int] = 1
MAX_POOL_SIZE: Final[int] = 100

# ============================================================================
# 网络和连接相关常量
# ============================================================================

# HTTP相关
DEFAULT_HTTP_TIMEOUT: Final[int] = 30
DEFAULT_MAX_CONNECTIONS: Final[int] = 100
DEFAULT_KEEP_ALIVE_TIMEOUT: Final[int] = 300

# API相关
DEFAULT_API_VERSION: Final[str] = "v1"
MAX_API_REQUEST_SIZE: Final[int] = 1048576  # 1MB
API_RATE_LIMIT_PER_SECOND: Final[int] = 10

# ============================================================================
# 日志和调试相关常量
# ============================================================================

# 日志级别
DEFAULT_LOG_LEVEL: Final[str] = "INFO"
LOG_MAX_FILE_SIZE: Final[int] = 104857600  # 100MB
LOG_BACKUP_COUNT: Final[int] = 10

# 调试模式
DEBUG_MODE_ENABLED: Final[bool] = False
TRACE_MODE_ENABLED: Final[bool] = False

# ============================================================================
# 其他通用常量
# ============================================================================

# 分页相关
DEFAULT_PAGE_SIZE: Final[int] = 50
MIN_PAGE_SIZE: Final[int] = 10
MAX_PAGE_SIZE: Final[int] = 1000

# 批量操作
DEFAULT_BATCH_SIZE: Final[int] = 100
MIN_BATCH_SIZE: Final[int] = 1
MAX_BATCH_SIZE: Final[int] = 1000

# 编码格式
DEFAULT_ENCODING: Final[str] = "utf-8"
SUPPORTED_ENCODINGS: Final[list] = ["utf-8", "utf-16", "ascii", "latin-1"]




