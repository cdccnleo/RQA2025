
# ============================================================================
# 日志级别相关常量
# 标准日志级别

from typing import Final
"""
日志管理模块常量定义
Logging Management Module Constants

定义日志管理相关的常量，避免魔法数字
"""

LOG_LEVEL_DEBUG: Final[str] = "DEBUG"
LOG_LEVEL_INFO: Final[str] = "INFO"
LOG_LEVEL_WARNING: Final[str] = "WARNING"
LOG_LEVEL_ERROR: Final[str] = "ERROR"
LOG_LEVEL_CRITICAL: Final[str] = "CRITICAL"

# 日志级别数值映射
LOG_LEVEL_VALUES: Final[dict] = {
    LOG_LEVEL_DEBUG: 10,
    LOG_LEVEL_INFO: 20,
    LOG_LEVEL_WARNING: 30,
    LOG_LEVEL_ERROR: 40,
    LOG_LEVEL_CRITICAL: 50
}

# ============================================================================
# 文件和路径相关常量
# ============================================================================

# 默认日志文件名
DEFAULT_LOG_FILENAME: Final[str] = "app.log"
DEFAULT_ERROR_LOG_FILENAME: Final[str] = "error.log"
DEFAULT_ACCESS_LOG_FILENAME: Final[str] = "access.log"

# 日志目录
DEFAULT_LOG_DIR: Final[str] = "logs"
ARCHIVE_LOG_DIR: Final[str] = "logs/archive"

# 日志文件扩展名
LOG_FILE_EXTENSION: Final[str] = ".log"
COMPRESSED_LOG_EXTENSION: Final[str] = ".log.gz"

# ============================================================================
# 文件大小和轮转相关常量
# ============================================================================

# 文件大小限制（字节）
DEFAULT_MAX_LOG_SIZE: Final[int] = 10485760      # 10MB
DEFAULT_MAX_LOG_SIZE_MB: Final[int] = 10         # 10MB
DEFAULT_MAX_LOG_SIZE_GB: Final[float] = 0.009765625  # ~10MB in GB
MAX_LOG_SIZE: Final[int] = 1073741824            # 1GB
MIN_LOG_SIZE: Final[int] = 1024                  # 1KB

# 文件大小转换常量
BYTES_PER_KB: Final[int] = 1024
BYTES_PER_MB: Final[int] = 1048576
BYTES_PER_GB: Final[int] = 1073741824

# 备份文件数量
DEFAULT_BACKUP_COUNT: Final[int] = 5
DEFAULT_MAX_BACKUP_COUNT: Final[int] = 5
MAX_BACKUP_COUNT: Final[int] = 100
MIN_BACKUP_COUNT: Final[int] = 1

# 轮转时间
DAILY_ROTATION: Final[str] = "MIDNIGHT"
HOURLY_ROTATION_SUFFIX: Final[str] = ".%Y%m%d-%H"
DAILY_ROTATION_SUFFIX: Final[str] = ".%Y%m%d"

# ============================================================================
# 时间和格式相关常量
# ============================================================================

# 时间格式
DEFAULT_TIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
ISO_TIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%S.%fZ"
COMPACT_TIME_FORMAT: Final[str] = "%Y%m%d_%H%M%S"

# 日志格式模板
DEFAULT_LOG_FORMAT: Final[str] = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DETAILED_LOG_FORMAT: Final[str] = "[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] %(message)s"
JSON_LOG_FORMAT: Final[str] = '{"timestamp": "%(asctime)s", "level": "%(levelname)s", "logger": "%(name)s", "message": "%(message)s"}'

# 格式分隔符
LOG_FORMAT_SEPARATOR: Final[str] = " | "
LOG_FIELD_SEPARATOR: Final[str] = ", "

# ============================================================================
# 队列和缓冲相关常量
# ============================================================================

# 队列大小
DEFAULT_QUEUE_SIZE: Final[int] = 1000
MAX_QUEUE_SIZE: Final[int] = 10000
MIN_QUEUE_SIZE: Final[int] = 10

# 缓冲区大小
DEFAULT_BUFFER_SIZE: Final[int] = 8192            # 8KB
MAX_BUFFER_SIZE: Final[int] = 1048576             # 1MB
MIN_BUFFER_SIZE: Final[int] = 1024                # 1KB

# ============================================================================
# 网络和连接相关常量
# ============================================================================

# 默认端口
DEFAULT_LOG_PORT: Final[int] = 8080
DEFAULT_SECURE_LOG_PORT: Final[int] = 8443
DEFAULT_SYSLOG_PORT: Final[int] = 514
DEFAULT_HTTP_LOG_PORT: Final[int] = 8080
DEFAULT_WEBSOCKET_PORT: Final[int] = 8081

# 网络限制
MAX_CONNECTIONS: Final[int] = 1000
MAX_CONNECTIONS_PER_HOST: Final[int] = 100
NETWORK_CONNECTION_TIMEOUT: Final[int] = 30
NETWORK_READ_TIMEOUT: Final[int] = 10
NETWORK_WRITE_TIMEOUT: Final[int] = 10

# 连接超时（秒）
DEFAULT_TIMEOUT: Final[int] = 30
CONNECTION_TIMEOUT: Final[int] = 30
READ_TIMEOUT: Final[int] = 10
WRITE_TIMEOUT: Final[int] = 10
DEFAULT_CONNECTION_TIMEOUT: Final[int] = 30
MAX_CONNECTION_TIMEOUT: Final[int] = 300
MIN_CONNECTION_TIMEOUT: Final[int] = 5

# 重试配置
DEFAULT_MAX_RETRIES: Final[int] = 3
DEFAULT_RETRY_DELAY: Final[float] = 1.0
MAX_RETRY_DELAY: Final[float] = 60.0

# 批处理和刷新
DEFAULT_FLUSH_INTERVAL: Final[int] = 5
MAX_FLUSH_INTERVAL: Final[int] = 300

# ============================================================================
# 监控和性能相关常量
# ============================================================================

# 监控间隔（秒）
DEFAULT_MONITOR_INTERVAL: Final[int] = 60
DEFAULT_MONITORING_INTERVAL: Final[int] = 60
MIN_MONITOR_INTERVAL: Final[int] = 5
MIN_MONITORING_INTERVAL: Final[int] = 5
MAX_MONITOR_INTERVAL: Final[int] = 3600
MAX_MONITORING_INTERVAL: Final[int] = 3600

# 性能阈值
LOG_PROCESSING_WARNING_THRESHOLD: Final[int] = 1000    # 1秒处理1000条日志
LOG_PROCESSING_ERROR_THRESHOLD: Final[int] = 100       # 1秒处理100条日志

# 错误和警告阈值
DEFAULT_ERROR_THRESHOLD: Final[float] = 0.05           # 5%
DEFAULT_WARNING_THRESHOLD: Final[float] = 0.10         # 10%
DEFAULT_CRITICAL_THRESHOLD: Final[float] = 0.20        # 20%

# 内存使用阈值
MEMORY_WARNING_THRESHOLD: Final[float] = 0.8           # 80%
MEMORY_CRITICAL_THRESHOLD: Final[float] = 0.95         # 95%

# 磁盘使用阈值
DISK_USAGE_WARNING_THRESHOLD: Final[float] = 0.85      # 85%
DISK_USAGE_CRITICAL_THRESHOLD: Final[float] = 0.95     # 95%

# 历史记录限制
MAX_METRICS_HISTORY: Final[int] = 10000
MAX_ALERT_HISTORY: Final[int] = 1000
MAX_PERFORMANCE_SAMPLES: Final[int] = 1000

# ============================================================================
# 过滤和安全相关常量
# ============================================================================

# 敏感信息模式
SENSITIVE_PATTERNS: Final[list] = [
    r'\b\d{4}[- ]?\d{4}[- ]?\d{4}[- ]?\d{4}\b',      # 信用卡号
    r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',                # SSN
    r'password[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]',    # 密码
    r'token[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]',       # Token
    r'key[\'"]?\s*[:=]\s*[\'"][^\'"]*[\'"]',         # 密钥
]

# 日志过滤级别
FILTER_LEVEL_LOW: Final[str] = "low"
FILTER_LEVEL_MEDIUM: Final[str] = "medium"
FILTER_LEVEL_HIGH: Final[str] = "high"
FILTER_LEVEL_CRITICAL: Final[str] = "critical"

# 安全级别
MIN_SECURITY_LEVEL: Final[int] = 1
MAX_SECURITY_LEVEL: Final[int] = 5

# 加密算法
DEFAULT_ENCRYPTION_ALGORITHM: Final[str] = "AES256"

# 密钥大小
DEFAULT_KEY_SIZE: Final[int] = 256

# ============================================================================
# 序列化和压缩相关常量
# ============================================================================

# 压缩算法
COMPRESSION_NONE: Final[str] = "none"
COMPRESSION_GZIP: Final[str] = "gzip"
COMPRESSION_ZIP: Final[str] = "zip"
COMPRESSION_LZ4: Final[str] = "lz4"

# 序列化格式
SERIALIZATION_TEXT: Final[str] = "text"
SERIALIZATION_JSON: Final[str] = "json"
SERIALIZATION_XML: Final[str] = "xml"
SERIALIZATION_BINARY: Final[str] = "binary"

# ============================================================================
# 告警和通知相关常量
# ============================================================================

# 告警级别
ALERT_LEVEL_INFO: Final[str] = "info"
ALERT_LEVEL_WARNING: Final[str] = "warning"
ALERT_LEVEL_ERROR: Final[str] = "error"
ALERT_LEVEL_CRITICAL: Final[str] = "critical"

# 告警阈值
ERROR_RATE_WARNING_THRESHOLD: Final[float] = 0.05      # 5%
ERROR_RATE_CRITICAL_THRESHOLD: Final[float] = 0.10     # 10%

DISK_USAGE_WARNING_THRESHOLD: Final[float] = 0.85      # 85%
DISK_USAGE_CRITICAL_THRESHOLD: Final[float] = 0.95     # 95%

# ============================================================================
# 缓存和性能优化相关常量
# ============================================================================

# 缓存配置
LOG_CACHE_SIZE: Final[int] = 1000
LOG_CACHE_TTL: Final[int] = 300                         # 5分钟

# 批量处理
DEFAULT_BATCH_SIZE: Final[int] = 100
MAX_BATCH_SIZE: Final[int] = 1000
MIN_BATCH_SIZE: Final[int] = 10

# ============================================================================
# 其他通用常量
# ============================================================================

# 编码格式
DEFAULT_ENCODING: Final[str] = "utf-8"
FALLBACK_ENCODING: Final[str] = "latin-1"

# 分页配置
DEFAULT_PAGE_SIZE: Final[int] = 100
MAX_PAGE_SIZE: Final[int] = 1000

# 会话和上下文
DEFAULT_SESSION_TIMEOUT: Final[int] = 3600              # 1小时
MAX_SESSION_TIMEOUT: Final[int] = 86400                 # 24小时

# ============================================================================
# 标准和协议相关常量
# ============================================================================

# Syslog设施
SYSLOG_FACILITY_USER: Final[int] = 1
SYSLOG_FACILITY_MAIL: Final[int] = 2
SYSLOG_FACILITY_DAEMON: Final[int] = 3
SYSLOG_FACILITY_AUTH: Final[int] = 4

# HTTP状态码
HTTP_OK: Final[int] = 200
HTTP_BAD_REQUEST: Final[int] = 400
HTTP_UNAUTHORIZED: Final[int] = 401
HTTP_FORBIDDEN: Final[int] = 403
HTTP_NOT_FOUND: Final[int] = 404
HTTP_INTERNAL_ERROR: Final[int] = 500

# ============================================================================
# 版本和兼容性相关常量
# ============================================================================

# API版本
DEFAULT_API_VERSION: Final[str] = "v1"
SUPPORTED_API_VERSIONS: Final[list] = ["v1", "v2"]

# 格式版本
LOG_FORMAT_VERSION: Final[str] = "1.0"
MIN_SUPPORTED_FORMAT_VERSION: Final[str] = "0.9"

# ============================================================================
# 数据库相关常量
# ============================================================================

# 数据库连接
DEFAULT_MAX_DB_CONNECTIONS: Final[int] = 10
DEFAULT_DB_TIMEOUT: Final[int] = 30
DEFAULT_DB_POOL_SIZE: Final[int] = 5

# 数据库重试
DEFAULT_DB_RETRY_INTERVAL: Final[float] = 1.0
MAX_DB_RETRY_ATTEMPTS: Final[int] = 3

# 数据库路径
DEFAULT_DB_PATH: Final[str] = "logs/database.db"

# ---------------------------------------------------------------------------
# 兼容性导出：自动收集并注入常量到内建命名空间
# ---------------------------------------------------------------------------

__all__ = [name for name in globals() if name.isupper()]

try:  # pragma: no cover - 兼容旧测试环境
    import builtins as _builtins
    import logging as _logging

    if not hasattr(_builtins, "logging"):
        setattr(_builtins, "logging", _logging)

    for _name in __all__:
        if not hasattr(_builtins, _name):
            setattr(_builtins, _name, globals()[_name])
except Exception:
    pass