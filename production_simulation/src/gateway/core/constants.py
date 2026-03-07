"""
网关层常量定义
Gateway Layer Constants

定义API网关相关的常量，避免魔法数字
"""

# HTTP状态码
HTTP_OK = 200
HTTP_CREATED = 201
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_FORBIDDEN = 403
HTTP_NOT_FOUND = 404
HTTP_METHOD_NOT_ALLOWED = 405
HTTP_TOO_MANY_REQUESTS = 429
HTTP_INTERNAL_SERVER_ERROR = 500

# 请求处理参数
DEFAULT_REQUEST_TIMEOUT = 30      # 默认请求超时时间(秒)
MAX_REQUEST_SIZE_BYTES = 10485760  # 最大请求大小(10MB)
MAX_REQUESTS_PER_MINUTE = 1000   # 每分钟最大请求数
MAX_CONCURRENT_REQUESTS = 100    # 最大并发请求数

# 路由参数
DEFAULT_ROUTE_TIMEOUT = 10        # 默认路由超时时间(秒)
MAX_ROUTE_DEPTH = 10              # 最大路由深度
ROUTE_CACHE_SIZE = 1000           # 路由缓存大小

# 负载均衡参数
DEFAULT_LOAD_BALANCER_TIMEOUT = 5  # 默认负载均衡器超时(秒)
HEALTH_CHECK_INTERVAL = 30        # 健康检查间隔(秒)
MAX_FAILED_REQUESTS = 3           # 最大失败请求数

# 安全参数
DEFAULT_AUTH_TIMEOUT = 5          # 默认认证超时时间(秒)
TOKEN_EXPIRY_SECONDS = 3600       # 令牌过期时间(秒)
RATE_LIMIT_WINDOW_SECONDS = 60    # 速率限制窗口(秒)

# 缓存参数
RESPONSE_CACHE_SIZE = 10000       # 响应缓存大小
CACHE_TTL_SECONDS = 300           # 缓存过期时间(秒)
CACHE_CLEANUP_INTERVAL = 60       # 缓存清理间隔(秒)

# 日志参数
LOG_RETENTION_DAYS = 30           # 日志保留天数
MAX_LOG_FILE_SIZE_MB = 100        # 最大日志文件大小(MB)
LOG_ROTATION_COUNT = 10           # 日志轮转数量

# 监控参数
METRICS_UPDATE_INTERVAL = 10      # 指标更新间隔(秒)
ALERT_THRESHOLD_HIGH = 0.9        # 高风险告警阈值
ALERT_THRESHOLD_MEDIUM = 0.7      # 中风险告警阈值

# WebSocket参数
WS_CONNECTION_TIMEOUT = 30        # WebSocket连接超时(秒)
WS_MESSAGE_SIZE_LIMIT = 65536     # WebSocket消息大小限制(64KB)
WS_HEARTBEAT_INTERVAL = 30        # WebSocket心跳间隔(秒)

# API版本参数
DEFAULT_API_VERSION = "v1"        # 默认API版本
API_VERSION_HEADER = "X-API-Version"  # API版本请求头
SUPPORTED_API_VERSIONS = ["v1", "v2"]  # 支持的API版本列表

# 文档参数
API_DOCS_PATH = "/docs"           # API文档路径
API_SPEC_PATH = "/openapi.json"   # API规范路径

# 代理参数
PROXY_TIMEOUT = 30                # 代理超时时间(秒)
PROXY_BUFFER_SIZE = 8192          # 代理缓冲区大小
PROXY_MAX_CONNECTIONS = 1000      # 代理最大连接数

# 静态文件参数
STATIC_FILE_CACHE_TTL = 86400     # 静态文件缓存TTL(1天)
MAX_STATIC_FILE_SIZE_MB = 50      # 最大静态文件大小(MB)

# 数据库连接参数
DB_CONNECTION_TIMEOUT = 10        # 数据库连接超时(秒)
DB_CONNECTION_POOL_SIZE = 20      # 数据库连接池大小
DB_QUERY_TIMEOUT = 30             # 数据库查询超时(秒)

# 外部服务参数
EXTERNAL_SERVICE_TIMEOUT = 15     # 外部服务超时时间(秒)
SERVICE_DISCOVERY_INTERVAL = 30   # 服务发现间隔(秒)
CIRCUIT_BREAKER_THRESHOLD = 5     # 熔断器阈值

# 性能参数
RESPONSE_TIME_TARGET_MS = 200     # 响应时间目标(毫秒)
THROUGHPUT_TARGET_RPS = 1000      # 吞吐量目标(RPS)
ERROR_RATE_TARGET_PCT = 1.0       # 错误率目标(%)

# 资源限制
MEMORY_USAGE_THRESHOLD_PCT = 80   # 内存使用率阈值(%)
CPU_USAGE_THRESHOLD_PCT = 70      # CPU使用率阈值(%)
DISK_USAGE_THRESHOLD_PCT = 85     # 磁盘使用率阈值(%)
