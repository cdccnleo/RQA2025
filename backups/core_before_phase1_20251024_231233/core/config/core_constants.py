"""
核心服务层常量定义
Core Services Layer Constants

定义核心服务相关的常量，避免魔法数字
"""

# 服务容器参数
DEFAULT_SERVICE_TTL = 3600          # 默认服务生存时间(秒)
SERVICE_DISCOVERY_TIMEOUT = 30      # 服务发现超时时间(秒)
SERVICE_HEALTH_CHECK_INTERVAL = 60  # 服务健康检查间隔(秒)

# 事件总线参数
EVENT_BUS_BUFFER_SIZE = 10000       # 事件总线缓冲区大小
EVENT_PROCESSING_TIMEOUT = 30       # 事件处理超时时间(秒)
MAX_EVENT_SUBSCRIBERS = 100         # 最大事件订阅者数

# 业务流程参数
WORKFLOW_EXECUTION_TIMEOUT = 1800   # 工作流执行超时时间(秒)
MAX_WORKFLOW_DEPTH = 10             # 最大工作流深度
WORKFLOW_CHECK_INTERVAL = 10        # 工作流检查间隔(秒)

# 集成参数
INTEGRATION_TIMEOUT = 60            # 集成超时时间(秒)
ADAPTER_CONNECTION_POOL_SIZE = 20   # 适配器连接池大小
BRIDGE_BUFFER_SIZE = 5000           # 桥接缓冲区大小

# 安全参数
AUTH_TOKEN_EXPIRY = 3600            # 认证令牌过期时间(秒)
ENCRYPTION_KEY_SIZE = 256           # 加密密钥大小
SECURITY_AUDIT_RETENTION = 90       # 安全审计保留天数

# 性能参数
PERFORMANCE_MONITOR_INTERVAL = 30   # 性能监控间隔(秒)
RESPONSE_TIME_TARGET_MS = 200       # 响应时间目标(毫秒)
CONCURRENCY_TARGET = 1000           # 并发目标

# 缓存参数
CORE_CACHE_SIZE = 10000             # 核心缓存大小
CACHE_EVICTION_INTERVAL = 300       # 缓存驱逐间隔(秒)

# 监控参数
METRICS_COLLECTION_INTERVAL = 15    # 指标收集间隔(秒)
ALERT_THRESHOLD_HIGH = 0.9          # 高风险告警阈值
ALERT_THRESHOLD_MEDIUM = 0.7        # 中风险告警阈值

# 异步处理参数
ASYNC_TASK_QUEUE_SIZE = 1000        # 异步任务队列大小
ASYNC_WORKER_COUNT = 4              # 异步工作进程数
ASYNC_TASK_TIMEOUT = 600            # 异步任务超时时间(秒)

# 资源管理参数
RESOURCE_POOL_SIZE = 50             # 资源池大小
RESOURCE_CHECK_INTERVAL = 60        # 资源检查间隔(秒)
RESOURCE_CLEANUP_INTERVAL = 300     # 资源清理间隔(秒)

# 通信参数
COMMUNICATION_TIMEOUT = 30          # 通信超时时间(秒)
MESSAGE_QUEUE_SIZE = 5000           # 消息队列大小
MAX_MESSAGE_SIZE_KB = 1024          # 最大消息大小(KB)

# 配置参数
CONFIG_UPDATE_INTERVAL = 300        # 配置更新间隔(秒)
CONFIG_VALIDATION_TIMEOUT = 10      # 配置验证超时时间(秒)

# 扩展参数
AUTO_SCALE_CHECK_INTERVAL = 60      # 自动扩展检查间隔(秒)
SCALE_UP_THRESHOLD = 0.8            # 扩容阈值
SCALE_DOWN_THRESHOLD = 0.3          # 缩容阈值
