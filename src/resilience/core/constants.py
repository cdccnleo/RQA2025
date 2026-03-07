"""
弹性层常量定义
Resilience Layer Constants

定义系统弹性相关的常量，避免魔法数字
"""

# 熔断器参数
CIRCUIT_BREAKER_FAILURE_THRESHOLD = 5    # 熔断器失败阈值
CIRCUIT_BREAKER_RECOVERY_TIMEOUT = 60    # 熔断器恢复超时时间(秒)
CIRCUIT_BREAKER_SUCCESS_THRESHOLD = 3    # 熔断器成功阈值

# 重试参数
DEFAULT_MAX_RETRIES = 3                  # 默认最大重试次数
DEFAULT_RETRY_DELAY = 1                  # 默认重试延迟(秒)
EXPONENTIAL_BACKOFF_MULTIPLIER = 2       # 指数退避乘数
MAX_RETRY_DELAY = 60                     # 最大重试延迟(秒)

# 降级参数
DEGRADATION_THRESHOLD_CPU = 80           # CPU降级阈值(%)
DEGRADATION_THRESHOLD_MEMORY = 85        # 内存降级阈值(%)
DEGRADATION_THRESHOLD_DISK = 90          # 磁盘降级阈值(%)

# 超时参数
DEFAULT_OPERATION_TIMEOUT = 30           # 默认操作超时时间(秒)
HEALTH_CHECK_TIMEOUT = 5                 # 健康检查超时时间(秒)
CONNECTION_TIMEOUT = 10                  # 连接超时时间(秒)

# 负载均衡参数
LOAD_BALANCER_HEALTH_CHECK_INTERVAL = 30  # 负载均衡器健康检查间隔(秒)
LOAD_BALANCER_MAX_FAILURES = 3           # 负载均衡器最大失败次数
LOAD_BALANCER_RECOVERY_TIME = 60         # 负载均衡器恢复时间(秒)

# 缓存参数
CACHE_DEGRADATION_THRESHOLD = 0.8        # 缓存降级阈值
CACHE_FAILURE_THRESHOLD = 0.9            # 缓存失败阈值
CACHE_RECOVERY_TIMEOUT = 300             # 缓存恢复超时时间(秒)

# 监控参数
RESILIENCE_METRICS_INTERVAL = 10         # 弹性指标收集间隔(秒)
ALERT_THRESHOLD_HIGH = 0.9               # 高风险告警阈值
ALERT_THRESHOLD_MEDIUM = 0.7             # 中风险告警阈值

# 优雅降级参数
GRACEFUL_DEGRADATION_LEVELS = 3          # 优雅降级级别数
DEGRADATION_RESPONSE_TIME_MS = 2000      # 降级响应时间(毫秒)
DEGRADATION_SUCCESS_RATE = 0.8           # 降级成功率阈值

# 故障恢复参数
FAILURE_DETECTION_WINDOW = 300           # 故障检测窗口(秒)
AUTO_RECOVERY_ATTEMPTS = 3               # 自动恢复尝试次数
RECOVERY_COOLDOWN_PERIOD = 600           # 恢复冷却期(秒)

# 资源池参数
POOL_MAX_SIZE = 100                      # 池最大大小
POOL_MIN_SIZE = 10                       # 池最小大小
POOL_IDLE_TIMEOUT = 300                  # 池空闲超时时间(秒)

# 服务发现参数
SERVICE_DISCOVERY_TIMEOUT = 5            # 服务发现超时时间(秒)
SERVICE_HEALTH_CHECK_INTERVAL = 30       # 服务健康检查间隔(秒)
SERVICE_FAILURE_THRESHOLD = 3            # 服务失败阈值

# 容错参数
FAULT_TOLERANCE_LEVEL = 2                # 容错级别
MAX_CONSECUTIVE_FAILURES = 5             # 最大连续失败次数
FAULT_RECOVERY_TIME = 120                # 故障恢复时间(秒)

# 备份参数
BACKUP_OPERATION_TIMEOUT = 300           # 备份操作超时时间(秒)
BACKUP_VERIFICATION_TIMEOUT = 60         # 备份验证超时时间(秒)
BACKUP_RETENTION_PERIOD = 86400          # 备份保留期(秒)

# 性能参数
PERFORMANCE_DEGRADATION_THRESHOLD = 0.7  # 性能降级阈值
THROUGHPUT_DEGRADATION_THRESHOLD = 0.6   # 吞吐量降级阈值
LATENCY_DEGRADATION_THRESHOLD_MS = 1000  # 延迟降级阈值(毫秒)

# 扩展参数
AUTO_SCALING_COOLDOWN = 300              # 自动扩展冷却时间(秒)
SCALING_DECISION_WINDOW = 600            # 扩展决策窗口(秒)
MAX_SCALE_FACTOR = 3.0                   # 最大扩展因子
