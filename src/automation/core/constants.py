"""
自动化层常量定义
Automation Layer Constants

定义自动化相关的常量，避免魔法数字
"""

# 任务调度参数
DEFAULT_MAX_WORKERS = 10         # 默认最大工作进程数
DEFAULT_QUEUE_SIZE = 1000        # 默认队列大小
DEFAULT_TASK_TIMEOUT = 3600      # 默认任务超时时间(秒)

# 重试和错误处理
MAX_RETRY_ATTEMPTS = 3           # 最大重试次数
RETRY_DELAY_SECONDS = 5          # 重试延迟(秒)
CIRCUIT_BREAKER_THRESHOLD = 5    # 熔断器阈值
CIRCUIT_BREAKER_TIMEOUT = 60     # 熔断器超时时间(秒)

# 监控和告警
MONITOR_UPDATE_INTERVAL = 30     # 监控更新间隔(秒)
ALERT_THRESHOLD_HIGH = 0.9       # 高风险告警阈值
ALERT_THRESHOLD_MEDIUM = 0.7     # 中风险告警阈值
HEALTH_CHECK_INTERVAL = 60       # 健康检查间隔(秒)

# 工作流参数
MAX_WORKFLOW_DEPTH = 10          # 最大工作流深度
DEFAULT_WORKFLOW_TIMEOUT = 7200  # 默认工作流超时时间(秒)
WORKFLOW_CHECK_INTERVAL = 10     # 工作流检查间隔(秒)

# 规则引擎参数
MAX_RULES_PER_ENGINE = 1000      # 每个引擎最大规则数
RULE_EVALUATION_TIMEOUT = 30     # 规则评估超时时间(秒)
RULE_CACHE_SIZE = 500           # 规则缓存大小

# 部署参数
DEPLOYMENT_TIMEOUT = 1800        # 部署超时时间(秒)
ROLLBACK_TIMEOUT = 600          # 回滚超时时间(秒)
HEALTH_CHECK_RETRIES = 5        # 健康检查重试次数

# 备份和恢复
BACKUP_RETENTION_DAYS = 30      # 备份保留天数
BACKUP_TIMEOUT = 3600           # 备份超时时间(秒)
RECOVERY_TIMEOUT = 1800         # 恢复超时时间(秒)

# 集成参数
API_TIMEOUT = 30                # API超时时间(秒)
DB_CONNECTION_TIMEOUT = 10      # 数据库连接超时时间(秒)
CLOUD_API_TIMEOUT = 60          # 云API超时时间(秒)

# 扩展和缩放
AUTO_SCALE_INTERVAL = 300       # 自动缩放检查间隔(秒)
SCALE_UP_THRESHOLD = 0.8        # 扩容阈值
SCALE_DOWN_THRESHOLD = 0.3      # 缩容阈值
MAX_SCALE_FACTOR = 3.0          # 最大缩放因子

# 维护参数
MAINTENANCE_WINDOW_HOURS = 4    # 维护窗口小时数
CLEANUP_INTERVAL_DAYS = 7       # 清理间隔天数
LOG_RETENTION_DAYS = 90         # 日志保留天数

# 性能参数
PERFORMANCE_SAMPLE_SIZE = 100   # 性能采样大小
THROUGHPUT_TARGET = 1000        # 吞吐量目标
LATENCY_TARGET_MS = 100         # 延迟目标(毫秒)

# 并发控制
MAX_CONCURRENT_TASKS = 50       # 最大并发任务数
TASK_PRIORITY_LEVELS = 5        # 任务优先级级别
QUEUE_PROCESSING_INTERVAL = 1   # 队列处理间隔(秒)

# 数据处理
BATCH_SIZE_DEFAULT = 1000       # 默认批处理大小
STREAM_BUFFER_SIZE = 10000      # 流缓冲区大小
DATA_VALIDATION_TIMEOUT = 60    # 数据验证超时时间(秒)
