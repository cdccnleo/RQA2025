
# ============================================================================
# 监控时间和间隔相关常量
# 监控间隔（秒）

from typing import Final
"""
监控管理模块常量定义
Monitoring Management Module Constants

定义监控管理相关的常量，避免魔法数字
"""

DEFAULT_MONITOR_INTERVAL: Final[int] = 30
FAST_MONITOR_INTERVAL: Final[int] = 5
SLOW_MONITOR_INTERVAL: Final[int] = 300

# 监控周期（秒）
SHORT_MONITORING_PERIOD: Final[int] = 300      # 5分钟
MEDIUM_MONITORING_PERIOD: Final[int] = 1800    # 30分钟
LONG_MONITORING_PERIOD: Final[int] = 3600      # 1小时

# 数据保留时间（秒）
METRICS_RETENTION_PERIOD: Final[int] = 604800  # 7天
ALERT_HISTORY_RETENTION: Final[int] = 2592000  # 30天

# ============================================================================
# 阈值和告警相关常量
# ============================================================================

# CPU使用率阈值（百分比）
CPU_WARNING_THRESHOLD: Final[float] = 70.0
CPU_CRITICAL_THRESHOLD: Final[float] = 90.0
CPU_RECOVERY_THRESHOLD: Final[float] = 60.0

# 内存使用率阈值（百分比）
MEMORY_WARNING_THRESHOLD: Final[float] = 75.0
MEMORY_CRITICAL_THRESHOLD: Final[float] = 90.0
MEMORY_RECOVERY_THRESHOLD: Final[float] = 70.0

# 磁盘使用率阈值（百分比）
DISK_WARNING_THRESHOLD: Final[float] = 80.0
DISK_CRITICAL_THRESHOLD: Final[float] = 95.0
DISK_RECOVERY_THRESHOLD: Final[float] = 75.0

# 网络延迟阈值（毫秒）
NETWORK_LATENCY_WARNING: Final[int] = 100
NETWORK_LATENCY_CRITICAL: Final[int] = 500
NETWORK_LATENCY_RECOVERY: Final[int] = 50

# ============================================================================
# 性能指标阈值
# ============================================================================

# 响应时间阈值（毫秒）
RESPONSE_TIME_WARNING: Final[int] = 1000
RESPONSE_TIME_CRITICAL: Final[int] = 5000
RESPONSE_TIME_RECOVERY: Final[int] = 500

# 吞吐量阈值（每秒请求数）
THROUGHPUT_WARNING: Final[int] = 100
THROUGHPUT_CRITICAL: Final[int] = 10
THROUGHPUT_RECOVERY: Final[int] = 200

# 错误率阈值（百分比）
ERROR_RATE_WARNING: Final[float] = 1.0
ERROR_RATE_CRITICAL: Final[float] = 5.0
ERROR_RATE_RECOVERY: Final[float] = 0.1

# ============================================================================
# 组件监控阈值
# ============================================================================

# 组件创建时间阈值（毫秒）
COMPONENT_CREATION_TIME_WARNING: Final[int] = 5000
COMPONENT_CREATION_TIME_CRITICAL: Final[int] = 30000

# 组件并发使用阈值
COMPONENT_CONCURRENT_WARNING: Final[int] = 50
COMPONENT_CONCURRENT_CRITICAL: Final[int] = 100

# 组件内存使用阈值（MB）
COMPONENT_MEMORY_WARNING: Final[int] = 500
COMPONENT_MEMORY_CRITICAL: Final[int] = 1000

# ============================================================================
# 告警配置常量
# ============================================================================

# 告警级别
ALERT_LEVEL_INFO: Final[str] = "INFO"
ALERT_LEVEL_WARNING: Final[str] = "WARNING"
ALERT_LEVEL_ERROR: Final[str] = "ERROR"
ALERT_LEVEL_CRITICAL: Final[str] = "CRITICAL"

# 告警状态
ALERT_STATUS_ACTIVE: Final[str] = "ACTIVE"
ALERT_STATUS_RESOLVED: Final[str] = "RESOLVED"
ALERT_STATUS_ACKED: Final[str] = "ACKNOWLEDGED"

# 告警优先级
ALERT_PRIORITY_LOW: Final[int] = 1
ALERT_PRIORITY_MEDIUM: Final[int] = 2
ALERT_PRIORITY_HIGH: Final[int] = 3
ALERT_PRIORITY_CRITICAL: Final[int] = 4

# ============================================================================
# 通知和通信常量
# ============================================================================

# 通知重试配置
NOTIFICATION_MAX_RETRIES: Final[int] = 3
NOTIFICATION_RETRY_DELAY: Final[int] = 5

# 通知通道
NOTIFICATION_EMAIL: Final[str] = "email"
NOTIFICATION_SMS: Final[str] = "sms"
NOTIFICATION_WEBHOOK: Final[str] = "webhook"
NOTIFICATION_SLACK: Final[str] = "slack"

# ============================================================================
# 数据收集和存储常量
# ============================================================================

# 批量处理大小
METRICS_BATCH_SIZE: Final[int] = 100
ALERT_BATCH_SIZE: Final[int] = 50

# 缓存大小
METRICS_CACHE_SIZE: Final[int] = 10000
ALERT_CACHE_SIZE: Final[int] = 1000

# ============================================================================
# 系统健康检查常量
# ============================================================================

# 健康检查超时（秒）
HEALTH_CHECK_TIMEOUT: Final[int] = 10
HEALTH_CHECK_INTERVAL: Final[int] = 60

# 健康状态
HEALTH_STATUS_HEALTHY: Final[str] = "HEALTHY"
HEALTH_STATUS_DEGRADED: Final[str] = "DEGRADED"
HEALTH_STATUS_UNHEALTHY: Final[str] = "UNHEALTHY"
HEALTH_STATUS_UNKNOWN: Final[str] = "UNKNOWN"

# ============================================================================
# 异常监控常量
# ============================================================================

# 异常计数阈值
EXCEPTION_COUNT_WARNING: Final[int] = 10
EXCEPTION_COUNT_CRITICAL: Final[int] = 50

# 异常类型统计
EXCEPTION_TYPE_LIMIT: Final[int] = 20

# ============================================================================
# 连续监控系统常量
# ============================================================================

# 分析窗口大小
ANALYSIS_WINDOW_SIZE: Final[int] = 100
TREND_ANALYSIS_WINDOW: Final[int] = 50

# 优化建议阈值
OPTIMIZATION_THRESHOLD_HIGH: Final[float] = 0.8
OPTIMIZATION_THRESHOLD_MEDIUM: Final[float] = 0.6
OPTIMIZATION_THRESHOLD_LOW: Final[float] = 0.4

# ============================================================================
# 灾难恢复常量
# ============================================================================

# 灾难检测阈值
DISASTER_DETECTION_THRESHOLD: Final[float] = 0.9

# 恢复超时（秒）
RECOVERY_TIMEOUT: Final[int] = 300

# 备份频率（秒）
BACKUP_FREQUENCY: Final[int] = 3600

# ============================================================================
# 存储监控常量
# ============================================================================

# 存储使用率阈值
STORAGE_WARNING_RATIO: Final[float] = 0.8
STORAGE_CRITICAL_RATIO: Final[float] = 0.95

# 文件系统检查间隔（秒）
FILESYSTEM_CHECK_INTERVAL: Final[int] = 300

# ============================================================================
# 生产环境监控常量
# ============================================================================

# 生产环境指标收集间隔（秒）
PRODUCTION_METRICS_INTERVAL: Final[int] = 15

# SLA阈值
SLA_AVAILABILITY_TARGET: Final[float] = 99.9  # 99.9% 可用性
SLA_RESPONSE_TIME_TARGET: Final[int] = 200   # 200ms 响应时间

# ============================================================================
# 日志池监控常量
# ============================================================================

# 日志池大小阈值
LOG_POOL_SIZE_WARNING: Final[int] = 1000
LOG_POOL_SIZE_CRITICAL: Final[int] = 5000

# 日志处理速度阈值（条/秒）
LOG_PROCESSING_RATE_WARNING: Final[int] = 100
LOG_PROCESSING_RATE_CRITICAL: Final[int] = 10

# ============================================================================
# 自适应配置常量
# ============================================================================

# 适应策略调整因子
ADAPTATION_FACTOR_CONSERVATIVE: Final[float] = 1.1   # 10% 调整
ADAPTATION_FACTOR_AGGRESSIVE: Final[float] = 1.5     # 50% 调整
ADAPTATION_FACTOR_BALANCED: Final[float] = 1.25      # 25% 调整

# 适应冷却时间（分钟）
ADAPTATION_COOLDOWN_DEFAULT: Final[int] = 5
ADAPTATION_COOLDOWN_HIGH: Final[int] = 1
ADAPTATION_COOLDOWN_LOW: Final[int] = 15

# 性能基线数据点限制
BASELINE_DATA_MAX_POINTS: Final[int] = 100

# 适应历史清理时间（天）
ADAPTATION_HISTORY_RETENTION_DAYS: Final[int] = 7

# ============================================================================
# 通用常量
# ============================================================================

# 默认编码
DEFAULT_ENCODING: Final[str] = "utf-8"

# 时间格式
DEFAULT_TIME_FORMAT: Final[str] = "%Y-%m-%d %H:%M:%S"
ISO_TIME_FORMAT: Final[str] = "%Y-%m-%dT%H:%M:%SZ"

# 分页大小
DEFAULT_PAGE_SIZE: Final[int] = 50
MAX_PAGE_SIZE: Final[int] = 1000

# 最大并发数
MAX_CONCURRENT_CHECKS: Final[int] = 10

# ============================================================================
# 监控类型常量
# ============================================================================

# 监控类型
MONITOR_TYPE_SYSTEM: Final[str] = "system"
MONITOR_TYPE_APPLICATION: Final[str] = "application"
MONITOR_TYPE_COMPONENT: Final[str] = "component"
MONITOR_TYPE_STORAGE: Final[str] = "storage"
MONITOR_TYPE_NETWORK: Final[str] = "network"
MONITOR_TYPE_LOG: Final[str] = "log"
MONITOR_TYPE_EXCEPTION: Final[str] = "exception"
MONITOR_TYPE_DISASTER: Final[str] = "disaster"

# ============================================================================
# 度量单位常量
# ============================================================================

# 时间单位
UNIT_MILLISECONDS: Final[str] = "ms"
UNIT_SECONDS: Final[str] = "s"
UNIT_MINUTES: Final[str] = "min"
UNIT_HOURS: Final[str] = "h"

# 数据大小单位
UNIT_BYTES: Final[str] = "B"
UNIT_KILOBYTES: Final[str] = "KB"
UNIT_MEGABYTES: Final[str] = "MB"
UNIT_GIGABYTES: Final[str] = "GB"

# 百分比单位
UNIT_PERCENT: Final[str] = "%"

# 频率单位
UNIT_PER_SECOND: Final[str] = "/s"
UNIT_PER_MINUTE: Final[str] = "/min"
UNIT_PER_HOUR: Final[str] = "/h"
