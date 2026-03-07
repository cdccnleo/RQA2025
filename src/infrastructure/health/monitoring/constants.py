"""健康监控常量定义"""

import logging
from typing import Dict, Any

logger = logging.getLogger(__name__)

# 时间转换常量
SECONDS_TO_HOURS = 3600  # 3600秒转小时

# 质量评分常量
QUALITY_SCORE_PENALTY_PER_ERROR = 0.1  # 质量评分降低因子

# GPU相关常量
GPU_PERCENTAGE_MULTIPLIER = 100.0  # GPU百分比转换因子

# 历史数据存储常量
DEFAULT_HISTORY_SIZE = 1000
MAX_HISTORY_SIZE = 10000
MIN_HISTORY_SIZE = 100
DEFAULT_HISTORY_HOURS = 24
HISTORY_HOURS_OPTIONS = [1, 6, 12, 24, 72, 168]  # 1小时, 6小时, 12小时, 1天, 3天, 1周

# 性能计数器常量
COUNTER_DEFAULT_VALUE = 0
COUNTER_API_CALLS = "api_calls"
COUNTER_CACHE_HITS = "cache_hits"
COUNTER_CACHE_MISSES = "cache_misses"
COUNTER_DB_CONNECTIONS = "db_connections"
COUNTER_ERRORS = "errors"
COUNTER_WARNINGS = "warnings"

# 线程超时常量
THREAD_JOIN_TIMEOUT_DEFAULT = 2.0  # 默认线程join超时时间（秒）

# 收集间隔常量
COLLECTION_INTERVAL_DEFAULT = 1.0  # 默认收集间隔时间（秒）
COLLECTION_INTERVAL_MIN = 0.1  # 最小收集间隔
COLLECTION_INTERVAL_MAX = 60.0  # 最大收集间隔
COLLECTION_INTERVAL_FAST = 0.5  # 快速收集间隔
COLLECTION_INTERVAL_SLOW = 5.0  # 慢速收集间隔

# 历史数据查询常量
DEFAULT_HISTORY_QUERY_HOURS = 24
AVERAGE_CALCULATION_HOURS = 1  # 计算平均值的历史小时数
DEFAULT_HISTORY_LIMIT = 100  # 默认历史数据查询限制

# 健康检查阈值常量
RESPONSE_TIME_THRESHOLD_MS = 100  # 响应时间阈值（毫秒）
HISTORY_CAPACITY_WARNING_RATIO = 0.9  # 历史数据容量警告比例
TREND_CALCULATION_PERIODS = 5  # 趋势计算周期数
TREND_CHANGE_THRESHOLD_RATIO = 0.05  # 趋势变化阈值比例（5%）

# 数据验证范围常量
CPU_USAGE_MIN = 0.0
CPU_USAGE_MAX = 100.0
MEMORY_USAGE_MIN = 0.0
MEMORY_USAGE_MAX = 100.0
DISK_USAGE_MIN = 0.0
DISK_USAGE_MAX = 100.0

# 健康状态评估阈值
CPU_HEALTHY_THRESHOLD = 90.0
MEMORY_HEALTHY_THRESHOLD = 85.0
DISK_HEALTHY_THRESHOLD = 95.0

# 性能阈值常量
CPU_THRESHOLD_WARNING = 80.0
CPU_THRESHOLD_CRITICAL = 95.0
MEMORY_THRESHOLD_WARNING = 85.0
MEMORY_THRESHOLD_CRITICAL = 95.0
DISK_THRESHOLD_WARNING = 85.0
DISK_THRESHOLD_CRITICAL = 95.0
NETWORK_THRESHOLD_WARNING = 80.0
NETWORK_THRESHOLD_CRITICAL = 95.0

# 数据质量常量
QUALITY_SCORE_MIN = 0.0
QUALITY_SCORE_MAX = 1.0
QUALITY_SCORE_DEFAULT = 1.0
QUALITY_SCORE_WARNING = 0.7
QUALITY_SCORE_CRITICAL = 0.5

# 并发控制常量
DEFAULT_THREAD_POOL_SIZE = 4
MAX_THREAD_POOL_SIZE = 16
THREAD_TIMEOUT_DEFAULT = 30.0

# 状态常量
STATUS_HEALTHY = "healthy"
STATUS_WARNING = "warning"
STATUS_CRITICAL = "critical"
STATUS_ERROR = "error"
STATUS_UNKNOWN = "unknown"
STATUS_ACTIVE = "active"
STATUS_INACTIVE = "inactive"
STATUS_ENABLED = "enabled"
STATUS_DISABLED = "disabled"

# 来源常量
SOURCE_SYSTEM = "system"
SOURCE_APPLICATION = "application"
SOURCE_DATABASE = "database"
SOURCE_NETWORK = "network"
SOURCE_CUSTOM = "custom"

# 错误处理函数 (临时定义，直到有专门的模块)
def handle_metrics_exceptions(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Metrics operation failed: {e}")
            return {"status": STATUS_ERROR, "error": str(e)}
    return wrapper

# 数据验证函数 (临时定义，直到有专门的模块)
def validate_metrics_data(metrics: Dict[str, Any]) -> bool:
    """简单的验证逻辑"""
    if not isinstance(metrics, dict):
        return False
    if "cpu_usage" in metrics and not (CPU_USAGE_MIN <= metrics["cpu_usage"] <= CPU_USAGE_MAX):
        return False
    return True

# 日志常量
LOG_LEVEL_DEBUG = "DEBUG"
LOG_LEVEL_INFO = "INFO"
LOG_LEVEL_WARNING = "WARNING"
LOG_LEVEL_ERROR = "ERROR"
LOG_LEVEL_CRITICAL = "CRITICAL"
DEFAULT_LOG_LEVEL = LOG_LEVEL_INFO

# 缓存常量
DEFAULT_CACHE_SIZE = 100
CACHE_TTL_DEFAULT = 300  # 5分钟
CACHE_CLEANUP_INTERVAL = 600  # 10分钟

# API常量
API_TIMEOUT_DEFAULT = 10.0
API_RETRY_COUNT_DEFAULT = 3
API_RETRY_DELAY_DEFAULT = 1.0

# 健康检查常量
HEALTH_CHECK_TIMEOUT_DEFAULT = 5.0
HEALTH_CHECK_INTERVAL_DEFAULT = 30.0
HEALTH_CHECK_FAILURE_THRESHOLD = 3

# 文件路径常量
DEFAULT_LOG_DIR = "/var/log/rqa2025"
DEFAULT_CONFIG_DIR = "/etc/rqa2025"
DEFAULT_DATA_DIR = "/var/lib/rqa2025"

# 单位转换常量 - 字节
BYTES_TO_KB = 1024
BYTES_TO_MB = 1024 * 1024
BYTES_TO_GB = 1024 * 1024 * 1024

# 单位转换常量 - 时间
SECONDS_TO_MINUTES = 60
MINUTES_TO_HOURS = 60

# 指标名称常量 - CPU
METRIC_CPU_USAGE = "cpu_usage_percent"
METRIC_CPU_COUNT = "cpu_count"
METRIC_CPU_COUNT_LOGICAL = "cpu_count_logical"
METRIC_CPU_FREQ_CURRENT = "cpu_freq_current"

# 指标名称常量 - 内存
METRIC_MEMORY_TOTAL = "memory_total"
METRIC_MEMORY_AVAILABLE = "memory_available"
METRIC_MEMORY_USED = "memory_used"
METRIC_MEMORY_FREE = "memory_free"
METRIC_MEMORY_PERCENT = "memory_percent"

# 指标名称常量 - 磁盘
METRIC_DISK_TOTAL = "disk_total"
METRIC_DISK_USED = "disk_used"
METRIC_DISK_FREE = "disk_free"
METRIC_DISK_PERCENT = "disk_percent"

# 指标名称常量 - 网络
METRIC_NETWORK_BYTES_SENT = "network_bytes_sent"
METRIC_NETWORK_BYTES_RECV = "network_bytes_recv"
METRIC_NETWORK_PACKETS_SENT = "network_packets_sent"
METRIC_NETWORK_PACKETS_RECV = "network_packets_recv"

# 指标名称常量 - GPU
METRIC_GPU_COUNT = "gpu_count"
METRIC_GPU_NAME = "gpu_name"
METRIC_GPU_MEMORY_TOTAL = "gpu_memory_total"
METRIC_GPU_MEMORY_USED = "gpu_memory_used"
METRIC_GPU_MEMORY_FREE = "gpu_memory_free"
METRIC_GPU_MEMORY_PERCENT = "gpu_memory_percent"
METRIC_GPU_UTILIZATION = "gpu_utilization"
METRIC_GPU_TEMPERATURE = "gpu_temperature"

# 消息常量
MESSAGE_SERVICE_UP = "Service is operating normally"
MESSAGE_SERVICE_DOWN = "Service is not responding"
MESSAGE_SERVICE_DEGRADED = "Service is experiencing issues"
MESSAGE_CHECK_TIMEOUT = "Health check timed out"
MESSAGE_CHECK_ERROR = "Health check encountered an error"

# 错误消息常量
ERROR_PERMISSION_DENIED = "权限不足"
ERROR_PROCESS_NOT_FOUND = "进程不存在"
ERROR_CONNECTION_FAILED = "连接失败"
ERROR_TIMEOUT = "操作超时"
ERROR_INVALID_DATA = "数据无效"
ERROR_COLLECTION_FAILED = "指标收集失败"

# 成功消息常量
SUCCESS_COLLECTION_STARTED = "指标收集已启动"
SUCCESS_COLLECTION_STOPPED = "指标收集已停止"
SUCCESS_DATA_STORED = "数据存储成功"
SUCCESS_HEALTH_CHECK_PASSED = "健康检查通过"

# 警告消息常量
WARNING_HIGH_CPU_USAGE = "CPU使用率过高"
WARNING_HIGH_MEMORY_USAGE = "内存使用率过高"
WARNING_LOW_DISK_SPACE = "磁盘空间不足"
WARNING_NETWORK_CONGESTION = "网络拥塞"

# 指标类型常量
METRIC_TYPE_COUNTER = "counter"
METRIC_TYPE_GAUGE = "gauge"
METRIC_TYPE_HISTOGRAM = "histogram"
METRIC_TYPE_SUMMARY = "summary"

# 标签常量
TAG_ENVIRONMENT = "environment"
TAG_SERVICE = "service"
TAG_COMPONENT = "component"
TAG_VERSION = "version"
TAG_DATACENTER = "datacenter"
TAG_TYPE_SYSTEM = "system"
TAG_TYPE_APPLICATION = "application"
TAG_CATEGORY_CPU = "cpu"
TAG_CATEGORY_MEMORY = "memory"
TAG_CATEGORY_DISK = "disk"
TAG_CATEGORY_NETWORK = "network"
TAG_DIRECTION_SENT = "sent"
TAG_DIRECTION_RECV = "recv"

# 线程常量
THREAD_NAME_PREFIX = "health_monitor_"
THREAD_DAEMON_DEFAULT = True
THREAD_JOIN_TIMEOUT = 5.0

# 健康评估常量
HEALTH_SCORE_EXCELLENT = 1.0
HEALTH_SCORE_GOOD = 0.8
HEALTH_SCORE_FAIR = 0.6
HEALTH_SCORE_POOR = 0.4
HEALTH_SCORE_CRITICAL = 0.2