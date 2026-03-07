"""
监控模块常量测试套件

针对src/infrastructure/health/monitoring/constants.py进行全面测试
验证常量定义的正确性和合理性
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from src.infrastructure.health.monitoring.constants import (
    DEFAULT_HISTORY_SIZE,
    MAX_HISTORY_SIZE,
    MIN_HISTORY_SIZE,
    COLLECTION_INTERVAL_DEFAULT,
    COLLECTION_INTERVAL_MIN,
    COLLECTION_INTERVAL_MAX,
    COLLECTION_INTERVAL_FAST,
    COLLECTION_INTERVAL_SLOW,
    CPU_THRESHOLD_WARNING,
    CPU_THRESHOLD_CRITICAL,
    MEMORY_THRESHOLD_WARNING,
    MEMORY_THRESHOLD_CRITICAL,
    DISK_THRESHOLD_WARNING,
    DISK_THRESHOLD_CRITICAL,
    NETWORK_THRESHOLD_WARNING,
    NETWORK_THRESHOLD_CRITICAL,
    QUALITY_SCORE_MIN,
    QUALITY_SCORE_DEFAULT,
    QUALITY_SCORE_WARNING,
    QUALITY_SCORE_CRITICAL,
    QUALITY_SCORE_MAX,
    DEFAULT_HISTORY_HOURS,
    HISTORY_HOURS_OPTIONS,
    DEFAULT_THREAD_POOL_SIZE,
    MAX_THREAD_POOL_SIZE,
    THREAD_TIMEOUT_DEFAULT,
    DEFAULT_CACHE_SIZE,
    CACHE_TTL_DEFAULT,
    CACHE_CLEANUP_INTERVAL,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_INFO,
    LOG_LEVEL_WARNING,
    LOG_LEVEL_ERROR,
    LOG_LEVEL_CRITICAL,
    DEFAULT_LOG_LEVEL,
    API_TIMEOUT_DEFAULT,
    API_RETRY_COUNT_DEFAULT,
    API_RETRY_DELAY_DEFAULT,
    HEALTH_CHECK_TIMEOUT_DEFAULT,
    HEALTH_CHECK_INTERVAL_DEFAULT,
    HEALTH_CHECK_FAILURE_THRESHOLD,
    DEFAULT_LOG_DIR,
    DEFAULT_CONFIG_DIR,
    DEFAULT_DATA_DIR,
    BYTES_TO_KB,
    BYTES_TO_MB,
    BYTES_TO_GB,
    SECONDS_TO_MINUTES,
    MINUTES_TO_HOURS,
    SECONDS_TO_HOURS,
    METRIC_CPU_USAGE,
    METRIC_CPU_COUNT,
    METRIC_CPU_COUNT_LOGICAL,
    METRIC_CPU_FREQ_CURRENT,
    METRIC_MEMORY_TOTAL,
    METRIC_MEMORY_AVAILABLE,
    METRIC_MEMORY_USED,
    METRIC_MEMORY_FREE,
    METRIC_MEMORY_PERCENT,
    METRIC_DISK_TOTAL,
    METRIC_DISK_USED,
    METRIC_DISK_FREE,
    METRIC_DISK_PERCENT,
    METRIC_NETWORK_BYTES_SENT,
    METRIC_NETWORK_BYTES_RECV,
    METRIC_NETWORK_PACKETS_SENT,
    METRIC_NETWORK_PACKETS_RECV,
    METRIC_GPU_COUNT,
    METRIC_GPU_NAME,
    METRIC_GPU_MEMORY_TOTAL,
    METRIC_GPU_MEMORY_USED,
    METRIC_GPU_MEMORY_FREE,
    METRIC_GPU_MEMORY_PERCENT,
    METRIC_GPU_UTILIZATION,
    METRIC_GPU_TEMPERATURE,
    STATUS_HEALTHY,
    STATUS_WARNING,
    STATUS_CRITICAL,
    STATUS_ERROR,
    STATUS_UNKNOWN,
    STATUS_ACTIVE,
    STATUS_INACTIVE,
    STATUS_ENABLED,
    STATUS_DISABLED,
    SOURCE_SYSTEM,
    SOURCE_APPLICATION,
    SOURCE_DATABASE,
    SOURCE_NETWORK,
    SOURCE_CUSTOM,
    METRIC_TYPE_COUNTER,
    METRIC_TYPE_GAUGE,
    METRIC_TYPE_HISTOGRAM,
    METRIC_TYPE_SUMMARY,
    TAG_ENVIRONMENT,
    TAG_SERVICE,
    TAG_COMPONENT,
    TAG_VERSION,
    TAG_DATACENTER,
    TAG_TYPE_SYSTEM,
    TAG_TYPE_APPLICATION,
    TAG_CATEGORY_CPU,
    TAG_CATEGORY_MEMORY,
    TAG_CATEGORY_DISK,
    TAG_CATEGORY_NETWORK,
    TAG_DIRECTION_SENT,
    TAG_DIRECTION_RECV,
    COUNTER_DEFAULT_VALUE,
    COUNTER_API_CALLS,
    COUNTER_CACHE_HITS,
    COUNTER_CACHE_MISSES,
    COUNTER_DB_CONNECTIONS,
    COUNTER_ERRORS,
    COUNTER_WARNINGS,
    THREAD_NAME_PREFIX,
    THREAD_DAEMON_DEFAULT,
    THREAD_JOIN_TIMEOUT,
    THREAD_JOIN_TIMEOUT_DEFAULT,
    DEFAULT_HISTORY_QUERY_HOURS,
    AVERAGE_CALCULATION_HOURS,
    DEFAULT_HISTORY_LIMIT,
    RESPONSE_TIME_THRESHOLD_MS,
    HISTORY_CAPACITY_WARNING_RATIO,
    TREND_CALCULATION_PERIODS,
    TREND_CHANGE_THRESHOLD_RATIO,
    CPU_USAGE_MIN,
    CPU_USAGE_MAX,
    MEMORY_USAGE_MIN,
    MEMORY_USAGE_MAX,
    DISK_USAGE_MIN,
    DISK_USAGE_MAX,
    CPU_HEALTHY_THRESHOLD,
    MEMORY_HEALTHY_THRESHOLD,
    DISK_HEALTHY_THRESHOLD,
    QUALITY_SCORE_PENALTY_PER_ERROR,
    GPU_PERCENTAGE_MULTIPLIER,
    SECONDS_TO_HOURS,
    MESSAGE_SERVICE_UP,
    MESSAGE_SERVICE_DOWN,
    ERROR_PERMISSION_DENIED,
    ERROR_PROCESS_NOT_FOUND,
    ERROR_CONNECTION_FAILED,
    ERROR_TIMEOUT,
    ERROR_INVALID_DATA,
    ERROR_COLLECTION_FAILED,
    SUCCESS_COLLECTION_STARTED,
    SUCCESS_COLLECTION_STOPPED,
    SUCCESS_DATA_STORED,
    SUCCESS_HEALTH_CHECK_PASSED,
    WARNING_HIGH_CPU_USAGE,
    WARNING_HIGH_MEMORY_USAGE,
    WARNING_LOW_DISK_SPACE,
    WARNING_NETWORK_CONGESTION
)
class TestMonitoringConstants:
    """监控常量测试"""

    def test_history_size_constants(self):
        """测试历史数据大小常量"""
        # 验证常量存在性
        assert hasattr(__import__('src.infrastructure.health.monitoring.constants', fromlist=['DEFAULT_HISTORY_SIZE']), 'DEFAULT_HISTORY_SIZE')

        # 验证常量值
        assert DEFAULT_HISTORY_SIZE == 1000
        assert MAX_HISTORY_SIZE == 10000
        assert MIN_HISTORY_SIZE == 100

        # 验证范围关系
        assert MIN_HISTORY_SIZE <= DEFAULT_HISTORY_SIZE <= MAX_HISTORY_SIZE

    def test_collection_interval_constants(self):
        """测试收集间隔常量"""
        # 验证常量值
        assert COLLECTION_INTERVAL_DEFAULT == 1.0
        assert COLLECTION_INTERVAL_MIN == 0.1
        assert COLLECTION_INTERVAL_MAX == 60.0
        assert COLLECTION_INTERVAL_FAST == 0.5
        assert COLLECTION_INTERVAL_SLOW == 5.0

        # 验证范围关系
        assert COLLECTION_INTERVAL_MIN <= COLLECTION_INTERVAL_DEFAULT <= COLLECTION_INTERVAL_MAX
        assert COLLECTION_INTERVAL_FAST >= COLLECTION_INTERVAL_MIN
        assert COLLECTION_INTERVAL_SLOW <= COLLECTION_INTERVAL_MAX

        # 验证快慢速关系
        assert COLLECTION_INTERVAL_FAST < COLLECTION_INTERVAL_DEFAULT < COLLECTION_INTERVAL_SLOW

    def test_performance_threshold_constants(self):
        """测试性能阈值常量"""
        # CPU阈值
        assert CPU_THRESHOLD_WARNING == 80.0
        assert CPU_THRESHOLD_CRITICAL == 95.0
        assert CPU_THRESHOLD_WARNING < CPU_THRESHOLD_CRITICAL

        # 内存阈值
        assert MEMORY_THRESHOLD_WARNING == 85.0
        assert MEMORY_THRESHOLD_CRITICAL == 95.0
        assert MEMORY_THRESHOLD_WARNING < MEMORY_THRESHOLD_CRITICAL

        # 磁盘阈值
        assert DISK_THRESHOLD_WARNING == 85.0
        assert DISK_THRESHOLD_CRITICAL == 95.0
        assert DISK_THRESHOLD_WARNING < DISK_THRESHOLD_CRITICAL

        # 网络阈值
        assert NETWORK_THRESHOLD_WARNING == 80.0
        assert NETWORK_THRESHOLD_CRITICAL == 95.0
        assert NETWORK_THRESHOLD_WARNING < NETWORK_THRESHOLD_CRITICAL

    def test_data_quality_constants(self):
        """测试数据质量常量"""
        # 质量评分范围
        assert QUALITY_SCORE_MIN == 0.0
        assert QUALITY_SCORE_MAX == 1.0
        assert QUALITY_SCORE_DEFAULT == 1.0

        # 验证范围
        assert QUALITY_SCORE_MIN < QUALITY_SCORE_DEFAULT <= QUALITY_SCORE_MAX

        # 质量阈值
        assert QUALITY_SCORE_WARNING == 0.7
        assert QUALITY_SCORE_CRITICAL == 0.5
        assert QUALITY_SCORE_CRITICAL < QUALITY_SCORE_WARNING < QUALITY_SCORE_MAX

    def test_time_window_constants(self):
        """测试时间窗口常量"""
        assert DEFAULT_HISTORY_HOURS == 24

        # 验证时间窗口选项
        expected_options = [1, 6, 12, 24, 72, 168]
        assert HISTORY_HOURS_OPTIONS == expected_options

        # 验证默认值在选项中
        assert DEFAULT_HISTORY_HOURS in HISTORY_HOURS_OPTIONS

    def test_concurrency_constants(self):
        """测试并发控制常量"""
        assert DEFAULT_THREAD_POOL_SIZE == 4
        assert MAX_THREAD_POOL_SIZE == 16
        assert THREAD_TIMEOUT_DEFAULT == 30.0

        # 验证范围关系
        assert DEFAULT_THREAD_POOL_SIZE <= MAX_THREAD_POOL_SIZE

    def test_log_constants(self):
        """测试日志常量"""
        # 日志级别
        assert LOG_LEVEL_DEBUG == "DEBUG"
        assert LOG_LEVEL_INFO == "INFO"
        assert LOG_LEVEL_WARNING == "WARNING"
        assert LOG_LEVEL_ERROR == "ERROR"
        assert LOG_LEVEL_CRITICAL == "CRITICAL"

        # 默认日志级别
        assert DEFAULT_LOG_LEVEL == LOG_LEVEL_INFO

    def test_cache_constants(self):
        """测试缓存常量"""
        assert DEFAULT_CACHE_SIZE == 100
        assert CACHE_TTL_DEFAULT == 300  # 5分钟
        assert CACHE_CLEANUP_INTERVAL == 600  # 10分钟

        # 验证时间关系
        assert CACHE_CLEANUP_INTERVAL > CACHE_TTL_DEFAULT

    def test_api_constants(self):
        """测试API常量"""
        assert API_TIMEOUT_DEFAULT == 10.0
        assert API_RETRY_COUNT_DEFAULT == 3
        assert API_RETRY_DELAY_DEFAULT == 1.0

        # 验证合理性
        assert API_TIMEOUT_DEFAULT > 0
        assert API_RETRY_COUNT_DEFAULT > 0
        assert API_RETRY_DELAY_DEFAULT > 0

    def test_health_check_constants(self):
        """测试健康检查常量"""
        assert HEALTH_CHECK_TIMEOUT_DEFAULT == 5.0
        assert HEALTH_CHECK_INTERVAL_DEFAULT == 30.0
        assert HEALTH_CHECK_FAILURE_THRESHOLD == 3

        # 验证合理性
        assert HEALTH_CHECK_TIMEOUT_DEFAULT > 0
        assert HEALTH_CHECK_INTERVAL_DEFAULT > HEALTH_CHECK_TIMEOUT_DEFAULT
        assert HEALTH_CHECK_FAILURE_THRESHOLD > 0

    def test_file_path_constants(self):
        """测试文件路径常量"""
        assert DEFAULT_LOG_DIR == "/var/log/rqa2025"
        assert DEFAULT_CONFIG_DIR == "/etc/rqa2025"
        assert DEFAULT_DATA_DIR == "/var/lib/rqa2025"

        # 验证路径格式
        assert all(path.startswith('/') for path in [DEFAULT_LOG_DIR, DEFAULT_CONFIG_DIR, DEFAULT_DATA_DIR])

    def test_unit_conversion_constants(self):
        """测试单位转换常量"""
        # 字节转换
        assert BYTES_TO_KB == 1024
        assert BYTES_TO_MB == 1024 * 1024
        assert BYTES_TO_GB == 1024 * 1024 * 1024

        # 验证递进关系
        assert BYTES_TO_KB < BYTES_TO_MB < BYTES_TO_GB

        # 时间转换
        assert SECONDS_TO_MINUTES == 60
        assert SECONDS_TO_HOURS == 3600
        assert MINUTES_TO_HOURS == 60

        # 验证时间关系
        assert SECONDS_TO_MINUTES * MINUTES_TO_HOURS == SECONDS_TO_HOURS

    def test_metric_name_constants(self):
        """测试指标名称常量"""
        # CPU指标
        assert METRIC_CPU_USAGE == "cpu_usage_percent"
        assert METRIC_CPU_COUNT == "cpu_count"
        assert METRIC_CPU_COUNT_LOGICAL == "cpu_count_logical"
        assert METRIC_CPU_FREQ_CURRENT == "cpu_freq_current"

        # 内存指标
        assert METRIC_MEMORY_TOTAL == "memory_total"
        assert METRIC_MEMORY_AVAILABLE == "memory_available"
        assert METRIC_MEMORY_USED == "memory_used"
        assert METRIC_MEMORY_FREE == "memory_free"
        assert METRIC_MEMORY_PERCENT == "memory_percent"

        # 磁盘指标
        assert METRIC_DISK_TOTAL == "disk_total"
        assert METRIC_DISK_USED == "disk_used"
        assert METRIC_DISK_FREE == "disk_free"
        assert METRIC_DISK_PERCENT == "disk_percent"

        # 网络指标
        assert METRIC_NETWORK_BYTES_SENT == "network_bytes_sent"
        assert METRIC_NETWORK_BYTES_RECV == "network_bytes_recv"
        assert METRIC_NETWORK_PACKETS_SENT == "network_packets_sent"
        assert METRIC_NETWORK_PACKETS_RECV == "network_packets_recv"

        # GPU指标
        assert METRIC_GPU_COUNT == "gpu_count"
        assert METRIC_GPU_NAME == "gpu_name"
        assert METRIC_GPU_MEMORY_TOTAL == "gpu_memory_total"
        assert METRIC_GPU_MEMORY_USED == "gpu_memory_used"
        assert METRIC_GPU_MEMORY_PERCENT == "gpu_memory_percent"

    def test_message_constants(self):
        """测试消息常量"""
        # 错误消息
        assert ERROR_PERMISSION_DENIED == "权限不足"
        assert ERROR_PROCESS_NOT_FOUND == "进程不存在"
        assert ERROR_CONNECTION_FAILED == "连接失败"
        assert ERROR_TIMEOUT == "操作超时"
        assert ERROR_INVALID_DATA == "数据无效"
        assert ERROR_COLLECTION_FAILED == "指标收集失败"

        # 成功消息
        assert SUCCESS_COLLECTION_STARTED == "指标收集已启动"
        assert SUCCESS_COLLECTION_STOPPED == "指标收集已停止"
        assert SUCCESS_DATA_STORED == "数据存储成功"
        assert SUCCESS_HEALTH_CHECK_PASSED == "健康检查通过"

        # 警告消息
        assert WARNING_HIGH_CPU_USAGE == "CPU使用率过高"
        assert WARNING_HIGH_MEMORY_USAGE == "内存使用率过高"
        assert WARNING_LOW_DISK_SPACE == "磁盘空间不足"
        assert WARNING_NETWORK_CONGESTION == "网络拥塞"

    def test_status_constants(self):
        """测试状态常量"""
        # 健康状态
        assert STATUS_HEALTHY == "healthy"
        assert STATUS_WARNING == "warning"
        assert STATUS_CRITICAL == "critical"
        assert STATUS_ERROR == "error"
        assert STATUS_UNKNOWN == "unknown"

        # 布尔值状态
        assert STATUS_ACTIVE == "active"
        assert STATUS_INACTIVE == "inactive"
        assert STATUS_ENABLED == "enabled"
        assert STATUS_DISABLED == "disabled"

    def test_source_constants(self):
        """测试数据源常量"""
        assert SOURCE_SYSTEM == "system"
        assert SOURCE_APPLICATION == "application"
        assert SOURCE_DATABASE == "database"
        assert SOURCE_NETWORK == "network"
        assert SOURCE_CUSTOM == "custom"

    def test_metric_type_constants(self):
        """测试指标类型常量"""
        assert METRIC_TYPE_GAUGE == "gauge"
        assert METRIC_TYPE_COUNTER == "counter"
        assert METRIC_TYPE_HISTOGRAM == "histogram"
        assert METRIC_TYPE_SUMMARY == "summary"

    def test_tag_constants(self):
        """测试标签常量"""
        assert TAG_TYPE_SYSTEM == "system"
        assert TAG_TYPE_APPLICATION == "application"
        assert TAG_CATEGORY_CPU == "cpu"
        assert TAG_CATEGORY_MEMORY == "memory"
        assert TAG_CATEGORY_DISK == "disk"
        assert TAG_CATEGORY_NETWORK == "network"
        assert TAG_DIRECTION_SENT == "sent"
        assert TAG_DIRECTION_RECV == "recv"

    def test_counter_constants(self):
        """测试计数器常量"""
        assert COUNTER_DEFAULT_VALUE == 0

        # 计数器名称
        assert COUNTER_API_CALLS == "api_calls"
        assert COUNTER_CACHE_HITS == "cache_hits"
        assert COUNTER_CACHE_MISSES == "cache_misses"
        assert COUNTER_DB_CONNECTIONS == "db_connections"
        assert COUNTER_ERRORS == "errors"
        assert COUNTER_WARNINGS == "warnings"

    def test_thread_constants(self):
        """测试线程常量"""
        assert THREAD_JOIN_TIMEOUT_DEFAULT == 2.0
        assert THREAD_JOIN_TIMEOUT_DEFAULT > 0

    def test_history_query_constants(self):
        """测试历史查询常量"""
        assert DEFAULT_HISTORY_QUERY_HOURS == 24
        assert AVERAGE_CALCULATION_HOURS == 1
        assert DEFAULT_HISTORY_LIMIT == 100

        # 验证合理性
        assert DEFAULT_HISTORY_QUERY_HOURS > 0
        assert AVERAGE_CALCULATION_HOURS > 0
        assert DEFAULT_HISTORY_LIMIT > 0

    def test_health_thresholds(self):
        """测试健康阈值常量"""
        assert RESPONSE_TIME_THRESHOLD_MS == 100
        assert HISTORY_CAPACITY_WARNING_RATIO == 0.9
        assert TREND_CALCULATION_PERIODS == 5
        assert TREND_CHANGE_THRESHOLD_RATIO == 0.05  # 5%

        # 验证合理性
        assert RESPONSE_TIME_THRESHOLD_MS > 0
        assert 0 < HISTORY_CAPACITY_WARNING_RATIO < 1
        assert TREND_CALCULATION_PERIODS > 0
        assert 0 < TREND_CHANGE_THRESHOLD_RATIO < 1

    def test_validation_range_constants(self):
        """测试数据验证范围常量"""
        # CPU使用率范围
        assert CPU_USAGE_MIN == 0.0
        assert CPU_USAGE_MAX == 100.0
        assert CPU_USAGE_MIN < CPU_USAGE_MAX

        # 内存使用率范围
        assert MEMORY_USAGE_MIN == 0.0
        assert MEMORY_USAGE_MAX == 100.0
        assert MEMORY_USAGE_MIN < MEMORY_USAGE_MAX

        # 磁盘使用率范围
        assert DISK_USAGE_MIN == 0.0
        assert DISK_USAGE_MAX == 100.0
        assert DISK_USAGE_MIN < DISK_USAGE_MAX

    def test_health_assessment_constants(self):
        """测试健康状态评估常量"""
        # CPU健康阈值 - 注意这里是使用率阈值，低于此值认为健康
        assert CPU_HEALTHY_THRESHOLD == 90.0

        # 内存健康阈值
        assert MEMORY_HEALTHY_THRESHOLD == 85.0

        # 磁盘健康阈值
        assert DISK_HEALTHY_THRESHOLD == 95.0

        # 验证阈值合理性
        assert 0 < CPU_HEALTHY_THRESHOLD <= 100
        assert 0 < MEMORY_HEALTHY_THRESHOLD <= 100
        assert 0 < DISK_HEALTHY_THRESHOLD <= 100

    def test_quality_scoring_constants(self):
        """测试质量评分常量"""
        assert QUALITY_SCORE_PENALTY_PER_ERROR == 0.1
        assert GPU_PERCENTAGE_MULTIPLIER == 100.0

        # 验证合理性
        assert 0 < QUALITY_SCORE_PENALTY_PER_ERROR < 1
        assert GPU_PERCENTAGE_MULTIPLIER > 0

    def test_constant_consistency_validation(self):
        """测试常量一致性验证"""
        # 验证警告阈值小于临界阈值
        assert CPU_THRESHOLD_WARNING < CPU_THRESHOLD_CRITICAL
        assert MEMORY_THRESHOLD_WARNING < MEMORY_THRESHOLD_CRITICAL
        assert DISK_THRESHOLD_WARNING < DISK_THRESHOLD_CRITICAL
        assert NETWORK_THRESHOLD_WARNING < NETWORK_THRESHOLD_CRITICAL

        # 验证质量评分范围合理
        assert QUALITY_SCORE_MIN < QUALITY_SCORE_WARNING < QUALITY_SCORE_MAX
        assert QUALITY_SCORE_MIN < QUALITY_SCORE_CRITICAL < QUALITY_SCORE_WARNING

        # 验证历史大小范围合理
        assert MIN_HISTORY_SIZE < DEFAULT_HISTORY_SIZE < MAX_HISTORY_SIZE

        # 验证收集间隔范围合理
        assert COLLECTION_INTERVAL_MIN < COLLECTION_INTERVAL_DEFAULT < COLLECTION_INTERVAL_MAX
        assert COLLECTION_INTERVAL_MIN < COLLECTION_INTERVAL_FAST < COLLECTION_INTERVAL_DEFAULT
        assert COLLECTION_INTERVAL_DEFAULT < COLLECTION_INTERVAL_SLOW < COLLECTION_INTERVAL_MAX

    def test_constant_type_validation(self):
        """测试常量类型验证"""
        # 数值类型常量
        numeric_constants = [
            DEFAULT_HISTORY_SIZE, MAX_HISTORY_SIZE, MIN_HISTORY_SIZE,
            COLLECTION_INTERVAL_DEFAULT, COLLECTION_INTERVAL_MIN, COLLECTION_INTERVAL_MAX,
            CPU_THRESHOLD_WARNING, CPU_THRESHOLD_CRITICAL,
            MEMORY_THRESHOLD_WARNING, MEMORY_THRESHOLD_CRITICAL,
            DISK_THRESHOLD_WARNING, DISK_THRESHOLD_CRITICAL,
            NETWORK_THRESHOLD_WARNING, NETWORK_THRESHOLD_CRITICAL,
            QUALITY_SCORE_MIN, QUALITY_SCORE_MAX, QUALITY_SCORE_DEFAULT,
            DEFAULT_THREAD_POOL_SIZE, MAX_THREAD_POOL_SIZE, THREAD_TIMEOUT_DEFAULT,
            DEFAULT_CACHE_SIZE, CACHE_TTL_DEFAULT, CACHE_CLEANUP_INTERVAL,
            API_TIMEOUT_DEFAULT, API_RETRY_COUNT_DEFAULT, API_RETRY_DELAY_DEFAULT,
            HEALTH_CHECK_TIMEOUT_DEFAULT, HEALTH_CHECK_INTERVAL_DEFAULT, HEALTH_CHECK_FAILURE_THRESHOLD
        ]

        # 验证都是数值类型
        for const in numeric_constants:
            assert isinstance(const, (int, float)), f"常量 {const} 应该是数值类型"

        # 字符串类型常量
        string_constants = [
            LOG_LEVEL_DEBUG, LOG_LEVEL_INFO, LOG_LEVEL_WARNING, LOG_LEVEL_ERROR, LOG_LEVEL_CRITICAL,
            DEFAULT_LOG_LEVEL, DEFAULT_LOG_DIR, DEFAULT_CONFIG_DIR, DEFAULT_DATA_DIR,
            METRIC_CPU_USAGE, METRIC_CPU_COUNT, STATUS_HEALTHY, STATUS_WARNING,
            SOURCE_SYSTEM, SOURCE_APPLICATION, METRIC_TYPE_GAUGE, METRIC_TYPE_COUNTER,
            ERROR_PERMISSION_DENIED, SUCCESS_COLLECTION_STARTED, WARNING_HIGH_CPU_USAGE
        ]

        # 验证都是字符串类型
        for const in string_constants:
            assert isinstance(const, str), f"常量 {const} 应该是字符串类型"

        # 列表类型常量
        assert isinstance(HISTORY_HOURS_OPTIONS, list)
        assert all(isinstance(x, int) for x in HISTORY_HOURS_OPTIONS)
