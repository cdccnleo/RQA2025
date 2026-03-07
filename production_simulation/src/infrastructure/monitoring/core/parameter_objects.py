#!/usr/bin/env python3
"""
RQA2025 基础设施层监控参数对象

提供各种监控操作的参数对象类，用于替换长参数列表，提高代码可维护性。
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime

# 导入常量
from .constants import (
    DEFAULT_MONITOR_INTERVAL, FAST_MONITOR_INTERVAL, SLOW_MONITOR_INTERVAL,
    DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE, MAX_CONCURRENT_CHECKS,
    HEALTH_CHECK_TIMEOUT, HEALTH_CHECK_INTERVAL
)


@dataclass
class MetricRecordConfig:
    """指标记录配置"""
    name: str
    value: Any
    tags: Optional[Dict[str, str]] = field(default=None)
    timestamp: Optional[float] = field(default=None)
    app_name: Optional[str] = field(default=None)


@dataclass
class MetricsQueryConfig:
    """指标查询配置"""
    limit: int = 100
    start_time: Optional[datetime] = field(default=None)
    end_time: Optional[datetime] = field(default=None)
    metric_names: Optional[List[str]] = field(default=None)
    tags_filter: Optional[Dict[str, str]] = field(default=None)
    sort_by: str = "timestamp"
    sort_order: str = "desc"


@dataclass
class AlertConditionConfig:
    """告警条件配置"""
    field: str
    value: Any
    operator: str = "eq"
    threshold: Optional[float] = field(default=None)
    comparison_type: str = "absolute"  # 'absolute' 或 'percentage'
    description: Optional[str] = field(default=None)

    def __post_init__(self) -> None:
        operator_mapping = {
            ">": "gt",
            "gt": "gt",
            "greater_than": "gt",
            ">=": "ge",
            "ge": "ge",
            "greater_equal": "ge",
            "<": "lt",
            "lt": "lt",
            "less_than": "lt",
            "<=": "le",
            "le": "le",
            "less_equal": "le",
            "==": "eq",
            "=": "eq",
            "eq": "eq",
            "equals": "eq",
            "!=": "ne",
            "<>": "ne",
            "ne": "ne",
            "not_equals": "ne",
        }

        if isinstance(self.operator, str):
            normalized = operator_mapping.get(self.operator.strip().lower(), self.operator.strip().lower())
            self.operator = normalized


@dataclass
class AlertRuleConfig:
    """告警规则配置"""

    rule_id: str
    name: str
    description: str = ""
    condition: Optional[AlertConditionConfig] = None
    conditions: Optional[List[AlertConditionConfig]] = field(default=None)
    level: str = "warning"
    severity: Optional[str] = None
    channels: List[str] = field(default_factory=lambda: ["console"])
    enabled: bool = True
    cooldown: int = 300
    cooldown_seconds: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = field(default=None)

    def __post_init__(self) -> None:
        if self.conditions is None:
            self.conditions = [self.condition] if self.condition is not None else []
        else:
            normalized_conditions: List[AlertConditionConfig] = []
            for cond in self.conditions:
                if isinstance(cond, dict):
                    normalized_conditions.append(AlertConditionConfig(**cond))
                else:
                    normalized_conditions.append(cond)
            self.conditions = normalized_conditions
            if self.condition is None and self.conditions:
                self.condition = self.conditions[0]

        # 确保 condition 包含在 conditions 列表中，避免重复
        if self.condition is not None and self.condition not in self.conditions:
            self.conditions = [self.condition] + [cond for cond in self.conditions if cond is not self.condition]

        # 处理严重程度兼容字段
        if self.severity is None:
            self.severity = self.level
        else:
            # 以 severity 为准，同时确保 level 与之保持一致
            self.level = self.severity

        # 兼容 cooldown_seconds 字段
        if self.cooldown_seconds is not None:
            try:
                self.cooldown = int(self.cooldown_seconds)
            except (TypeError, ValueError):
                # 保持原 cooldown
                pass


@dataclass
class MonitoringConfig:
    """监控配置"""
    collection_interval: int = 60
    max_history_size: int = 1000
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'hit_rate_low': 0.8,
        'pool_usage_high': 0.9,
        'memory_high': 100.0,
    })
    enable_prometheus_export: bool = True
    prometheus_port: int = 9090


@dataclass
class PerformanceMetricsConfig:
    """性能指标收集配置"""
    include_detailed_stats: bool = True
    calculate_percentiles: bool = True
    include_memory_stats: bool = True
    include_thread_stats: bool = True
    include_queue_stats: bool = True
    include_error_stats: bool = True
    include_performance_trends: bool = True
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    alert_on_anomalies: bool = False
    anomaly_detection_config: Optional[Dict[str, Any]] = field(default=None)
    export_format: str = "json"
    export_path: Optional[str] = field(default=None)
    create_charts: bool = False
    chart_config: Optional[Dict[str, Any]] = field(default=None)
    summary_report: bool = True
    report_config: Optional[Dict[str, Any]] = field(default=None)
    enable_ai_insights: bool = False
    ai_config: Optional[Dict[str, Any]] = field(default=None)
    custom_metrics: Optional[List[str]] = field(default=None)
    custom_config: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class CoverageCollectionConfig:
    """覆盖率收集配置"""
    include_line_coverage: bool = True
    include_branch_coverage: bool = True
    include_function_coverage: bool = True
    include_class_coverage: bool = True
    exclude_patterns: List[str] = field(default_factory=lambda: ["test_*", "*_test.py"])
    source_dirs: List[str] = field(default_factory=lambda: ["src"])
    report_formats: List[str] = field(default_factory=lambda: ["html", "xml"])
    minimum_coverage: float = 80.0
    fail_under_minimum: bool = False


@dataclass
class ResourceUsageConfig:
    """资源使用收集配置"""
    include_cpu: bool = True
    include_memory: bool = True
    include_disk: bool = True
    include_network: bool = True
    include_processes: bool = False
    include_system_load: bool = True
    sampling_interval: float = 1.0
    collection_duration: int = 60
    enable_historical_tracking: bool = True
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'cpu_percent': 80.0,
        'memory_percent': 85.0,
        'disk_percent': 90.0,
    })


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    check_timeout: int = 30
    retry_count: int = 3
    retry_delay: float = 1.0
    enable_detailed_logging: bool = False
    custom_checks: List[str] = field(default_factory=list)
    health_score_weights: Dict[str, float] = field(default_factory=lambda: {
        'response_time': 0.3,
        'error_rate': 0.3,
        'resource_usage': 0.2,
        'availability': 0.2,
    })


@dataclass
class PrometheusExportConfig:
    """Prometheus导出配置"""
    include_help_text: bool = True
    include_type_info: bool = True
    metric_prefix: str = "monitoring"
    label_names: List[str] = field(default_factory=lambda: ["service", "instance"])
    default_labels: Dict[str, str] = field(default_factory=dict)
    export_timeout: int = 30
    enable_compression: bool = False


@dataclass
class OptimizationSuggestionConfig:
    """优化建议配置"""
    enable_ai_suggestions: bool = True
    max_suggestions_per_category: int = 10
    suggestion_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'coverage_drop': 5.0,
        'performance_degradation': 10.0,
        'memory_increase': 20.0,
    })
    priority_weights: Dict[str, int] = field(default_factory=lambda: {
        'critical': 5,
        'high': 3,
        'medium': 2,
        'low': 1,
    })


@dataclass
class DataPersistenceConfig:
    """数据持久化配置"""
    enable_file_storage: bool = True
    enable_database_storage: bool = False
    storage_path: str = "data/monitoring"
    max_file_age_days: int = 30
    compression_enabled: bool = True
    backup_enabled: bool = True
    backup_interval_hours: int = 24
    retention_policy: str = "time_based"  # 'time_based' 或 'size_based'
    max_storage_size_mb: int = 1024


@dataclass
class LoggerPoolStatsConfig:
    """Logger池统计配置"""
    include_hit_rate: bool = True
    include_memory_usage: bool = True
    include_pool_size: bool = True
    include_access_times: bool = True
    include_error_counts: bool = True
    calculate_percentiles: bool = True
    percentile_values: List[float] = field(default_factory=lambda: [50.0, 95.0, 99.0])
    enable_trend_analysis: bool = True
    trend_window_size: int = 10


# ============================================================================
# 新增参数对象 - 用于替换长参数列表
# ============================================================================

@dataclass
class ApplicationMonitorInitConfig:
    """应用监控器初始化配置"""
    pool_name: str
    monitor_interval: int = DEFAULT_MONITOR_INTERVAL
    enable_performance_monitoring: bool = True
    enable_error_tracking: bool = True
    enable_resource_monitoring: bool = True
    max_metrics_cache_size: int = 10000
    metrics_retention_hours: int = 24
    alert_thresholds: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class MetricsRecordConfig:
    """指标记录配置"""
    name: str
    value: Any
    timestamp: Optional[float] = field(default=None)
    tags: Optional[Dict[str, str]] = field(default=None)
    app_name: Optional[str] = field(default=None)
    instance_id: Optional[str] = field(default=None)
    environment: Optional[str] = field(default=None)
    version: Optional[str] = field(default=None)


@dataclass
class MetricsQueryConfig:
    """指标查询配置"""
    metric_names: List[str]
    start_time: Optional[float] = field(default=None)
    end_time: Optional[float] = field(default=None)
    tags_filter: Optional[Dict[str, str]] = field(default=None)
    aggregation_type: str = "avg"  # 'avg', 'sum', 'min', 'max', 'count'
    group_by_tags: Optional[List[str]] = field(default=None)
    limit: int = DEFAULT_PAGE_SIZE
    offset: int = 0


@dataclass
class HealthCheckConfig:
    """健康检查配置"""
    component_name: str
    check_type: str = "basic"  # 'basic', 'deep', 'comprehensive'
    timeout_seconds: int = HEALTH_CHECK_TIMEOUT
    include_dependencies: bool = True
    include_performance_metrics: bool = True
    custom_checks: Optional[List[str]] = field(default=None)
    environment_context: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class PerformanceMetricsCollectionConfig:
    """性能指标收集配置"""
    cpu_usage: bool = True
    memory_usage: bool = True
    disk_usage: bool = True
    network_io: bool = True
    thread_count: bool = True
    process_count: bool = True
    response_time: bool = True
    error_rate: bool = True
    throughput: bool = True
    cache_hit_rate: bool = True
    connection_pool_size: bool = True
    database_connections: bool = True
    api_calls: bool = True
    latency: bool = True
    gc_collections: bool = True
    io_operations: bool = True
    custom_metrics: Optional[List[str]] = field(default=None)


@dataclass
class AlertTriggerConfig:
    """告警触发配置"""
    alert_type: str
    severity: str  # 'info', 'warning', 'error', 'critical'
    title: str
    description: str
    component_name: str
    metric_name: Optional[str] = field(default=None)
    current_value: Optional[Any] = field(default=None)
    threshold_value: Optional[Any] = field(default=None)
    tags: Optional[Dict[str, str]] = field(default=None)
    context_info: Optional[Dict[str, Any]] = field(default=None)


@dataclass
class PrometheusMetricsExportConfig:
    """Prometheus指标导出配置"""
    include_system_metrics: bool = True
    include_application_metrics: bool = True
    include_custom_metrics: bool = True
    metric_prefix: str = "rqa2025_monitoring"
    export_format: str = "prometheus"  # 'prometheus', 'json', 'xml'
    include_timestamps: bool = True
    include_labels: bool = True
    compression_enabled: bool = False
    batch_size: int = 1000
    export_interval_seconds: int = 15
    endpoint_url: Optional[str] = field(default=None)
    authentication_token: Optional[str] = field(default=None)


@dataclass
class StatsCollectionConfig:
    """统计收集配置"""
    pool_name: str
    collection_interval: int = DEFAULT_MONITOR_INTERVAL
    include_hit_rate: bool = True
    include_memory_usage: bool = True
    include_pool_size: bool = True
    include_access_patterns: bool = True
    include_error_stats: bool = True
    enable_real_time_updates: bool = True
    max_stats_history: int = 1000
    enable_anomaly_detection: bool = False
    anomaly_threshold: float = 2.0
    enable_trend_analysis: bool = True
    trend_window_size: int = 10
    enable_forecasting: bool = False
    forecasting_periods: int = 5
