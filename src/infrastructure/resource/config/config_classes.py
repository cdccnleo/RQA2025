from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

"""
配置数据类集合

资源管理模块中的多个子系统共用一组配置数据结构，
原始实现散落在不同文件且存在循环依赖。为了保持
向后兼容并便于单元测试，这里集中定义常用的数据类。
"""


@dataclass
class TaskConfig:
    """任务调度配置"""

    task_type: str = "generic_task"
    priority: int = 3
    timeout: int = 3600
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProcessConfig:
    """资源流程配置"""

    action: str = "allocate_quota"
    params: Dict[str, Any] = field(default_factory=dict)
    retries: int = 0
    timeout: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonitorConfig:
    """监控配置"""

    monitor_interval: int = 60
    history_size: int = 1000
    enable_alerts: bool = True
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_warning": 80.0,
        "memory_warning": 85.0,
        "disk_warning": 90.0,
    })
    monitor_targets: List[str] = field(default_factory=lambda: ["cpu", "memory", "disk"])
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertConfig:
    """告警配置"""

    severity: str = "medium"
    message: str = ""
    threshold: Optional[float] = None
    operator: str = "gt"
    channels: List[Dict[str, Any]] = field(default_factory=list)
    routing_rules: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceConfig:
    """资源配置"""

    resource_types: List[str] = field(default_factory=lambda: ["cpu", "memory", "disk"])
    default_limits: Dict[str, float] = field(default_factory=dict)
    allocation_strategy: str = "balanced"
    auto_scale_enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationConfig:
    """优化配置"""

    optimization_type: str = "performance"
    constraints: Dict[str, Any] = field(default_factory=dict)
    targets: Dict[str, Any] = field(default_factory=dict)
    enable_recommendations: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class APIConfig:
    """API配置"""

    base_url: str = ""
    timeout: int = 30
    max_retries: int = 3
    headers: Dict[str, str] = field(default_factory=dict)
    verify_ssl: bool = True
    extra_options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyUsageConfig:
    """策略使用配置"""

    default_strategy: str = "balanced"
    strategies: Dict[str, bool] = field(default_factory=lambda: {"balanced": True})
    fallback_strategy: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationReportConfig:
    """优化报告配置"""

    report_type: str = "summary"
    include_sections: List[str] = field(default_factory=lambda: ["system_resources", "recommendations"])
    output_format: str = "json"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMonitorConfig:
    """资源监控配置"""

    monitor_interval: int = 60  # 监控间隔秒
    alert_threshold: Dict[str, float] = field(default_factory=lambda: {
        "cpu": 90.0,
        "memory": 85.0,
        "disk": 80.0,
    })
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    auto_scale: bool = True
    max_resources: int = 100
    enable_cpu_monitoring: bool = True
    enable_memory_monitoring: bool = True
    enable_disk_monitoring: bool = True
    history_size: int = 1000
    precision: int = 2
    thresholds: Dict[str, float] = field(default_factory=lambda: {
        "cpu_warning": 80.0,
        "memory_warning": 85.0,
        "disk_warning": 90.0,
    })
    cpu_threshold: float = 90.0
    memory_threshold: float = 85.0
    disk_threshold: float = 80.0


__all__ = [
    "TaskConfig",
    "ProcessConfig",
    "MonitorConfig",
    "AlertConfig",
    "ResourceConfig",
    "OptimizationConfig",
    "APIConfig",
    "StrategyUsageConfig",
    "OptimizationReportConfig",
    "ResourceMonitorConfig",
]
