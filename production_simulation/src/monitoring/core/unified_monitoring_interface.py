#!/usr/bin/env python3
"""
统一监控系统接口

定义监控层统一接口，确保所有监控组件实现统一的API。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union, Callable
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class MonitorType(Enum):
    """监控类型"""
    SYSTEM = "system"           # 系统监控
    APPLICATION = "application"  # 应用监控
    BUSINESS = "business"       # 业务监控
    PERFORMANCE = "performance"  # 性能监控
    SECURITY = "security"       # 安全监控
    INFRASTRUCTURE = "infrastructure"  # 基础设施监控


class AlertLevel(Enum):
    """告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertStatus(Enum):
    """告警状态"""
    ACTIVE = "active"       # 激活
    ACKNOWLEDGED = "acknowledged"  # 已确认
    RESOLVED = "resolved"   # 已解决
    SUPPRESSED = "suppressed"  # 已抑制


class MetricType(Enum):
    """指标类型"""
    GAUGE = "gauge"         # 瞬时值
    COUNTER = "counter"     # 计数器
    HISTOGRAM = "histogram"  # 直方图
    SUMMARY = "summary"     # 摘要
    TIMER = "timer"         # 计时器


class HealthStatus(Enum):
    """健康状态"""
    HEALTHY = "healthy"     # 健康
    DEGRADED = "degraded"   # 降级
    UNHEALTHY = "unhealthy"  # 不健康
    UNKNOWN = "unknown"     # 未知


@dataclass
class Metric:
    """
    指标数据类

    表示监控指标的数据。
    """
    name: str
    value: Union[int, float]
    timestamp: datetime
    labels: Dict[str, str]
    metric_type: MetricType
    unit: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Alert:
    """
    告警数据类

    表示监控告警的信息。
    """
    alert_id: str
    title: str
    description: str
    level: AlertLevel
    status: AlertStatus
    source: str
    component: str
    created_at: datetime
    updated_at: datetime
    resolved_at: Optional[datetime] = None
    acknowledged_at: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    tags: List[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class HealthCheck:
    """
    健康检查数据类

    表示组件健康检查的结果。
    """
    component: str
    status: HealthStatus
    timestamp: datetime
    response_time: Optional[float] = None
    message: Optional[str] = None
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


@dataclass
class PerformanceMetrics:
    """
    性能指标数据类

    表示系统性能的综合指标。
    """
    timestamp: datetime
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    response_time: Optional[float] = None
    throughput: Optional[float] = None
    error_rate: Optional[float] = None
    active_connections: Optional[int] = None


@dataclass
class MonitoringConfig:
    """
    监控配置数据类

    定义监控系统的配置参数。
    """
    monitor_type: MonitorType
    enabled: bool = True
    interval: int = 60  # 监控间隔(秒)
    timeout: int = 30   # 超时时间(秒)
    retries: int = 3    # 重试次数
    alert_thresholds: Dict[str, Union[int, float]] = None
    notification_channels: List[str] = None

    def __post_init__(self):
        if self.alert_thresholds is None:
            self.alert_thresholds = {}
        if self.notification_channels is None:
            self.notification_channels = []


class IMonitor(ABC):
    """
    监控器统一接口

    所有监控器实现必须遵循此接口，确保API的一致性。
    """

    @abstractmethod
    def get_monitor_type(self) -> MonitorType:
        """
        获取监控器类型

        Returns:
            监控器类型
        """

    @abstractmethod
    def start_monitoring(self) -> bool:
        """
        启动监控

        Returns:
            是否启动成功
        """

    @abstractmethod
    def stop_monitoring(self) -> bool:
        """
        停止监控

        Returns:
            是否停止成功
        """

    @abstractmethod
    def is_monitoring_active(self) -> bool:
        """
        检查监控是否激活

        Returns:
            是否激活
        """

    @abstractmethod
    def collect_metrics(self) -> List[Metric]:
        """
        收集指标

        Returns:
            指标列表
        """

    @abstractmethod
    def get_current_metrics(self) -> Dict[str, Any]:
        """
        获取当前指标

        Returns:
            当前指标字典
        """

    @abstractmethod
    def get_metric_history(self, metric_name: str, start_time: datetime,
                           end_time: datetime) -> List[Metric]:
        """
        获取指标历史

        Args:
            metric_name: 指标名称
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            指标历史列表
        """

    @abstractmethod
    def set_alert_thresholds(self, thresholds: Dict[str, Union[int, float]]) -> bool:
        """
        设置告警阈值

        Args:
            thresholds: 阈值字典

        Returns:
            是否设置成功
        """

    @abstractmethod
    def get_alert_thresholds(self) -> Dict[str, Union[int, float]]:
        """
        获取告警阈值

        Returns:
            阈值字典
        """

    @abstractmethod
    def check_thresholds(self) -> List[Alert]:
        """
        检查阈值并生成告警

        Returns:
            告警列表
        """

    @abstractmethod
    def get_monitor_config(self) -> MonitoringConfig:
        """
        获取监控配置

        Returns:
            监控配置
        """

    @abstractmethod
    def update_monitor_config(self, config: MonitoringConfig) -> bool:
        """
        更新监控配置

        Args:
            config: 新的监控配置

        Returns:
            是否更新成功
        """


class IAlertManager(ABC):
    """
    告警管理器接口
    """

    @abstractmethod
    def create_alert(self, alert: Alert) -> bool:
        """
        创建告警

        Args:
            alert: 告警对象

        Returns:
            是否创建成功
        """

    @abstractmethod
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        确认告警

        Args:
            alert_id: 告警ID
            acknowledged_by: 确认人

        Returns:
            是否确认成功
        """

    @abstractmethod
    def resolve_alert(self, alert_id: str) -> bool:
        """
        解决告警

        Args:
            alert_id: 告警ID

        Returns:
            是否解决成功
        """

    @abstractmethod
    def suppress_alert(self, alert_id: str, duration: int) -> bool:
        """
        抑制告警

        Args:
            alert_id: 告警ID
            duration: 抑制持续时间(秒)

        Returns:
            是否抑制成功
        """

    @abstractmethod
    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """
        获取告警

        Args:
            alert_id: 告警ID

        Returns:
            告警对象
        """

    @abstractmethod
    def get_alerts(self, status: Optional[AlertStatus] = None,
                   level: Optional[AlertLevel] = None,
                   source: Optional[str] = None) -> List[Alert]:
        """
        获取告警列表

        Args:
            status: 告警状态过滤
            level: 告警级别过滤
            source: 告警来源过滤

        Returns:
            告警列表
        """

    @abstractmethod
    def get_active_alerts(self) -> List[Alert]:
        """
        获取活跃告警

        Returns:
            活跃告警列表
        """

    @abstractmethod
    def get_alert_history(self, start_time: datetime, end_time: datetime) -> List[Alert]:
        """
        获取告警历史

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            告警历史列表
        """

    @abstractmethod
    def escalate_alert(self, alert_id: str, new_level: AlertLevel) -> bool:
        """
        升级告警

        Args:
            alert_id: 告警ID
            new_level: 新告警级别

        Returns:
            是否升级成功
        """

    @abstractmethod
    def send_notifications(self, alert: Alert, channels: List[str]) -> bool:
        """
        发送告警通知

        Args:
            alert: 告警对象
            channels: 通知渠道列表

        Returns:
            是否发送成功
        """


class IHealthChecker(ABC):
    """
    健康检查器接口
    """

    @abstractmethod
    def perform_health_check(self, component: str) -> HealthCheck:
        """
        执行健康检查

        Args:
            component: 组件名称

        Returns:
            健康检查结果
        """

    @abstractmethod
    def get_health_status(self, component: str) -> HealthStatus:
        """
        获取组件健康状态

        Args:
            component: 组件名称

        Returns:
            健康状态
        """

    @abstractmethod
    def get_all_health_status(self) -> Dict[str, HealthStatus]:
        """
        获取所有组件健康状态

        Returns:
            健康状态字典
        """

    @abstractmethod
    def get_health_history(self, component: str, start_time: datetime,
                           end_time: datetime) -> List[HealthCheck]:
        """
        获取健康检查历史

        Args:
            component: 组件名称
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            健康检查历史列表
        """

    @abstractmethod
    def register_health_check(self, component: str, check_function: Callable) -> bool:
        """
        注册健康检查函数

        Args:
            component: 组件名称
            check_function: 健康检查函数

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister_health_check(self, component: str) -> bool:
        """
        注销健康检查函数

        Args:
            component: 组件名称

        Returns:
            是否注销成功
        """

    @abstractmethod
    def set_health_thresholds(self, component: str, thresholds: Dict[str, Any]) -> bool:
        """
        设置健康检查阈值

        Args:
            component: 组件名称
            thresholds: 阈值字典

        Returns:
            是否设置成功
        """


class IPerformanceMonitor(ABC):
    """
    性能监控器接口
    """

    @abstractmethod
    def collect_performance_metrics(self) -> PerformanceMetrics:
        """
        收集性能指标

        Returns:
            性能指标
        """

    @abstractmethod
    def get_performance_history(self, start_time: datetime,
                                end_time: datetime) -> List[PerformanceMetrics]:
        """
        获取性能历史

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            性能指标历史列表
        """

    @abstractmethod
    def analyze_performance_trends(self, metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """
        分析性能趋势

        Args:
            metrics: 性能指标列表

        Returns:
            趋势分析结果
        """

    @abstractmethod
    def detect_performance_anomalies(self, current_metrics: PerformanceMetrics,
                                     historical_metrics: List[PerformanceMetrics]) -> Dict[str, Any]:
        """
        检测性能异常

        Args:
            current_metrics: 当前性能指标
            historical_metrics: 历史性能指标

        Returns:
            异常检测结果
        """

    @abstractmethod
    def get_resource_usage(self) -> Dict[str, float]:
        """
        获取资源使用情况

        Returns:
            资源使用字典
        """

    @abstractmethod
    def monitor_resource_limits(self) -> Dict[str, Any]:
        """
        监控资源限制

        Returns:
            资源限制监控结果
        """

    @abstractmethod
    def generate_performance_report(self, start_time: datetime,
                                    end_time: datetime) -> Dict[str, Any]:
        """
        生成性能报告

        Args:
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            性能报告字典
        """


class IMonitoringDashboard(ABC):
    """
    监控仪表板接口
    """

    @abstractmethod
    def get_dashboard_data(self, dashboard_type: str) -> Dict[str, Any]:
        """
        获取仪表板数据

        Args:
            dashboard_type: 仪表板类型

        Returns:
            仪表板数据字典
        """

    @abstractmethod
    def generate_dashboard_html(self, dashboard_type: str) -> str:
        """
        生成仪表板HTML

        Args:
            dashboard_type: 仪表板类型

        Returns:
            HTML字符串
        """

    @abstractmethod
    def get_available_dashboards(self) -> List[str]:
        """
        获取可用仪表板列表

        Returns:
            仪表板名称列表
        """

    @abstractmethod
    def customize_dashboard(self, dashboard_type: str, config: Dict[str, Any]) -> bool:
        """
        自定义仪表板

        Args:
            dashboard_type: 仪表板类型
            config: 自定义配置

        Returns:
            是否自定义成功
        """

    @abstractmethod
    def export_dashboard_data(self, dashboard_type: str, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        导出仪表板数据

        Args:
            dashboard_type: 仪表板类型
            format: 导出格式

        Returns:
            导出的数据
        """


class IMonitoringSystem(ABC):
    """
    监控系统统一接口
    """

    @abstractmethod
    def initialize_monitoring(self, config: Dict[str, Any]) -> bool:
        """
        初始化监控系统

        Args:
            config: 监控配置

        Returns:
            是否初始化成功
        """

    @abstractmethod
    def register_monitor(self, monitor: IMonitor) -> bool:
        """
        注册监控器

        Args:
            monitor: 监控器实例

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister_monitor(self, monitor_type: MonitorType) -> bool:
        """
        注销监控器

        Args:
            monitor_type: 监控器类型

        Returns:
            是否注销成功
        """

    @abstractmethod
    def get_monitor(self, monitor_type: MonitorType) -> Optional[IMonitor]:
        """
        获取监控器

        Args:
            monitor_type: 监控器类型

        Returns:
            监控器实例
        """

    @abstractmethod
    def get_all_monitors(self) -> Dict[str, IMonitor]:
        """
        获取所有监控器

        Returns:
            监控器字典
        """

    @abstractmethod
    def start_all_monitors(self) -> bool:
        """
        启动所有监控器

        Returns:
            是否全部启动成功
        """

    @abstractmethod
    def stop_all_monitors(self) -> bool:
        """
        停止所有监控器

        Returns:
            是否全部停止成功
        """

    @abstractmethod
    def get_system_status(self) -> Dict[str, Any]:
        """
        获取系统整体状态

        Returns:
            系统状态字典
        """

    @abstractmethod
    def get_system_health_score(self) -> float:
        """
        获取系统健康评分

        Returns:
            健康评分(0-100)
        """

    @abstractmethod
    def generate_system_report(self) -> Dict[str, Any]:
        """
        生成系统综合报告

        Returns:
            系统报告字典
        """

    @abstractmethod
    def set_global_alert_rules(self, rules: Dict[str, Any]) -> bool:
        """
        设置全局告警规则

        Args:
            rules: 告警规则字典

        Returns:
            是否设置成功
        """

    @abstractmethod
    def get_global_alert_rules(self) -> Dict[str, Any]:
        """
        获取全局告警规则

        Returns:
            告警规则字典
        """

    @abstractmethod
    def enable_maintenance_mode(self, duration: int) -> bool:
        """
        启用维护模式

        Args:
            duration: 维护持续时间(秒)

        Returns:
            是否启用成功
        """

    @abstractmethod
    def disable_maintenance_mode(self) -> bool:
        """
        禁用维护模式

        Returns:
            是否禁用成功
        """

    @abstractmethod
    def is_maintenance_mode_active(self) -> bool:
        """
        检查维护模式是否激活

        Returns:
            是否激活
        """
