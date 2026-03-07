#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
策略监控服务接口定义
Strategy Monitoring Service Interfaces

定义统一的策略监控接口，支持实时监控、性能跟踪和告警管理。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AlertLevel(Enum):

    """告警级别枚举"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class MetricType(Enum):

    """指标类型枚举"""
    PERFORMANCE = "performance"
    RISK = "risk"
    SYSTEM = "system"
    CUSTOM = "custom"


@dataclass
class AlertRule:

    """告警规则"""
    rule_id: str
    strategy_id: str
    metric_name: str
    condition: str  # '>', '<', '>=', '<=', '==', '!='
    threshold: float
    level: AlertLevel
    enabled: bool = True
    description: str = ""
    cooldown_minutes: int = 60  # 冷却时间（分钟）
    created_at: datetime = None

    def __post_init__(self):

        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class Alert:

    """告警信息"""
    alert_id: str
    strategy_id: str
    rule_id: str
    level: AlertLevel
    message: str
    metric_value: float
    threshold: float
    timestamp: datetime = None
    resolved: bool = False
    resolved_at: Optional[datetime] = None

    def __post_init__(self):

        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MetricData:

    """指标数据"""
    metric_name: str
    value: float
    timestamp: datetime
    strategy_id: str
    metric_type: MetricType
    metadata: Dict[str, Any] = None

    def __post_init__(self):

        if self.metadata is None:
            self.metadata = {}


@dataclass
class PerformanceReport:

    """性能报告"""
    strategy_id: str
    report_period: str  # 'daily', 'weekly', 'monthly'
    start_date: datetime
    end_date: datetime
    metrics: Dict[str, float]
    alerts: List[Alert]
    recommendations: List[str]
    generated_at: datetime = None

    def __post_init__(self):

        if self.generated_at is None:
            self.generated_at = datetime.now()


class IMonitoringService(ABC):

    """
    监控服务接口
    Monitoring Service Interface

    定义策略监控的核心功能接口。
    """

    @abstractmethod
    def start_monitoring(self, strategy_id: str) -> bool:
        """
        开始监控策略

        Args:
            strategy_id: 策略ID

        Returns:
            bool: 启动是否成功
        """

    @abstractmethod
    def stop_monitoring(self, strategy_id: str) -> bool:
        """
        停止监控策略

        Args:
            strategy_id: 策略ID

        Returns:
            bool: 停止是否成功
        """

    @abstractmethod
    def get_current_metrics(self, strategy_id: str,


                            metric_types: Optional[List[MetricType]] = None) -> Dict[str, MetricData]:
        """
        获取当前指标

        Args:
            strategy_id: 策略ID
            metric_types: 指标类型过滤器

        Returns:
            Dict[str, MetricData]: 指标数据字典
        """

    @abstractmethod
    def get_metric_history(self, strategy_id: str, metric_name: str,


                           start_time: Optional[datetime] = None,
                           end_time: Optional[datetime] = None) -> List[MetricData]:
        """
        获取指标历史

        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            start_time: 开始时间
            end_time: 结束时间

        Returns:
            List[MetricData]: 指标历史数据列表
        """


class IAlertService(ABC):

    """
    告警服务接口
    Alert Service Interface

    定义告警管理功能接口。
    """

    @abstractmethod
    def create_alert_rule(self, rule: AlertRule) -> bool:
        """
        创建告警规则

        Args:
            rule: 告警规则

        Returns:
            bool: 创建是否成功
        """

    @abstractmethod
    def update_alert_rule(self, rule_id: str, updates: Dict[str, Any]) -> bool:
        """
        更新告警规则

        Args:
            rule_id: 规则ID
            updates: 更新内容

        Returns:
            bool: 更新是否成功
        """

    @abstractmethod
    def delete_alert_rule(self, rule_id: str) -> bool:
        """
        删除告警规则

        Args:
            rule_id: 规则ID

        Returns:
            bool: 删除是否成功
        """

    @abstractmethod
    def get_alert_rules(self, strategy_id: str) -> List[AlertRule]:
        """
        获取告警规则

        Args:
            strategy_id: 策略ID

        Returns:
            List[AlertRule]: 告警规则列表
        """

    @abstractmethod
    def get_active_alerts(self, strategy_id: Optional[str] = None) -> List[Alert]:
        """
        获取活跃告警

        Args:
            strategy_id: 策略ID过滤器

        Returns:
            List[Alert]: 活跃告警列表
        """

    @abstractmethod
    def resolve_alert(self, alert_id: str, resolution_notes: Optional[str] = None) -> bool:
        """
        解决告警

        Args:
            alert_id: 告警ID
            resolution_notes: 解决说明

        Returns:
            bool: 解决是否成功
        """


class IPerformanceMonitor(ABC):

    """
    性能监控接口
    Performance Monitor Interface

    定义性能监控功能接口。
    """

    @abstractmethod
    def record_metric(self, metric_data: MetricData) -> bool:
        """
        记录指标

        Args:
            metric_data: 指标数据

        Returns:
            bool: 记录是否成功
        """

    @abstractmethod
    def get_performance_summary(self, strategy_id: str, period: str = "daily") -> Dict[str, Any]:
        """
        获取性能摘要

        Args:
            strategy_id: 策略ID
            period: 时间周期 ('daily', 'weekly', 'monthly')

        Returns:
            Dict[str, Any]: 性能摘要
        """

    @abstractmethod
    def calculate_rolling_metrics(self, strategy_id: str, window: int = 20) -> Dict[str, float]:
        """
        计算滚动指标

        Args:
            strategy_id: 策略ID
            window: 滚动窗口大小

        Returns:
            Dict[str, float]: 滚动指标字典
        """

    @abstractmethod
    def detect_performance_anomalies(self, strategy_id: str,


                                     lookback_period: int = 30) -> List[Dict[str, Any]]:
        """
        检测性能异常

        Args:
            strategy_id: 策略ID
            lookback_period: 回溯周期（天）

        Returns:
            List[Dict[str, Any]]: 异常检测结果
        """


class IRiskMonitor(ABC):

    """
    风险监控接口
    Risk Monitor Interface

    定义风险监控功能接口。
    """

    @abstractmethod
    def monitor_risk_metrics(self, strategy_id: str) -> Dict[str, float]:
        """
        监控风险指标

        Args:
            strategy_id: 策略ID

        Returns:
            Dict[str, float]: 风险指标字典
        """

    @abstractmethod
    def check_risk_limits(self, strategy_id: str) -> List[Dict[str, Any]]:
        """
        检查风险限额

        Args:
            strategy_id: 策略ID

        Returns:
            List[Dict[str, Any]]: 风险限额检查结果
        """

    @abstractmethod
    def calculate_var(self, strategy_id: str, confidence_level: float = 0.95,


                      time_horizon: int = 1) -> float:
        """
        计算VaR

        Args:
            strategy_id: 策略ID
            confidence_level: 置信水平
            time_horizon: 时间跨度（天）

        Returns:
            float: VaR值
        """

    @abstractmethod
    def calculate_expected_shortfall(self, strategy_id: str,


                                     confidence_level: float = 0.95) -> float:
        """
        计算期望损失

        Args:
            strategy_id: 策略ID
            confidence_level: 置信水平

        Returns:
            float: 期望损失值
        """


class IMonitoringPersistence(ABC):

    """
    监控持久化接口
    Monitoring Persistence Interface

    处理监控数据和告警的持久化存储。
    """

    @abstractmethod
    def save_metric_data(self, metric_data: MetricData) -> bool:
        """
        保存指标数据

        Args:
            metric_data: 指标数据

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def save_alert(self, alert: Alert) -> bool:
        """
        保存告警

        Args:
            alert: 告警信息

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def save_performance_report(self, report: PerformanceReport) -> bool:
        """
        保存性能报告

        Args:
            report: 性能报告

        Returns:
            bool: 保存是否成功
        """

    @abstractmethod
    def get_historical_metrics(self, strategy_id: str, metric_name: str,


                               start_date: datetime, end_date: datetime) -> List[MetricData]:
        """
        获取历史指标

        Args:
            strategy_id: 策略ID
            metric_name: 指标名称
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            List[MetricData]: 历史指标数据列表
        """

    @abstractmethod
    def get_historical_alerts(self, strategy_id: str,


                              start_date: datetime, end_date: datetime) -> List[Alert]:
        """
        获取历史告警

        Args:
            strategy_id: 策略ID
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            List[Alert]: 历史告警列表
        """


# 导出所有接口
__all__ = [
    'AlertLevel',
    'MetricType',
    'AlertRule',
    'Alert',
    'MetricData',
    'PerformanceReport',
    'IMonitoringService',
    'IAlertService',
    'IPerformanceMonitor',
    'IRiskMonitor',
    'IMonitoringPersistence'
]
