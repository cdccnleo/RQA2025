#!/usr/bin/env python3
"""
统一数据质量监控接口

定义数据管理层数据质量监控的统一接口，确保所有质量监控器实现统一的API。
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class QualityCheckType(Enum):
    """质量检查类型"""
    COMPLETENESS = "completeness"  # 完整性检查
    ACCURACY = "accuracy"         # 准确性检查
    CONSISTENCY = "consistency"   # 一致性检查
    TIMELINESS = "timeliness"     # 时效性检查
    VALIDITY = "validity"         # 有效性检查
    UNIQUENESS = "uniqueness"     # 唯一性检查


class QualityAlertLevel(Enum):
    """质量告警级别"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class QualityMetric(Enum):
    """质量指标"""
    COMPLETENESS_RATIO = "completeness_ratio"  # 完整性比率
    ACCURACY_RATIO = "accuracy_ratio"          # 准确性比率
    CONSISTENCY_RATIO = "consistency_ratio"    # 一致性比率
    TIMELINESS_RATIO = "timeliness_ratio"      # 时效性比率
    VALIDITY_RATIO = "validity_ratio"          # 有效性比率
    UNIQUENESS_RATIO = "uniqueness_ratio"      # 唯一性比率
    DATA_FRESHNESS = "data_freshness"          # 数据新鲜度
    ERROR_RATE = "error_rate"                  # 错误率


@dataclass
class QualityCheckResult:
    """
    质量检查结果

    表示单次质量检查的结果。
    """
    check_type: QualityCheckType
    metric: QualityMetric
    value: Union[float, int, bool]
    threshold: Union[float, int]
    passed: bool
    details: Dict[str, Any]
    timestamp: datetime
    duration: float  # 检查耗时(秒)


@dataclass
class QualityAlert:
    """
    质量告警

    表示质量问题告警。
    """
    alert_id: str
    level: QualityAlertLevel
    title: str
    description: str
    check_results: List[QualityCheckResult]
    affected_data: Dict[str, Any]
    timestamp: datetime
    resolved: bool = False
    resolution_time: Optional[datetime] = None


@dataclass
class QualityReport:
    """
    质量报告

    表示综合质量评估报告。
    """
    report_id: str
    data_source: str
    time_range: Dict[str, datetime]
    check_results: List[QualityCheckResult]
    alerts: List[QualityAlert]
    overall_score: float
    summary: Dict[str, Any]
    recommendations: List[str]
    generated_at: datetime


class IDataQualityChecker(ABC):
    """
    数据质量检查器接口
    """

    @abstractmethod
    def check_completeness(self, data: Any, config: Dict[str, Any]) -> QualityCheckResult:
        """
        检查数据完整性

        Args:
            data: 要检查的数据
            config: 检查配置

        Returns:
            质量检查结果
        """

    @abstractmethod
    def check_accuracy(self, data: Any, reference: Any, config: Dict[str, Any]) -> QualityCheckResult:
        """
        检查数据准确性

        Args:
            data: 要检查的数据
            reference: 参考数据
            config: 检查配置

        Returns:
            质量检查结果
        """

    @abstractmethod
    def check_consistency(self, data: Any, rules: Dict[str, Any]) -> QualityCheckResult:
        """
        检查数据一致性

        Args:
            data: 要检查的数据
            rules: 一致性规则

        Returns:
            质量检查结果
        """

    @abstractmethod
    def check_timeliness(self, data: Any, max_age: int) -> QualityCheckResult:
        """
        检查数据时效性

        Args:
            data: 要检查的数据
            max_age: 最大年龄(秒)

        Returns:
            质量检查结果
        """

    @abstractmethod
    def check_validity(self, data: Any, schema: Dict[str, Any]) -> QualityCheckResult:
        """
        检查数据有效性

        Args:
            data: 要检查的数据
            schema: 数据模式

        Returns:
            质量检查结果
        """

    @abstractmethod
    def check_uniqueness(self, data: Any, key_fields: List[str]) -> QualityCheckResult:
        """
        检查数据唯一性

        Args:
            data: 要检查的数据
            key_fields: 关键字段列表

        Returns:
            质量检查结果
        """

    @abstractmethod
    def get_supported_checks(self) -> List[QualityCheckType]:
        """
        获取支持的质量检查类型

        Returns:
            支持的检查类型列表
        """


class IDataQualityMonitor(ABC):
    """
    数据质量监控器接口
    """

    @abstractmethod
    def monitor_data_source(self, source_id: str, config: Dict[str, Any]) -> bool:
        """
        监控数据源

        Args:
            source_id: 数据源ID
            config: 监控配置

        Returns:
            是否开始监控成功
        """

    @abstractmethod
    def stop_monitoring(self, source_id: str) -> bool:
        """
        停止监控数据源

        Args:
            source_id: 数据源ID

        Returns:
            是否停止成功
        """

    @abstractmethod
    def check_data_quality(self, data: Any, source_id: str) -> List[QualityCheckResult]:
        """
        检查数据质量

        Args:
            data: 要检查的数据
            source_id: 数据源ID

        Returns:
            质量检查结果列表
        """

    @abstractmethod
    def get_quality_metrics(self, source_id: str) -> Dict[str, Any]:
        """
        获取质量指标

        Args:
            source_id: 数据源ID

        Returns:
            质量指标字典
        """

    @abstractmethod
    def get_alerts(self, source_id: str, level: Optional[QualityAlertLevel] = None,
                   resolved: bool = False) -> List[QualityAlert]:
        """
        获取告警列表

        Args:
            source_id: 数据源ID
            level: 告警级别过滤
            resolved: 是否包含已解决的告警

        Returns:
            告警列表
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
    def generate_report(self, source_id: str, time_range: Dict[str, datetime]) -> QualityReport:
        """
        生成质量报告

        Args:
            source_id: 数据源ID
            time_range: 时间范围

        Returns:
            质量报告
        """

    @abstractmethod
    def set_thresholds(self, source_id: str, thresholds: Dict[str, Union[float, int]]) -> bool:
        """
        设置质量阈值

        Args:
            source_id: 数据源ID
            thresholds: 阈值字典

        Returns:
            是否设置成功
        """

    @abstractmethod
    def get_thresholds(self, source_id: str) -> Dict[str, Union[float, int]]:
        """
        获取质量阈值

        Args:
            source_id: 数据源ID

        Returns:
            阈值字典
        """


class IDataQualityReporter(ABC):
    """
    数据质量报告器接口
    """

    @abstractmethod
    def generate_quality_report(self, report: QualityReport, format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        生成质量报告

        Args:
            report: 质量报告对象
            format: 输出格式 (json, html, pdf, etc.)

        Returns:
            格式化的报告内容
        """

    @abstractmethod
    def send_alert_notification(self, alert: QualityAlert, channels: List[str]) -> bool:
        """
        发送告警通知

        Args:
            alert: 告警对象
            channels: 通知渠道列表 (email, slack, webhook, etc.)

        Returns:
            是否发送成功
        """

    @abstractmethod
    def export_metrics(self, metrics: Dict[str, Any], format: str = "json") -> Union[str, Dict[str, Any]]:
        """
        导出质量指标

        Args:
            metrics: 质量指标字典
            format: 导出格式

        Returns:
            导出的指标数据
        """

    @abstractmethod
    def generate_dashboard_data(self, source_id: str) -> Dict[str, Any]:
        """
        生成仪表板数据

        Args:
            source_id: 数据源ID

        Returns:
            仪表板数据字典
        """


class IDataQualityManager(ABC):
    """
    数据质量管理器接口
    """

    @abstractmethod
    def register_monitor(self, monitor: IDataQualityMonitor) -> bool:
        """
        注册质量监控器

        Args:
            monitor: 质量监控器实例

        Returns:
            是否注册成功
        """

    @abstractmethod
    def unregister_monitor(self, monitor_id: str) -> bool:
        """
        注销质量监控器

        Args:
            monitor_id: 监控器ID

        Returns:
            是否注销成功
        """

    @abstractmethod
    def get_monitor(self, source_id: str) -> Optional[IDataQualityMonitor]:
        """
        获取质量监控器

        Args:
            source_id: 数据源ID

        Returns:
            质量监控器实例
        """

    @abstractmethod
    def start_monitoring(self, source_id: str) -> bool:
        """
        开始监控数据源

        Args:
            source_id: 数据源ID

        Returns:
            是否开始成功
        """

    @abstractmethod
    def stop_monitoring(self, source_id: str) -> bool:
        """
        停止监控数据源

        Args:
            source_id: 数据源ID

        Returns:
            是否停止成功
        """

    @abstractmethod
    def check_all_sources(self) -> Dict[str, List[QualityCheckResult]]:
        """
        检查所有数据源的质量

        Returns:
            数据源质量检查结果字典
        """

    @abstractmethod
    def get_system_health(self) -> Dict[str, Any]:
        """
        获取系统质量健康状态

        Returns:
            系统健康状态字典
        """

    @abstractmethod
    def configure_monitoring(self, config: Dict[str, Any]) -> bool:
        """
        配置监控系统

        Args:
            config: 监控配置字典

        Returns:
            是否配置成功
        """

    @abstractmethod
    def get_monitoring_config(self) -> Dict[str, Any]:
        """
        获取监控配置

        Returns:
            监控配置字典
        """

    @abstractmethod
    def optimize_monitoring(self) -> Dict[str, Any]:
        """
        优化监控性能

        Returns:
            优化结果字典
        """


class IDataQualityRule(ABC):
    """
    数据质量规则接口
    """

    @abstractmethod
    def get_rule_name(self) -> str:
        """
        获取规则名称

        Returns:
            规则名称
        """

    @abstractmethod
    def get_rule_type(self) -> QualityCheckType:
        """
        获取规则类型

        Returns:
            规则类型
        """

    @abstractmethod
    def validate(self, data: Any, context: Dict[str, Any]) -> QualityCheckResult:
        """
        执行规则验证

        Args:
            data: 要验证的数据
            context: 验证上下文

        Returns:
            验证结果
        """

    @abstractmethod
    def get_description(self) -> str:
        """
        获取规则描述

        Returns:
            规则描述
        """

    @abstractmethod
    def get_severity(self) -> QualityAlertLevel:
        """
        获取规则严重程度

        Returns:
            严重程度
        """

    @abstractmethod
    def is_enabled(self) -> bool:
        """
        检查规则是否启用

        Returns:
            是否启用
        """

    @abstractmethod
    def set_enabled(self, enabled: bool) -> None:
        """
        设置规则启用状态

        Args:
            enabled: 是否启用
        """
