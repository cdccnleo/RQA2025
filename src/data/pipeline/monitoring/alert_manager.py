"""
告警管理模块

提供告警生成、分级、通知和抑制功能
"""

import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union


class AlertSeverity(Enum):
    """告警严重级别枚举"""
    DEBUG = "debug"        # 调试信息
    INFO = "info"          # 一般信息
    WARNING = "warning"    # 警告
    ERROR = "error"        # 错误
    CRITICAL = "critical"  # 严重错误


class AlertStatus(Enum):
    """告警状态枚举"""
    ACTIVE = "active"          # 活跃
    ACKNOWLEDGED = "acknowledged"  # 已确认
    RESOLVED = "resolved"      # 已解决
    SUPPRESSED = "suppressed"  # 已抑制


@dataclass
class Alert:
    """
    告警数据类
    
    Attributes:
        alert_id: 告警ID
        title: 告警标题
        message: 告警消息
        severity: 严重级别
        status: 状态
        source: 告警来源
        metric_name: 相关指标名称
        metric_value: 指标值
        threshold: 阈值
        timestamp: 创建时间
        acknowledged_by: 确认人
        acknowledged_at: 确认时间
        resolved_at: 解决时间
        metadata: 额外元数据
    """
    alert_id: str
    title: str
    message: str
    severity: AlertSeverity
    source: str
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    status: AlertStatus = AlertStatus.ACTIVE
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "alert_id": self.alert_id,
            "title": self.title,
            "message": self.message,
            "severity": self.severity.value,
            "status": self.status.value,
            "source": self.source,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
            "threshold": self.threshold,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "metadata": self.metadata
        }
    
    @property
    def is_active(self) -> bool:
        """检查是否活跃"""
        return self.status == AlertStatus.ACTIVE
    
    @property
    def duration_seconds(self) -> float:
        """获取告警持续时间"""
        end_time = self.resolved_at or datetime.now()
        return (end_time - self.timestamp).total_seconds()


@dataclass
class AlertRule:
    """
    告警规则
    
    Attributes:
        rule_id: 规则ID
        name: 规则名称
        metric_name: 监控指标名称
        operator: 比较操作符
        threshold: 阈值
        severity: 告警级别
        duration_minutes: 持续时间（分钟）
        enabled: 是否启用
        description: 描述
    """
    rule_id: str
    name: str
    metric_name: str
    operator: str  # greater_than, less_than, equal, not_equal, decrease_by, increase_by
    threshold: float
    severity: AlertSeverity
    duration_minutes: int = 5
    enabled: bool = True
    description: str = ""
    
    def evaluate(self, metric_value: float, baseline_value: Optional[float] = None) -> bool:
        """
        评估规则
        
        Args:
            metric_value: 当前指标值
            baseline_value: 基线值（用于变化率计算）
            
        Returns:
            是否触发告警
        """
        if not self.enabled:
            return False
        
        if self.operator == "greater_than":
            return metric_value > self.threshold
        elif self.operator == "less_than":
            return metric_value < self.threshold
        elif self.operator == "equal":
            return abs(metric_value - self.threshold) < 1e-6
        elif self.operator == "not_equal":
            return abs(metric_value - self.threshold) >= 1e-6
        elif self.operator == "decrease_by":
            if baseline_value is None:
                return False
            decrease = (baseline_value - metric_value) / baseline_value if baseline_value != 0 else 0
            return decrease > self.threshold
        elif self.operator == "increase_by":
            if baseline_value is None:
                return False
            increase = (metric_value - baseline_value) / baseline_value if baseline_value != 0 else 0
            return increase > self.threshold
        
        return False


class AlertManager:
    """
    告警管理器
    
    管理告警规则、告警生成和通知
    """
    
    def __init__(self):
        """初始化告警管理器"""
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: Dict[str, Alert] = {}
        self._alert_history: deque = deque(maxlen=1000)
        self._suppressed_metrics: Set[str] = set()
        self._notification_handlers: List[Callable[[Alert], None]] = []
        self._metric_baselines: Dict[str, float] = {}
        
        self.logger = logging.getLogger("monitoring.alert_manager")
    
    def register_rule(self, rule: AlertRule) -> None:
        """
        注册告警规则
        
        Args:
            rule: 告警规则
        """
        self._rules[rule.rule_id] = rule
        self.logger.info(f"注册告警规则: {rule.name}")
    
    def unregister_rule(self, rule_id: str) -> bool:
        """
        注销告警规则
        
        Args:
            rule_id: 规则ID
            
        Returns:
            是否成功
        """
        if rule_id in self._rules:
            del self._rules[rule_id]
            self.logger.info(f"注销告警规则: {rule_id}")
            return True
        return False
    
    def set_baseline(self, metric_name: str, value: float) -> None:
        """
        设置指标基线
        
        Args:
            metric_name: 指标名称
            value: 基线值
        """
        self._metric_baselines[metric_name] = value
    
    def evaluate_metrics(self, metrics: Dict[str, float]) -> List[Alert]:
        """
        评估指标并生成告警
        
        Args:
            metrics: 指标字典
            
        Returns:
            生成的告警列表
        """
        triggered_alerts = []
        
        for rule in self._rules.values():
            if not rule.enabled:
                continue
            
            if rule.metric_name not in metrics:
                continue
            
            # 检查是否被抑制
            if rule.metric_name in self._suppressed_metrics:
                continue
            
            metric_value = metrics[rule.metric_name]
            baseline_value = self._metric_baselines.get(rule.metric_name)
            
            # 评估规则
            if rule.evaluate(metric_value, baseline_value):
                # 检查是否已有相同规则的活跃告警
                existing_alert = self._find_active_alert(rule.rule_id)
                
                if existing_alert is None:
                    # 创建新告警
                    alert = self._create_alert(rule, metric_value)
                    triggered_alerts.append(alert)
                    self.logger.warning(
                        f"触发告警: {alert.title} [{alert.severity.value}]"
                    )
        
        return triggered_alerts
    
    def _find_active_alert(self, rule_id: str) -> Optional[Alert]:
        """查找活跃告警"""
        for alert in self._alerts.values():
            if alert.is_active and alert.metadata.get("rule_id") == rule_id:
                return alert
        return None
    
    def _create_alert(self, rule: AlertRule, metric_value: float) -> Alert:
        """创建告警"""
        import uuid
        
        alert_id = f"alert_{uuid.uuid4().hex[:8]}"
        
        alert = Alert(
            alert_id=alert_id,
            title=f"{rule.name} 告警",
            message=rule.description or f"指标 {rule.metric_name} 触发告警",
            severity=rule.severity,
            source="alert_manager",
            metric_name=rule.metric_name,
            metric_value=metric_value,
            threshold=rule.threshold,
            metadata={
                "rule_id": rule.rule_id,
                "operator": rule.operator,
                "duration_minutes": rule.duration_minutes
            }
        )
        
        self._alerts[alert_id] = alert
        self._alert_history.append(alert)
        
        # 发送通知
        self._notify(alert)
        
        return alert
    
    def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        确认告警
        
        Args:
            alert_id: 告警ID
            acknowledged_by: 确认人
            
        Returns:
            是否成功
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            return False
        
        alert.status = AlertStatus.ACKNOWLEDGED
        alert.acknowledged_by = acknowledged_by
        alert.acknowledged_at = datetime.now()
        
        self.logger.info(f"告警已确认: {alert_id} by {acknowledged_by}")
        return True
    
    def resolve_alert(self, alert_id: str) -> bool:
        """
        解决告警
        
        Args:
            alert_id: 告警ID
            
        Returns:
            是否成功
        """
        alert = self._alerts.get(alert_id)
        if alert is None:
            return False
        
        alert.status = AlertStatus.RESOLVED
        alert.resolved_at = datetime.now()
        
        self.logger.info(f"告警已解决: {alert_id}")
        return True
    
    def suppress_metric(self, metric_name: str, duration_minutes: int = 60) -> None:
        """
        抑制指标告警
        
        Args:
            metric_name: 指标名称
            duration_minutes: 抑制持续时间
        """
        self._suppressed_metrics.add(metric_name)
        self.logger.info(f"抑制指标告警: {metric_name} for {duration_minutes}分钟")
        
        # 计划恢复
        import threading
        def restore():
            self._suppressed_metrics.discard(metric_name)
            self.logger.info(f"恢复指标告警: {metric_name}")
        
        threading.Timer(duration_minutes * 60, restore).start()
    
    def register_notification_handler(self, handler: Callable[[Alert], None]) -> None:
        """
        注册通知处理器
        
        Args:
            handler: 处理函数
        """
        self._notification_handlers.append(handler)
    
    def _notify(self, alert: Alert) -> None:
        """发送通知"""
        for handler in self._notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"通知处理器异常: {e}")
    
    def get_active_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None
    ) -> List[Alert]:
        """
        获取活跃告警
        
        Args:
            severity: 严重级别过滤
            source: 来源过滤
            
        Returns:
            告警列表
        """
        alerts = [a for a in self._alerts.values() if a.is_active]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if source:
            alerts = [a for a in alerts if a.source == source]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_alert_history(
        self,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> List[Alert]:
        """
        获取告警历史
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            
        Returns:
            告警列表
        """
        alerts = list(self._alert_history)
        
        if start_time:
            alerts = [a for a in alerts if a.timestamp >= start_time]
        if end_time:
            alerts = [a for a in alerts if a.timestamp <= end_time]
        
        return sorted(alerts, key=lambda a: a.timestamp, reverse=True)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            统计信息字典
        """
        active_alerts = self.get_active_alerts()
        
        return {
            "total_rules": len(self._rules),
            "enabled_rules": sum(1 for r in self._rules.values() if r.enabled),
            "active_alerts": len(active_alerts),
            "total_alerts_history": len(self._alert_history),
            "alerts_by_severity": {
                s.value: len([a for a in active_alerts if a.severity == s])
                for s in AlertSeverity
            },
            "suppressed_metrics": list(self._suppressed_metrics)
        }
    
    def clear_resolved_alerts(self, max_age_hours: int = 24) -> int:
        """
        清理已解决的告警
        
        Args:
            max_age_hours: 最大保留时间
            
        Returns:
            清理数量
        """
        cutoff = datetime.now() - timedelta(hours=max_age_hours)
        to_remove = [
            alert_id for alert_id, alert in self._alerts.items()
            if alert.status == AlertStatus.RESOLVED and alert.resolved_at < cutoff
        ]
        
        for alert_id in to_remove:
            del self._alerts[alert_id]
        
        self.logger.info(f"清理 {len(to_remove)} 个已解决告警")
        return len(to_remove)


def create_default_alert_rules() -> List[AlertRule]:
    """
    创建默认告警规则
    
    Returns:
        告警规则列表
    """
    return [
        AlertRule(
            rule_id="accuracy_drop",
            name="准确率下降",
            metric_name="accuracy",
            operator="decrease_by",
            threshold=0.1,  # 下降10%
            severity=AlertSeverity.CRITICAL,
            duration_minutes=5,
            description="模型准确率下降超过10%"
        ),
        AlertRule(
            rule_id="high_drawdown",
            name="最大回撤过高",
            metric_name="max_drawdown",
            operator="greater_than",
            threshold=0.15,
            severity=AlertSeverity.CRITICAL,
            duration_minutes=1,
            description="最大回撤超过15%"
        ),
        AlertRule(
            rule_id="low_sharpe",
            name="夏普比率过低",
            metric_name="sharpe_ratio",
            operator="less_than",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
            duration_minutes=30,
            description="夏普比率低于0.5"
        ),
        AlertRule(
            rule_id="high_latency",
            name="推理延迟过高",
            metric_name="p95_latency_ms",
            operator="greater_than",
            threshold=200,
            severity=AlertSeverity.WARNING,
            duration_minutes=5,
            description="P95推理延迟超过200ms"
        ),
        AlertRule(
            rule_id="high_error_rate",
            name="错误率过高",
            metric_name="error_rate",
            operator="greater_than",
            threshold=0.05,
            severity=AlertSeverity.ERROR,
            duration_minutes=3,
            description="错误率超过5%"
        ),
        AlertRule(
            rule_id="data_drift",
            name="数据漂移",
            metric_name="drift_score",
            operator="greater_than",
            threshold=0.5,
            severity=AlertSeverity.WARNING,
            duration_minutes=10,
            description="检测到数据漂移"
        )
    ]
