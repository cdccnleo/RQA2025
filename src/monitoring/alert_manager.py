"""
告警管理器模块

负责告警规则管理、告警触发和通知发送
"""

from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import threading
import logging


class AlertSeverity(Enum):
    """告警级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AlertOperator(Enum):
    """告警操作符"""
    GREATER_THAN = "greater_than"
    LESS_THAN = "less_than"
    EQUAL = "equal"
    NOT_EQUAL = "not_equal"
    INCREASE = "increase"  # 相比历史值增加
    DECREASE = "decrease"  # 相比历史值减少


@dataclass
class AlertRule:
    """告警规则"""
    name: str
    metric: str
    operator: AlertOperator
    threshold: float
    severity: AlertSeverity
    duration_minutes: int = 5
    enabled: bool = True
    labels: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "metric": self.metric,
            "operator": self.operator.value,
            "threshold": self.threshold,
            "severity": self.severity.value,
            "duration_minutes": self.duration_minutes,
            "enabled": self.enabled,
            "labels": self.labels
        }


@dataclass
class Alert:
    """告警实例"""
    rule_name: str
    severity: AlertSeverity
    metric: str
    value: float
    threshold: float
    operator: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "severity": self.severity.value,
            "metric": self.metric,
            "value": self.value,
            "threshold": self.threshold,
            "operator": self.operator,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "acknowledged": self.acknowledged,
            "resolved": self.resolved,
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None
        }


class AlertManager:
    """
    告警管理器
    
    功能：
    - 告警规则管理
    - 告警触发判断
    - 告警通知发送
    - 告警历史记录
    """
    
    def __init__(self):
        self.logger = logging.getLogger("monitoring.alert_manager")
        self._rules: Dict[str, AlertRule] = {}
        self._alerts: List[Alert] = []
        self._handlers: List[Callable[[Alert], None]] = []
        self._lock = threading.Lock()
        self._metric_history: Dict[str, List[tuple]] = {}  # metric -> [(timestamp, value)]
    
    def add_rule(self, rule: AlertRule) -> None:
        """添加告警规则"""
        with self._lock:
            self._rules[rule.name] = rule
        self.logger.info(f"添加告警规则: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """移除告警规则"""
        with self._lock:
            if rule_name in self._rules:
                del self._rules[rule_name]
                return True
        return False
    
    def get_rules(self) -> List[AlertRule]:
        """获取所有告警规则"""
        with self._lock:
            return list(self._rules.values())
    
    def register_handler(self, handler: Callable[[Alert], None]) -> None:
        """注册告警处理器"""
        self._handlers.append(handler)
    
    def evaluate_metric(self, metric_name: str, value: float, labels: Optional[Dict[str, str]] = None) -> List[Alert]:
        """
        评估指标是否触发告警
        
        Args:
            metric_name: 指标名称
            value: 指标值
            labels: 标签
            
        Returns:
            触发的告警列表
        """
        triggered_alerts = []
        
        # 保存历史数据
        if metric_name not in self._metric_history:
            self._metric_history[metric_name] = []
        self._metric_history[metric_name].append((datetime.now(), value))
        # 限制历史数据量
        if len(self._metric_history[metric_name]) > 1000:
            self._metric_history[metric_name] = self._metric_history[metric_name][-500:]
        
        with self._lock:
            for rule in self._rules.values():
                if not rule.enabled:
                    continue
                
                if rule.metric != metric_name:
                    continue
                
                # 检查标签匹配
                if rule.labels:
                    if not labels or not all(labels.get(k) == v for k, v in rule.labels.items()):
                        continue
                
                # 评估规则
                if self._evaluate_rule(rule, value, metric_name):
                    alert = Alert(
                        rule_name=rule.name,
                        severity=rule.severity,
                        metric=metric_name,
                        value=value,
                        threshold=rule.threshold,
                        operator=rule.operator.value,
                        message=self._generate_alert_message(rule, value)
                    )
                    triggered_alerts.append(alert)
                    
                    # 保存告警
                    self._alerts.append(alert)
                    
                    # 发送通知
                    self._notify_handlers(alert)
        
        return triggered_alerts
    
    def _evaluate_rule(self, rule: AlertRule, value: float, metric_name: str) -> bool:
        """评估单个规则"""
        op = rule.operator
        
        if op == AlertOperator.GREATER_THAN:
            return value > rule.threshold
        elif op == AlertOperator.LESS_THAN:
            return value < rule.threshold
        elif op == AlertOperator.EQUAL:
            return abs(value - rule.threshold) < 1e-6
        elif op == AlertOperator.NOT_EQUAL:
            return abs(value - rule.threshold) >= 1e-6
        elif op == AlertOperator.INCREASE:
            # 检查相比历史值是否增加超过阈值
            history = self._metric_history.get(metric_name, [])
            if len(history) >= 2:
                prev_value = history[-2][1]
                increase_pct = (value - prev_value) / prev_value if prev_value != 0 else 0
                return increase_pct > rule.threshold
            return False
        elif op == AlertOperator.DECREASE:
            # 检查相比历史值是否减少超过阈值
            history = self._metric_history.get(metric_name, [])
            if len(history) >= 2:
                prev_value = history[-2][1]
                decrease_pct = (prev_value - value) / prev_value if prev_value != 0 else 0
                return decrease_pct > rule.threshold
            return False
        
        return False
    
    def _generate_alert_message(self, rule: AlertRule, value: float) -> str:
        """生成告警消息"""
        operator_str = {
            AlertOperator.GREATER_THAN: "超过",
            AlertOperator.LESS_THAN: "低于",
            AlertOperator.EQUAL: "等于",
            AlertOperator.NOT_EQUAL: "不等于",
            AlertOperator.INCREASE: "增加超过",
            AlertOperator.DECREASE: "减少超过"
        }.get(rule.operator, "触发")
        
        return f"指标 {rule.metric} 当前值 {value:.4f} {operator_str} 阈值 {rule.threshold}"
    
    def _notify_handlers(self, alert: Alert) -> None:
        """通知所有处理器"""
        for handler in self._handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"告警处理器失败: {e}")
    
    def acknowledge_alert(self, alert_index: int) -> bool:
        """确认告警"""
        with self._lock:
            if 0 <= alert_index < len(self._alerts):
                self._alerts[alert_index].acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_index: int) -> bool:
        """解决告警"""
        with self._lock:
            if 0 <= alert_index < len(self._alerts):
                alert = self._alerts[alert_index]
                alert.resolved = True
                alert.resolved_at = datetime.now()
                return True
        return False
    
    def get_alerts(
        self,
        severity: Optional[AlertSeverity] = None,
        resolved: Optional[bool] = None,
        acknowledged: Optional[bool] = None
    ) -> List[Alert]:
        """获取告警列表"""
        with self._lock:
            alerts = self._alerts.copy()
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        if resolved is not None:
            alerts = [a for a in alerts if a.resolved == resolved]
        if acknowledged is not None:
            alerts = [a for a in alerts if a.acknowledged == acknowledged]
        
        return alerts
    
    def get_active_alerts(self) -> List[Alert]:
        """获取活动告警"""
        return self.get_alerts(resolved=False)
    
    def clear_alerts(self) -> None:
        """清除所有告警"""
        with self._lock:
            self._alerts.clear()
    
    def export_rules(self) -> List[Dict[str, Any]]:
        """导出所有规则"""
        return [rule.to_dict() for rule in self.get_rules()]
    
    def export_alerts(self) -> List[Dict[str, Any]]:
        """导出所有告警"""
        return [alert.to_dict() for alert in self._alerts]


# 默认告警规则配置
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="high_error_rate",
        metric="error_rate",
        operator=AlertOperator.GREATER_THAN,
        threshold=0.05,
        severity=AlertSeverity.HIGH,
        duration_minutes=5
    ),
    AlertRule(
        name="low_accuracy",
        metric="accuracy",
        operator=AlertOperator.LESS_THAN,
        threshold=0.55,
        severity=AlertSeverity.HIGH,
        duration_minutes=10
    ),
    AlertRule(
        name="high_latency",
        metric="latency_p95",
        operator=AlertOperator.GREATER_THAN,
        threshold=200,
        severity=AlertSeverity.MEDIUM,
        duration_minutes=5
    ),
    AlertRule(
        name="accuracy_drop",
        metric="accuracy",
        operator=AlertOperator.DECREASE,
        threshold=0.1,
        severity=AlertSeverity.CRITICAL,
        duration_minutes=5
    ),
    AlertRule(
        name="data_drift",
        metric="drift_score",
        operator=AlertOperator.GREATER_THAN,
        threshold=0.5,
        severity=AlertSeverity.MEDIUM,
        duration_minutes=15
    ),
    AlertRule(
        name="high_drawdown",
        metric="max_drawdown",
        operator=AlertOperator.GREATER_THAN,
        threshold=0.15,
        severity=AlertSeverity.HIGH,
        duration_minutes=5
    )
]


# 全局告警管理器实例
_global_alert_manager: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """获取全局告警管理器"""
    global _global_alert_manager
    if _global_alert_manager is None:
        _global_alert_manager = AlertManager()
        # 添加默认规则
        for rule in DEFAULT_ALERT_RULES:
            _global_alert_manager.add_rule(rule)
    return _global_alert_manager


def reset_alert_manager() -> None:
    """重置全局告警管理器"""
    global _global_alert_manager
    _global_alert_manager = None
