#!/usr/bin/env python3
"""
特征层告警管理器

提供特征层的告警管理和监控功能
"""

import logging
import threading
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


logger = logging.getLogger(__name__)


class AlertSeverity(Enum):

    """告警严重程度"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(Enum):

    """告警状态"""
    ACTIVE = "active"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


class Alert:

    """告警对象"""

    def __init__(self, alert_id: str, title: str, message: str, severity: AlertSeverity,


                 source: str, timestamp: Optional[datetime] = None):
        self.alert_id = alert_id
        self.title = title
        self.message = message
        self.severity = severity
        self.source = source
        self.timestamp = timestamp or datetime.now()
        self.status = AlertStatus.ACTIVE
        self.acknowledged_by = None
        self.acknowledged_at = None
        self.resolved_at = None
        self.resolution_notes = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'alert_id': self.alert_id,
            'title': self.title,
            'message': self.message,
            'severity': self.severity.value,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'status': self.status.value,
            'acknowledged_by': self.acknowledged_by,
            'acknowledged_at': self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            'resolved_at': self.resolved_at.isoformat() if self.resolved_at else None,
            'resolution_notes': self.resolution_notes
        }


class AlertManager:

    """特征层告警管理器"""

    def __init__(self):

        self._alerts: Dict[str, Alert] = {}
        self._alert_history: List[Alert] = []
        self._handlers: Dict[str, List[Callable]] = {}
        self._lock = threading.RLock()
        self._next_alert_id = 1
        # 规则驱动的轻量告警信息（用于单测和实时统计）
        self.rules: Dict[str, Dict[str, Any]] = {}
        self.alerts: List[Dict[str, Any]] = []
        self.notifications_sent: List[Dict[str, Any]] = []

    def send_alert(self, title: str, message: str, severity: AlertSeverity = AlertSeverity.INFO,


                   source: str = "feature_monitoring") -> str:
        """发送告警"""
        with self._lock:
            alert_id = f"alert_{self._next_alert_id}"
            self._next_alert_id += 1

            alert = Alert(
                alert_id=alert_id,
                title=title,
                message=message,
                severity=severity,
                source=source
            )

            self._alerts[alert_id] = alert
            self._alert_history.append(alert)

            # 触发处理函数
            self._trigger_handlers(severity.value, alert)

            return alert_id

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Dict[str, Any]]:
        """获取活跃告警"""
        with self._lock:
            alerts = [alert for alert in self._alerts.values()
                      if alert.status == AlertStatus.ACTIVE]

            if severity:
                alerts = [alert for alert in alerts if alert.severity == severity]

            return [alert.to_dict() for alert in alerts]

    def get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取最近的告警"""
        with self._lock:
            recent_alerts = self._alert_history[-limit:] if limit > 0 else self._alert_history
            return [alert.to_dict() for alert in recent_alerts]

    def acknowledge_alert(self, alert_id: str, user: str, notes: Optional[str] = None) -> bool:
        """确认告警"""
        with self._lock:
            for alert in self.alerts:
                if alert.get('id') == alert_id or alert.get('alert_id') == alert_id:
                    alert['status'] = AlertStatus.ACKNOWLEDGED.value
                    alert['acknowledged_by'] = user
                    alert['acknowledged_at'] = datetime.now()
                    if notes:
                        alert['resolution_notes'] = notes
                    return True

            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.ACKNOWLEDGED
                alert.acknowledged_by = user
                alert.acknowledged_at = datetime.now()
                if notes:
                    alert.resolution_notes = notes
                return True
            return False

    def resolve_alert(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """解决告警"""
        with self._lock:
            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                if notes:
                    alert.resolution_notes = notes
                return True
            return False

    def add_alert_rule(
        self,
        rule_name: str,
        condition_func: Callable[[float, Optional[Dict[str, Any]]], bool],
        severity: str = 'warning',
        message_template: str = "",
    ) -> bool:
        """注册基于指标的告警规则"""
        if not callable(condition_func):
            raise ValueError("condition_func 必须可调用")
        with self._lock:
            self.rules[rule_name] = {
                'condition': condition_func,
                'severity': severity,
                'message_template': message_template or "{metric_name}: {value}",
                'enabled': True,
            }
        return True

    def check_condition(
        self,
        metric_name: str,
        metric_value: float,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """根据已注册规则检测是否需要触发告警"""
        context = context or {}
        triggered: List[Dict[str, Any]] = []
        with self._lock:
            for rule_name, rule in self.rules.items():
                if not rule.get('enabled', True):
                    continue
                try:
                    if rule['condition'](metric_value, context):
                        alert_id = f"rule_alert_{self._next_alert_id}"
                        self._next_alert_id += 1
                        alert_payload = {
                            'id': alert_id,
                            'rule_name': rule_name,
                            'metric_name': metric_name,
                            'metric_value': metric_value,
                            'severity': rule['severity'],
                            'message': rule['message_template'].format(
                                metric_name=metric_name,
                                value=metric_value,
                                **context,
                            ),
                            'timestamp': datetime.now(),
                            'status': AlertStatus.ACTIVE.value,
                            'resolved': False,
                            'context': context,
                        }
                        self.alerts.append(alert_payload)
                        triggered.append(alert_payload.copy())
                except Exception as exc:
                    logger.error("告警规则 %s 执行失败: %s", rule_name, exc)

        return triggered

    def send_notification(
        self,
        alert: Dict[str, Any],
        channels: Optional[List[str]] = None,
    ) -> bool:
        """发送告警通知"""
        notification = {
            'alert': alert,
            'channels': channels or ['email'],
            'timestamp': datetime.now(),
            'status': 'sent',
        }
        self.notifications_sent.append(notification)
        return True

    def add_handler(self, severity: str, handler: Callable[[Alert], None]):
        """添加告警处理函数"""
        with self._lock:
            if severity not in self._handlers:
                self._handlers[severity] = []
            self._handlers[severity].append(handler)

    def _trigger_handlers(self, severity: str, alert: Alert):
        """触发处理函数"""
        if severity in self._handlers:
            for handler in self._handlers[severity]:
                try:
                    handler(alert)
                except Exception as e:
                    print(f"Error in alert handler: {e}")

    def clear_history(self):
        """清除历史记录"""
        with self._lock:
            # 只保留最近30天的记录
            cutoff_date = datetime.now() - timedelta(days=30)
            self._alert_history = [alert for alert in self._alert_history
                                   if alert.timestamp > cutoff_date]

    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """获取告警历史"""
        with self._lock:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_history = [alert for alert in self._alert_history
                              if alert.timestamp > cutoff_time]
            return [alert.to_dict() for alert in recent_history]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """返回规则驱动告警的统计信息"""
        with self._lock:
            total_alerts = len(self.alerts)
            active_alerts = sum(1 for alert in self.alerts if not alert.get('resolved'))
            severity_counts: Dict[str, int] = {}
            for alert in self.alerts:
                severity = alert.get('severity', 'info')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1

            return {
                'total_alerts': total_alerts,
                'active_alerts': active_alerts,
                'resolved_alerts': total_alerts - active_alerts,
                'severity_distribution': severity_counts,
                'notifications_sent': len(self.notifications_sent),
            }

    def get_active_alerts(self, severity: Optional[AlertSeverity] = None) -> List[Dict[str, Any]]:
        """获取当前活跃的告警（包含规则驱动和显式发送的告警）"""
        with self._lock:
            custom_active = [alert for alert in self.alerts if not alert.get('resolved')]
            if severity:
                custom_active = [a for a in custom_active if a.get('severity') == severity.value]

            builtin_active = [
                alert.to_dict() for alert in self._alerts.values()
                if alert.status == AlertStatus.ACTIVE
                and (not severity or alert.severity == severity)
            ]
            return [alert.copy() for alert in custom_active] + builtin_active

    def resolve_alert(self, alert_id: str, notes: Optional[str] = None) -> bool:
        """解决规则驱动或传统告警"""
        with self._lock:
            for alert in self.alerts:
                if alert.get('id') == alert_id or alert.get('alert_id') == alert_id:
                    alert['resolved'] = True
                    alert['status'] = AlertStatus.RESOLVED.value
                    alert['resolved_at'] = datetime.now()
                    if notes:
                        alert['resolution_notes'] = notes
                    return True

            if alert_id in self._alerts:
                alert = self._alerts[alert_id]
                alert.status = AlertStatus.RESOLVED
                alert.resolved_at = datetime.now()
                if notes:
                    alert.resolution_notes = notes
                return True

        return False


# 全局告警管理器实例
_alert_manager_instance = None
_alert_manager_lock = threading.Lock()


def get_alert_manager() -> AlertManager:
    """获取全局告警管理器实例"""
    global _alert_manager_instance
    if _alert_manager_instance is None:
        with _alert_manager_lock:
            if _alert_manager_instance is None:
                _alert_manager_instance = AlertManager()
    return _alert_manager_instance
