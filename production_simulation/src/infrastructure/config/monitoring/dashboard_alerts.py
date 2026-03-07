
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any
import logging
import threading
from .dashboard_models import (
    Alert, AlertSeverity, AlertStatus
)

"""
监控面板告警管理

实现告警的创建、管理和处理
"""
logger = logging.getLogger(__name__)


class AlertManager(ABC):
    """告警管理器基类"""

    def __init__(self):
        self._alerts: Dict[str, Alert] = {}
        self._listeners: List[Callable[[Alert], None]] = []
        self._lock = threading.RLock()

    @abstractmethod
    def create_alert(self, name: str, description: str, severity: AlertSeverity,
                     labels: Optional[Dict[str, str]] = None,
                     annotations: Optional[Dict[str, str]] = None,
                     value: Optional[float] = None,
                     threshold: Optional[float] = None) -> str:
        """创建告警"""

    @abstractmethod
    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""

    @abstractmethod
    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""

    def add_listener(self, listener: Callable[[Alert], None]):
        """添加告警监听器"""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def remove_listener(self, listener: Callable[[Alert], None]):
        """移除告警监听器"""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def _notify_listeners(self, alert: Alert):
        """通知所有监听器"""
        for listener in self._listeners:
            try:
                listener(alert)
            except Exception as e:
                logger.error(f"告警监听器执行失败: {e}")

    def get_alert(self, alert_id: str) -> Optional[Alert]:
        """获取告警"""
        with self._lock:
            return self._alerts.get(alert_id)

    def get_alerts_by_status(self, status: AlertStatus) -> List[Alert]:
        """按状态获取告警"""
        with self._lock:
            return [alert for alert in self._alerts.values() if alert.status == status]

    def get_alerts_by_severity(self, severity: AlertSeverity) -> List[Alert]:
        """按严重程度获取告警"""
        with self._lock:
            return [alert for alert in self._alerts.values() if alert.severity == severity]

    def get_all_alerts(self) -> Dict[str, Alert]:
        """获取所有告警"""
        with self._lock:
            return self._alerts.copy()

    def clear_resolved_alerts(self, max_age_days: int = 30) -> int:
        """清理已解决的过期告警"""
        if max_age_days == 0:
            # max_age_days=0 表示清除所有已解决的告警
            cutoff_time = datetime.now() + timedelta(days=1)  # 未来的时间，确保所有都清除
        else:
            cutoff_time = datetime.now() - timedelta(days=max_age_days)

        cleared_count = 0

        with self._lock:
            to_remove = []
            for alert_id, alert in self._alerts.items():
                if (alert.status == AlertStatus.RESOLVED and
                        alert.timestamp < cutoff_time):
                    to_remove.append(alert_id)

            for alert_id in to_remove:
                del self._alerts[alert_id]
                cleared_count += 1

        if cleared_count > 0:
            logger.info(f"清理了 {cleared_count} 个过期的已解决告警")

        return cleared_count


class InMemoryAlertManager(AlertManager):
    """内存告警管理器实现"""

    def __init__(self):
        super().__init__()
        self._next_id = 1

    def create_alert(self, name: str, description: str, severity: AlertSeverity,
                     labels: Optional[Dict[str, str]] = None,
                     annotations: Optional[Dict[str, str]] = None,
                     value: Optional[float] = None,
                     threshold: Optional[float] = None) -> str:
        """创建告警"""
        alert_id = f"alert_{self._next_id}"
        self._next_id += 1

        alert = Alert(
            id=alert_id,
            name=name,
            description=description,
            severity=severity,
            status=AlertStatus.ACTIVE,
            timestamp=datetime.now(),
            labels=labels or {},
            annotations=annotations or {},
            value=value,
            threshold=threshold
        )

        with self._lock:
            self._alerts[alert_id] = alert

        self._notify_listeners(alert)
        logger.info(f"创建告警: {alert_id} - {name} ({severity.value})")

        return alert_id

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.RESOLVED
                alert.timestamp = datetime.now()  # 更新解决时间
                self._notify_listeners(alert)
                logger.info(f"解决告警: {alert_id}")
                return True
        return False

    def acknowledge_alert(self, alert_id: str) -> bool:
        """确认告警"""
        with self._lock:
            alert = self._alerts.get(alert_id)
            if alert and alert.status == AlertStatus.ACTIVE:
                alert.status = AlertStatus.ACKNOWLEDGED
                self._notify_listeners(alert)
                logger.info(f"确认告警: {alert_id}")
                return True
        return False

    def get_active_alerts_count(self) -> int:
        """获取活跃告警数量"""
        with self._lock:
            return len([a for a in self._alerts.values()
                       if a.status == AlertStatus.ACTIVE])

    def get_alerts_summary(self) -> Dict[str, Any]:
        """获取告警摘要"""
        with self._lock:
            summary = {
                'total': len(self._alerts),
                'firing': 0,  # 激活状态的告警
                'resolved': 0,
                'acknowledged': 0,
                'by_severity': {
                    'critical': 0,
                    'error': 0,
                    'warning': 0,
                    'info': 0
                }
            }

            for alert in self._alerts.values():
                if alert.status == AlertStatus.ACTIVE:
                    summary['firing'] += 1
                elif alert.status == AlertStatus.RESOLVED:
                    summary['resolved'] += 1
                elif alert.status == AlertStatus.ACKNOWLEDGED:
                    summary['acknowledged'] += 1

                # 按严重程度统计
                severity_name = alert.severity.name.lower()
                if severity_name in summary['by_severity']:
                    summary['by_severity'][severity_name] += 1

            return summary




