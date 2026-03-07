
import threading
import time

from ..alert_dataclasses import Alert, PerformanceMetrics
from ..alert_enums import AlertType, AlertLevel
from ..shared_interfaces import ILogger, StandardLogger
from typing import List, Dict, Any, Optional
"""
告警协调器

职责：协调告警的检测、处理和通知
"""


class AlertCoordinator:
    """
    告警协调器

    职责：协调告警的检测、处理和通知
    """

    def __init__(self, alert_manager, logger: Optional[ILogger] = None):
        self.alert_manager = alert_manager
        self.logger = logger or StandardLogger(f"{self.__class__.__name__}")
        self._lock = threading.Lock()
        self._active_alerts: Dict[str, Alert] = {}

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        with self._lock:
            return list(self._active_alerts.values())

    def resolve_alert(self, alert_id: str) -> bool:
        """解决告警"""
        with self._lock:
            try:
                if alert_id in self._active_alerts:
                    del self._active_alerts[alert_id]
                    self.logger.log_info(f"告警已解决: {alert_id}")
                    return True
                else:
                    self.logger.log_warning(f"未找到告警: {alert_id}")
                    return False
            except Exception as e:
                self.logger.log_error(f"解决告警失败: {e}")
                return False

    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        with self._lock:
            stats = {
                'total_active': len(self._active_alerts),
                'by_level': {},
                'by_type': {}
            }

            for alert in self._active_alerts.values():
                # 按级别统计
                level_str = alert.level.value if hasattr(alert.level, 'value') else str(alert.level)
                stats['by_level'][level_str] = stats['by_level'].get(level_str, 0) + 1

                # 按类型统计
                type_str = alert.type.value if hasattr(alert.type, 'value') else str(alert.type)
                stats['by_type'][type_str] = stats['by_type'].get(type_str, 0) + 1

            return stats

    def check_alerts(self, metrics: PerformanceMetrics) -> List[Alert]:
        """检查告警条件"""
        with self._lock:
            alerts = []
            try:
                # CPU使用率告警
                if metrics.cpu_usage > 80.0:
                    alert = Alert(
                        type=AlertType.PERFORMANCE,
                        level=AlertLevel.WARNING,
                        message=f"CPU使用率过高: {metrics.cpu_usage}%",
                        timestamp=time.time()
                    )
                    alerts.append(alert)
                    self._active_alerts[str(id(alert))] = alert

                # 内存使用率告警
                if metrics.memory_usage > 85.0:
                    alert = Alert(
                        type=AlertType.PERFORMANCE,
                        level=AlertLevel.WARNING,
                        message=f"内存使用率过高: {metrics.memory_usage}%",
                        timestamp=time.time()
                    )
                    alerts.append(alert)
                    self._active_alerts[str(id(alert))] = alert

                # 磁盘使用率告警
                if metrics.disk_usage > 90.0:
                    alert = Alert(
                        type=AlertType.PERFORMANCE,
                        level=AlertLevel.CRITICAL,
                        message=f"磁盘使用率过高: {metrics.disk_usage}%",
                        timestamp=time.time()
                    )
                    alerts.append(alert)
                    self._active_alerts[str(id(alert))] = alert

                return alerts

            except Exception as e:
                self.logger.log_error(f"检查告警失败: {e}")
                return []
