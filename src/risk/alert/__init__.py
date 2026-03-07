"""
风险控制层告警模块

提供告警规则引擎和告警系统。
"""

try:
    from .alert_system import (
        AlertSystem,
        AlertLevel,
        AlertType,
        AlertStatus,
        Alert,
        AlertRule,
        NotificationConfig
    )
except ImportError:
    # 如果导入失败，定义占位类
    AlertSystem = None
    AlertLevel = None
    AlertType = None
    AlertStatus = None
    Alert = None
    AlertRule = None
    NotificationConfig = None

__all__ = [
    'AlertSystem',
    'AlertLevel',
    'AlertType',
    'AlertStatus',
    'Alert',
    'AlertRule',
    'NotificationConfig'
]