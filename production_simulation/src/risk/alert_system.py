"""
风险告警系统模块（别名模块）
提供向后兼容的导入路径

实际实现在 alert/alert_system.py 中
"""

try:
    from .alert.alert_system import (
        AlertSystem,
        AlertLevel,
        AlertType,
        AlertStatus,
        AlertRule,
        Alert,
        NotificationConfig
    )
except ImportError:
    # 提供基础实现
    class AlertSystem:
        pass
    
    class AlertLevel:
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"
    
    class AlertType:
        pass
    
    class AlertStatus:
        pass
    
    class AlertRule:
        pass
    
    class Alert:
        pass
    
    class NotificationConfig:
        pass

__all__ = [
    'AlertSystem',
    'AlertLevel',
    'AlertType',
    'AlertStatus',
    'AlertRule',
    'Alert',
    'NotificationConfig'
]

