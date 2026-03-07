"""
监控告警系统模块（别名模块）
提供向后兼容的导入路径
"""

try:
    from .services.alert_service import (
        AlertSystem,
        AlertLevel,
        Alert,
        IntelligentAlertSystem,
    )
except ImportError:
    from enum import Enum

    class AlertSystem:
        """基础告警系统占位实现"""

        def __init__(self, *_, **__):
            pass

    class AlertLevel(Enum):
        INFO = "info"
        WARNING = "warning"
        ERROR = "error"
        CRITICAL = "critical"

    class Alert:
        """告警占位数据结构"""

        def __init__(self, *_, **__):
            pass

    class IntelligentAlertSystem:
        """智能告警系统占位实现"""

        def __init__(self, *_, **__):
            pass

__all__ = ['AlertSystem', 'AlertLevel', 'Alert', 'IntelligentAlertSystem']

