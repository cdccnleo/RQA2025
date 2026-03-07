"""
监控告警系统模块（别名模块）
提供向后兼容的导入路径
"""

try:
    from ..monitoring.alert_system import AlertSystem as MonitoringAlertSystem
except ImportError:
    try:
        from src.risk.alert_system import AlertSystem as MonitoringAlertSystem
    except ImportError:
        # 提供基础实现
        class MonitoringAlertSystem:
            pass

__all__ = ['MonitoringAlertSystem']

