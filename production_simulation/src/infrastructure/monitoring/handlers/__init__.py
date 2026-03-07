"""
监控处理层

提供组件监控和异常监控告警等处理功能。
"""

# 导入处理器
try:
    from .component_monitor import ComponentMonitor
    from .exception_monitoring_alert import ExceptionMonitor
    # 添加别名以保持向后兼容
    ExceptionMonitoringAlert = ExceptionMonitor
except ImportError:
    # 兼容模式，如果新目录结构还未完全迁移
    try:
        from ..component_monitor import ComponentMonitor
        from ..exception_monitoring_alert import ExceptionMonitor
        ExceptionMonitoringAlert = ExceptionMonitor
    except ImportError:
        ComponentMonitor = None
        ExceptionMonitoringAlert = None

__all__ = [
    "ComponentMonitor",
    "ExceptionMonitoringAlert",
]

