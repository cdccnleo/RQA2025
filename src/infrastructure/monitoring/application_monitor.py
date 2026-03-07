"""
应用监控器模块（别名模块）
提供向后兼容的导入路径

实际实现在 application/application_monitor.py 中
"""

try:
    from .application.application_monitor import ApplicationMonitor
except ImportError:
    try:
        from .health.monitoring.application_monitor import ApplicationMonitor
    except ImportError:
        # 提供基础实现
        class ApplicationMonitor:
            pass

__all__ = ['ApplicationMonitor']

