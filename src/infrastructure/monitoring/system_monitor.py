"""
系统监控器模块（别名模块）
提供向后兼容的导入路径

实际实现在 monitoring/infrastructure/system_monitor.py 中
"""

try:
    from .infrastructure.system_monitor import SystemMonitor
except ImportError:
    try:
        from ..resource.system_monitor import SystemMonitor
    except ImportError:
        # 提供基础实现
        class SystemMonitor:
            pass

__all__ = ['SystemMonitor']

