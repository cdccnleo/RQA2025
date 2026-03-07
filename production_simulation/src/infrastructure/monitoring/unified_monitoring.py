"""
统一监控模块（别名模块）
提供向后兼容的导入路径

实际实现在 services/unified_monitoring_service.py 中
"""

try:
    from .services.unified_monitoring_service import UnifiedMonitoring, UnifiedMonitoringService, MonitoringService
except ImportError:
    # 提供基础实现
    class UnifiedMonitoring:
        pass
    
    UnifiedMonitoringService = UnifiedMonitoring
    MonitoringService = UnifiedMonitoring

__all__ = ['UnifiedMonitoring', 'UnifiedMonitoringService', 'MonitoringService']

