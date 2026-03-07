from .core.performance_monitor import PerformanceMonitor
from .components.metrics_collector import MetricsCollector
from .services.unified_monitoring_service import UnifiedMonitoring
from .infrastructure.system_monitor import SystemMonitor
from .infrastructure.storage_monitor import StorageMonitor
try:
    from .application_monitor import ApplicationMonitor
except ImportError:
    from .application.application_monitor import ApplicationMonitor

# 导出system_monitor别名（向后兼容）
try:
    from .infrastructure.system_monitor import SystemMonitor as MonitoringSystemMonitor
    # 如果其他地方需要，也可以导出
except ImportError:
    pass

# 为了向后兼容，保留旧名称
MonitoringService = UnifiedMonitoring
UnifiedMonitoringService = UnifiedMonitoring
MonitoringSystem = UnifiedMonitoring


# class ApplicationMonitor:
#     def __init__(self, *args, **kwargs):
#         pass
