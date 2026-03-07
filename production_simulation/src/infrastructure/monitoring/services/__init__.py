"""
监控服务层

提供核心监控服务，包括连续监控、统一监控接口和告警系统。
"""

# 导入核心服务
try:
    from .continuous_monitoring_service import ContinuousMonitoringSystem
    from .unified_monitoring_service import UnifiedMonitoring
    from .alert_service import IntelligentAlertSystem
    # 导入重构后的组件
    from .continuous_monitoring_system_refactored import ContinuousMonitoringSystemRefactored
    from .intelligent_alert_system_refactored import IntelligentAlertSystemRefactored
except ImportError:
    # 兼容模式，如果新目录结构还未完全迁移
    try:
        from ..continuous_monitoring_system import ContinuousMonitoringSystem
        from ..unified_monitoring import UnifiedMonitoring
        from ..alert_system import IntelligentAlertSystem
        ContinuousMonitoringSystemRefactored = None
        IntelligentAlertSystemRefactored = None
    except ImportError:
        ContinuousMonitoringSystem = None
        UnifiedMonitoring = None
        IntelligentAlertSystem = None
        ContinuousMonitoringSystemRefactored = None
        IntelligentAlertSystemRefactored = None

__all__ = [
    "ContinuousMonitoringSystem",
    "ContinuousMonitoringSystemRefactored",  # 重构后的版本
    "UnifiedMonitoring",
    "IntelligentAlertSystem",
    "IntelligentAlertSystemRefactored",  # 重构后的版本
]

