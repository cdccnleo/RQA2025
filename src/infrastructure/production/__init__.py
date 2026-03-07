"""生产就绪模块

提供系统生产环境部署前的全面检查和验证功能。
"""

try:
    from .production_readiness_manager import (
        ProductionReadinessManager,
        ProductionReadinessConfig,
        ReadinessReport,
        CheckResult,
        CheckStatus,
        CheckCategory,
        HealthChecker,
        ConfigValidator,
        PerformanceChecker,
        SecurityChecker,
        get_global_manager,
        clear_global_manager,
    )
except ImportError:
    class ProductionReadinessManager:
        pass
    class ProductionReadinessConfig:
        pass
    class ReadinessReport:
        pass
    class CheckResult:
        pass
    class CheckStatus:
        pass
    class CheckCategory:
        pass
    class HealthChecker:
        pass
    class ConfigValidator:
        pass
    class PerformanceChecker:
        pass
    class SecurityChecker:
        pass
    
    def get_global_manager(config=None):
        pass
    
    def clear_global_manager():
        pass

__all__ = [
    'ProductionReadinessManager',
    'ProductionReadinessConfig',
    'ReadinessReport',
    'CheckResult',
    'CheckStatus',
    'CheckCategory',
    'HealthChecker',
    'ConfigValidator',
    'PerformanceChecker',
    'SecurityChecker',
    'get_global_manager',
    'clear_global_manager',
]
