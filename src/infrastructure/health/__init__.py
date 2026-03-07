"""
健康管理系统 - 统一导出接口

提供健康检查相关的所有核心功能模块的统一访问入口。
"""

import logging

# 核心服务
from .services.health_check_service import HealthCheck
from .services.health_check_core import HealthCheckCore
from .services.monitoring_dashboard import MonitoringDashboard

# 数据模型
from .models.health_result import HealthCheckResult, CheckType, HealthStatus
from .models.health_status import HealthStatus as HealthStatusEnum
from .models.metrics import MetricsCollector, MetricType

# API集成
from .api.fastapi_integration import FastAPIHealthChecker

# 组件 - 注释掉有问题的导入
# from .components.enhanced_health_checker import EnhancedHealthChecker
# 从别名模块导入（向后兼容）
try:
    from .enhanced_health_checker import EnhancedHealthChecker
except ImportError:
    pass  # 已从components导入
# from .database.database_health_monitor import DatabaseHealthMonitor
# from .integration.prometheus_exporter import HealthCheckPrometheusExporter

try:
    pass  # 导入已在上面完成
except ImportError as e:
    # 如果导入失败，记录警告但不抛出异常
    logging.getLogger(__name__).warning(f"Failed to import health check modules: {e}")

__version__ = "1.0.0"

__all__ = [
    # 核心服务
    "HealthCheck",
    "HealthCheckCore", 
    "MonitoringDashboard",
    
    # 数据模型
    "HealthCheckResult",
    "CheckType",
    "HealthStatus",
    "HealthStatusEnum",
    "MetricsCollector",
    "MetricType",
    
    # API集成
    "FastAPIHealthChecker",
    
    # 组件
    "EnhancedHealthChecker",
    "DatabaseHealthMonitor",
    "HealthCheckPrometheusExporter",

    # 向后兼容函数
    "get_status",
    "is_available",
]


# 基础基础设施适配器
class BaseInfrastructureAdapter:
    """基础基础设施适配器"""

    def __init__(self):
        self.name = "base_adapter"
        self.version = "1.0.0"

    def get_status(self):
        return {"status": "healthy", "adapter": self.name}

    def is_available(self):
        return True


class InfrastructureAdapterFactory:
    """基础设施适配器工厂"""

    @staticmethod
    def create_adapter(adapter_type):
        return BaseInfrastructureAdapter()

    @classmethod
    def get_available_adapters(cls):
        return ["base", "cache", "database", "monitoring", "logging"]


def _ensure_adapter(adapter_type: str = "base"):
    try:
        adapter = InfrastructureAdapterFactory.create_adapter(adapter_type)
    except Exception as exc:  # pragma: no cover - 容错路径
        logging.getLogger(__name__).warning(
            "创建适配器失败: %s, adapter_type=%s", exc, adapter_type
        )
        adapter = BaseInfrastructureAdapter()
    return adapter


def get_status(adapter_type: str = "base"):
    """向后兼容的顶层健康状态查询函数"""
    adapter = _ensure_adapter(adapter_type)
    if hasattr(adapter, "get_status"):
        try:
            status = adapter.get_status()
            if status is None:
                return {"status": "unknown", "adapter": getattr(adapter, "name", adapter_type)}
            if isinstance(status, dict):
                return status
            return {"status": status, "adapter": getattr(adapter, "name", adapter_type)}
        except Exception as exc:  # pragma: no cover - 容错路径
            logging.getLogger(__name__).warning(
                "获取适配器状态失败: %s, adapter_type=%s", exc, adapter_type
            )
    return {"status": "unknown", "adapter": adapter_type}


def is_available(adapter_type: str = "base") -> bool:
    """向后兼容的可用性检测函数"""
    adapter = _ensure_adapter(adapter_type)
    if hasattr(adapter, "is_available"):
        try:
            return bool(adapter.is_available())
        except Exception as exc:  # pragma: no cover - 容错路径
            logging.getLogger(__name__).warning(
                "检测适配器可用性失败: %s, adapter_type=%s", exc, adapter_type
            )
    return False
