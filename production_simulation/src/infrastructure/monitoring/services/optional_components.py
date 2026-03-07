"""可选依赖加载工具。

提供简易的惰性导入机制，用于连续监控服务在缺失部分组件
时优雅地降级。"""

from importlib import import_module
from functools import lru_cache
from typing import Any, Optional


_COMPONENT_PATHS = {
    "MetricsCollector": (
        "src.infrastructure.monitoring.services.metrics_collector.MetricsCollector",
    ),
    "AlertManager": (
        "src.infrastructure.monitoring.components.alert_manager.AlertManager",
    ),
    "DataPersistence": (
        "src.infrastructure.monitoring.components.data_persistence.DataPersistence",
    ),
    "OptimizationEngine": (
        "src.infrastructure.monitoring.components.optimization_engine.OptimizationEngine",
    ),
    "HealthCheckInterface": (
        "src.infrastructure.core.health_check_interface.HealthCheckInterface",
    ),
}


def _optional_import(path: str) -> Optional[Any]:
    module_name, attr = path.rsplit(".", 1)
    try:
        module = import_module(module_name)
        return getattr(module, attr)
    except Exception:
        return None


@lru_cache(maxsize=None)
def get_optional_component(name: str) -> Optional[Any]:
    for candidate in _COMPONENT_PATHS.get(name, ()):  # pragma: no branch
        component = _optional_import(candidate)
        if component is not None:
            return component
    return None
