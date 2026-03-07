"""
基础设施层性能监控集成

将性能监控集成到现有组件中
"""

from phase3_performance_monitoring import UnifiedPerformanceMonitor
from typing import Dict, Any, Optional
from abc import ABC, abstractmethod


class IPerformanceMonitorable(ABC):
    """性能监控接口"""

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""

    @abstractmethod
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""


class BasePerformanceMonitorable(IPerformanceMonitorable):
    """基础性能监控类"""

    def __init__(self, component_name: str):
        self._component_name = component_name
        self._performance_monitor = UnifiedPerformanceMonitor()

    def get_performance_metrics(self) -> Dict[str, Any]:
        """获取性能指标"""
        return self._performance_monitor.get_performance_summary(self._component_name)

    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态"""
        # 实现基本的健康检查逻辑
        try:
            # 检查组件是否能正常工作
            self._perform_health_check()
            return {
                "component": self._component_name,
                "status": "healthy",
                "timestamp": "2025-01-21T12:00:00Z"
            }
        except Exception as e:
            return {
                "component": self._component_name,
                "status": "unhealthy",
                "error": str(e),
                "timestamp": "2025-01-21T12:00:00Z"
            }

    def _perform_health_check(self):
        """执行健康检查"""
        # 子类应该重写此方法

    def measure_operation(self, operation_name: str, metadata: Optional[Dict[str, Any]] = None):
        """操作性能测量装饰器"""
        return self._performance_monitor.measure_performance(
            self._component_name,
            operation_name,
            metadata
        )


# 全局监控实例
_global_monitor = UnifiedPerformanceMonitor()


def get_global_monitor() -> UnifiedPerformanceMonitor:
    """获取全局监控实例"""
    return _global_monitor


def start_global_monitoring():
    """启动全局监控"""
    _global_monitor.start_system_monitoring()


def stop_global_monitoring():
    """停止全局监控"""
    _global_monitor.stop_system_monitoring()


def measure_performance(component_name: str, operation_name: str):
    """性能测量装饰器"""
    return _global_monitor.measure_performance(component_name, operation_name)
