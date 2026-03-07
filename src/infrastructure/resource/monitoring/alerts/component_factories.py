"""
组件工厂类

负责创建各种监控和告警相关的组件实例。
"""

import logging
from typing import Dict, Any, Optional, Callable, List

# 使用try-except来处理可能缺失的导入
try:
    from ..core.alert_rule_manager import AlertRuleManager
except ImportError:
    AlertRuleManager = None

try:
    from ..core.notification_channel_manager import NotificationChannelManager
except ImportError:
    NotificationChannelManager = None

try:
    from .alert_coordinator import AlertCoordinator
except ImportError:
    AlertCoordinator = None

try:
    from .metrics_collector import CachedMetricsCollector
except ImportError:
    CachedMetricsCollector = None

try:
    from .test_execution_manager import TestExecutionManager
except ImportError:
    TestExecutionManager = None

try:
    from .alert_manager_component import AlertManager
except ImportError:
    AlertManager = None

try:
    from .monitoring_system_coordinator import MonitoringSystemCoordinator
except ImportError:
    MonitoringSystemCoordinator = None

try:
    from .notification_manager_component import NotificationManager
except ImportError:
    NotificationManager = None

try:
    from .performance_monitor_component import PerformanceMonitor
except ImportError:
    PerformanceMonitor = None

try:
    from .test_execution_monitor_component import TestExecutionMonitor
except ImportError:
    TestExecutionMonitor = None


class ComponentFactoryRegistry:
    """组件工厂注册表"""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self._factories: Dict[str, Callable] = {}
        self._components: Dict[str, Any] = {}
    
    def register_factory(self, component_name: str, factory_func: Callable) -> None:
        """注册组件工厂函数"""
        self._factories[component_name] = factory_func
    
    def create_component(self, component_name: str, config: Optional[Dict[str, Any]] = None) -> Any:
        """创建组件实例"""
        if component_name in self._components:
            return self._components[component_name]
        
        factory = self._factories.get(component_name)
        if not factory:
            self.logger.warning(f"未找到组件工厂: {component_name}")
            return None
        
        try:
            component = factory(config)
            self._components[component_name] = component
            return component
        except Exception as e:
            self.logger.error(f"创建组件 {component_name} 失败: {e}")
            return None
    
    def list_components(self) -> List[str]:
        """列出所有已注册的组件名称"""
        return list(self._factories.keys())


class AlertSystemComponentFactory:
    """告警系统组件工厂"""
    
    def __init__(self, registry: ComponentFactoryRegistry, logger: logging.Logger):
        self.registry = registry
        self.logger = logger
    
    def create_performance_monitor(self, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """创建性能监控器"""
        if PerformanceMonitor is None:
            self.logger.warning("PerformanceMonitor组件不可用")
            return None
        try:
            return PerformanceMonitor(config or {})
        except ImportError:
            self.logger.warning("PerformanceMonitor组件不可用")
            return None

    def create_alert_manager(self, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """创建告警管理器"""
        if AlertManager is None:
            self.logger.warning("AlertManager组件不可用")
            return None
        try:
            return AlertManager(config or {})
        except ImportError:
            self.logger.warning("AlertManager组件不可用")
            return None

    def create_notification_manager(self, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """创建通知管理器"""
        if NotificationManager is None:
            self.logger.warning("NotificationManager组件不可用")
            return None
        try:
            return NotificationManager(config or {})
        except ImportError:
            self.logger.warning("NotificationManager组件不可用")
            return None

    def create_test_monitor(self, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """创建测试监控器"""
        if TestExecutionMonitor is None:
            self.logger.warning("TestExecutionMonitor组件不可用")
            return None
        try:
            return TestExecutionMonitor(config or {})
        except ImportError:
            self.logger.warning("TestExecutionMonitor组件不可用")
            return None

    def create_test_execution_manager(self, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """创建测试执行管理器"""
        if TestExecutionManager is None:
            self.logger.warning("TestExecutionManager组件不可用")
            return None
        try:
            test_monitor = self.registry.create_component('test_monitor', config)
            return TestExecutionManager(test_monitor, self.logger)
        except ImportError:
            self.logger.warning("TestExecutionManager组件不可用")
            return None

    def create_metrics_collector(self, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """创建指标收集器"""
        if CachedMetricsCollector is None:
            self.logger.warning("CachedMetricsCollector组件不可用")
            return None
        try:
            performance_monitor = self.registry.create_component('performance_monitor', config)
            return CachedMetricsCollector(performance_monitor, self.logger)
        except ImportError:
            self.logger.warning("CachedMetricsCollector组件不可用")
            return None

    def create_alert_coordinator(self, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """创建告警协调器"""
        if AlertCoordinator is None:
            self.logger.warning("AlertCoordinator组件不可用")
            return None
        try:
            alert_manager = self.registry.create_component('alert_manager', config)
            return AlertCoordinator(alert_manager, self.logger)
        except ImportError:
            self.logger.warning("AlertCoordinator组件不可用")
            return None

    def create_monitoring_system_coordinator(self, config: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """创建监控系统协调器"""
        if MonitoringSystemCoordinator is None:
            self.logger.warning("MonitoringSystemCoordinator组件不可用")
            return None
        try:
            return MonitoringSystemCoordinator(config or {}, self.logger)
        except ImportError:
            self.logger.warning("MonitoringSystemCoordinator组件不可用")
            return None
