"""
monitoring_alert_system_facade 模块

提供 monitoring_alert_system_facade 相关功能和接口。
"""

import logging

import time

from ....core.component_registry import InfrastructureComponentRegistry
from .component_factories import ComponentFactoryRegistry, AlertSystemComponentFactory
from .component_proxy_manager import ComponentProxyManager
from .system_operations_manager import SystemOperationsManager
from ...models.alert_dataclasses import AlertRule, Alert, AlertPerformanceMetrics, TestExecutionInfo
from ...models.alert_enums import AlertType, AlertLevel
from typing import Dict, List, Optional, Callable, Any
"""
监控告警系统门面 - 优化版本

优化说明：
- 使用依赖注入和延迟导入减少直接依赖
- 通过InfrastructureComponentRegistry管理组件依赖
- 简化导入结构，提高模块化程度
"""


class LegacyMonitoringAlertSystemFacade:
    """
    监控告警系统门面 (重构后的外观类)

    职责：协调各个专门的组件提供统一的接口，保持向后兼容性
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None,
                 logger: Optional[str] = None):

        # 使用统一的日志记录器
        if isinstance(logger, str):
            self.logger = logging.getLogger(logger)
        elif logger is not None:
            self.logger = logger
        else:
            self.logger = logging.getLogger(self.__class__.__name__)

        # 基础配置
        self.config = config or {}

        # 初始化组件管理系统
        self._init_component_management()

        # 注册默认通知通道和告警规则
        self._setup_system()

    def _init_component_management(self):
        """初始化组件管理系统"""
        # 创建组件工厂注册表
        self.factory_registry = ComponentFactoryRegistry(self.logger)
        
        # 创建告警系统组件工厂
        self.component_factory = AlertSystemComponentFactory(self.factory_registry, self.logger)
        
        # 注册组件工厂
        self._register_component_factories()
        
        # 创建组件代理管理器
        self.component_proxy = ComponentProxyManager(self.factory_registry, self.logger)
        
        # 创建系统操作管理器
        self.system_ops = SystemOperationsManager(self.factory_registry, self.logger)

    def _register_component_factories(self):
        """注册组件工厂函数 - 实现延迟加载"""
        # 注册各种组件工厂
        self.factory_registry.register_factory('performance_monitor', 
                                               self.component_factory.create_performance_monitor)
        self.factory_registry.register_factory('alert_manager', 
                                               self.component_factory.create_alert_manager)
        self.factory_registry.register_factory('notification_manager', 
                                               self.component_factory.create_notification_manager)
        self.factory_registry.register_factory('test_monitor', 
                                               self.component_factory.create_test_monitor)
        self.factory_registry.register_factory('test_execution_manager', 
                                               self.component_factory.create_test_execution_manager)
        self.factory_registry.register_factory('metrics_collector', 
                                               self.component_factory.create_metrics_collector)
        self.factory_registry.register_factory('alert_coordinator', 
                                               self.component_factory.create_alert_coordinator)
        self.factory_registry.register_factory('monitoring_system_coordinator', 
                                               self.component_factory.create_monitoring_system_coordinator)

    # 旧的创建方法已移至ComponentFactory类中

    # 属性访问器 - 使用组件代理管理器
    @property
    def alert_rule_manager(self):
        """获取告警规则管理器"""
        return self.component_proxy.get_alert_rule_manager()

    @property
    def notification_channel_manager(self):
        """获取通知渠道管理器"""
        return self.component_proxy.get_notification_channel_manager()

    @property
    def test_execution_manager(self):
        """获取测试执行管理器"""
        return self.component_proxy.get_test_execution_manager()

    @property
    def metrics_collector(self):
        """获取指标收集器"""
        return self.component_proxy.get_metrics_collector()

    @property
    def alert_coordinator(self):
        """获取告警协调器"""
        return self.component_proxy.get_alert_coordinator()

    # 外观方法 - 保持向后兼容性
    def add_alert_rule(self, rule: AlertRule):
        """添加告警规则"""
        manager = self.alert_rule_manager
        if manager:
            return manager.add_alert_rule(rule)
        return False

    def remove_alert_rule(self, rule_name: str):
        """移除告警规则"""
        manager = self.alert_rule_manager
        if manager:
            return manager.remove_alert_rule(rule_name)
        return False

    def get_alert_rules(self) -> List[AlertRule]:
        """获取告警规则"""
        manager = self.alert_rule_manager
        if manager:
            return manager.get_alert_rules()
        return []

    def configure_notification_channel(self, channel_name: str, config: Dict[str, Any]):
        """配置通知渠道"""
        manager = self.notification_channel_manager
        if manager:
            return manager.configure_notification_channel(channel_name, config)
        return False

    def get_notification_channels(self) -> Dict[str, bool]:
        """获取通知渠道"""
        manager = self.notification_channel_manager
        if manager:
            return manager.get_notification_channels()
        return {}

    def register_test(self, test_id: str, test_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """注册测试"""
        manager = self.test_execution_manager
        if manager:
            return manager.register_test(test_id, test_name, metadata)
        return ""

    def update_test_status(self, test_id: str, status: str,
                           message: Optional[str] = None,
                           metrics: Optional[Dict[str, Any]] = None):
        """更新测试状态"""
        manager = self.test_execution_manager
        if manager:
            return manager.update_test_status(test_id, status, message, metrics)
        return False

    def get_active_tests(self) -> List[TestExecutionInfo]:
        """获取活跃测试"""
        manager = self.test_execution_manager
        if manager:
            return manager.get_active_tests()
        return []

    def get_test_history(self, hours: int = 24) -> List[TestExecutionInfo]:
        """获取测试历史"""
        manager = self.test_execution_manager
        if manager:
            return manager.get_test_history(hours)
        return []

    def get_current_metrics(self) -> AlertPerformanceMetrics:
        """获取当前指标"""
        collector = self.metrics_collector
        if collector:
            return collector.get_current_metrics()
        return AlertPerformanceMetrics()

    def get_metrics_history(self, minutes: int = 60) -> List[AlertPerformanceMetrics]:
        """获取指标历史"""
        collector = self.metrics_collector
        if collector:
            return collector.get_metrics_history(minutes)
        return []

    def get_average_metrics(self, minutes: int = 60) -> AlertPerformanceMetrics:
        """获取平均指标"""
        collector = self.metrics_collector
        if collector:
            return collector.get_average_metrics(minutes)
        return AlertPerformanceMetrics()

    def get_active_alerts(self) -> List[Alert]:
        """获取活跃告警"""
        coordinator = self.alert_coordinator
        if coordinator:
            return coordinator.get_active_alerts()
        return []

    def resolve_alert(self, alert_id: str):
        """解决告警"""
        coordinator = self.alert_coordinator
        if coordinator:
            return coordinator.resolve_alert(alert_id)
        return False

    def get_alert_statistics(self) -> Dict[str, Any]:
        """获取告警统计"""
        coordinator = self.alert_coordinator
        if coordinator:
            return coordinator.get_alert_statistics()
        return {}

    def start(self):
        """启动系统"""
        return self.system_ops.start_system()

    def stop(self):
        """停止系统"""
        return self.system_ops.stop_system()

    def get_system_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return self.system_ops.get_system_status()

    def get_system_health_report(self) -> Dict[str, Any]:
        """获取系统健康报告"""
        return self.system_ops.get_system_health_report()

    def update_configuration(self, config: Dict[str, Any]):
        """更新配置"""
        self.config.update(config)
        return self.system_ops.update_configuration(config)

    def get_configuration(self) -> Dict[str, Any]:
        """获取配置"""
        return self.config.copy()

    def _setup_system(self):
        """设置系统（兼容性方法）"""
        self.logger.info("监控告警系统初始化完成")
