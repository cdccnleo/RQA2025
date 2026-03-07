"""
组件代理管理器

负责管理各种组件的延迟加载和代理访问。
"""

from typing import Dict, Any, Optional, List
import logging

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
    from ...models.alert_dataclasses import AlertRule, Alert, TestExecutionInfo, AlertPerformanceMetrics
except ImportError:
    AlertRule = Alert = TestExecutionInfo = AlertPerformanceMetrics = None


class ComponentProxyManager:
    """组件代理管理器"""
    
    def __init__(self, registry, logger: logging.Logger):
        self.registry = registry
        self.logger = logger
        self._components = {}

    def get_alert_rule_manager(self) -> Optional[Any]:
        """获取告警规则管理器"""
        # 首先检查缓存
        if 'alert_rule_manager' in self._components:
            return self._components['alert_rule_manager']
        
        # 尝试创建组件
        alert_manager = self.registry.create_component('alert_manager')
        
        # 如果AlertRuleManager可用且alert_manager创建成功，则创建AlertRuleManager
        if AlertRuleManager is not None and alert_manager:
            try:
                self._components['alert_rule_manager'] = AlertRuleManager(
                    alert_manager, self.logger)
                return self._components['alert_rule_manager']
            except ImportError:
                self.logger.error("AlertRuleManager不可用")
                return None
        
        # AlertRuleManager不可用或alert_manager创建失败
        return None

    def get_notification_channel_manager(self) -> Optional[Any]:
        """获取通知渠道管理器"""
        if NotificationChannelManager is None:
            return None
        if 'notification_channel_manager' not in self._components:
            notification_manager = self.registry.create_component('notification_manager')
            if notification_manager:
                try:
                    self._components['notification_channel_manager'] = NotificationChannelManager(
                        notification_manager, self.logger)
                except ImportError:
                    self.logger.error("NotificationChannelManager不可用")
                    return None
        return self._components.get('notification_channel_manager')

    def get_test_execution_manager(self) -> Optional[Any]:
        """获取测试执行管理器"""
        if TestExecutionManager is None:
            return None
        if 'test_execution_manager' not in self._components:
            test_monitor = self.registry.create_component('test_monitor')
            if test_monitor:
                try:
                    self._components['test_execution_manager'] = TestExecutionManager(
                        test_monitor, self.logger)
                except ImportError:
                    self.logger.error("TestExecutionManager不可用")
                    return None
        return self._components.get('test_execution_manager')

    def get_metrics_collector(self) -> Optional[Any]:
        """获取指标收集器"""
        if CachedMetricsCollector is None:
            return None
        if 'metrics_collector' not in self._components:
            performance_monitor = self.registry.create_component('performance_monitor')
            if performance_monitor:
                try:
                    self._components['metrics_collector'] = CachedMetricsCollector(
                        performance_monitor, self.logger)
                except ImportError:
                    self.logger.error("CachedMetricsCollector不可用")
                    return None
        return self._components.get('metrics_collector')

    def get_alert_coordinator(self) -> Optional[Any]:
        """获取告警协调器"""
        if AlertCoordinator is None:
            return None
        if 'alert_coordinator' not in self._components:
            alert_manager = self.registry.create_component('alert_manager')
            if alert_manager:
                try:
                    self._components['alert_coordinator'] = AlertCoordinator(
                        alert_manager, self.logger)
                except ImportError:
                    self.logger.error("AlertCoordinator不可用")
                    return None
        return self._components.get('alert_coordinator')
