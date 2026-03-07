"""
component_factories.py 测试模块

测试组件工厂类的功能，提升覆盖率。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, MagicMock, patch
import logging

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../..'))

# 导入被测试的模块
try:
    from src.infrastructure.resource.monitoring.alerts.component_factories import (
        ComponentFactoryRegistry,
        AlertSystemComponentFactory
    )
except ImportError as e:
    # 如果导入失败，我们跳过这些测试
    ComponentFactoryRegistry = None
    AlertSystemComponentFactory = None
    IMPORT_ERROR = str(e)


class TestComponentFactoryRegistry(unittest.TestCase):
    """测试组件工厂注册表"""

    def setUp(self):
        """设置测试环境"""
        if ComponentFactoryRegistry is None:
            self.skipTest(f"Cannot import ComponentFactoryRegistry: {IMPORT_ERROR}")
        self.registry = ComponentFactoryRegistry()
        self.logger = logging.getLogger('test')

    def test_component_factory_registry_initialization(self):
        """测试组件工厂注册表初始化"""
        registry = ComponentFactoryRegistry()
        self.assertIsNotNone(registry._factories)
        self.assertIsNotNone(registry._components)
        self.assertIsInstance(registry._factories, dict)
        self.assertIsInstance(registry._components, dict)

    def test_component_factory_registry_with_logger(self):
        """测试使用自定义日志器初始化"""
        registry = ComponentFactoryRegistry(self.logger)
        self.assertEqual(registry.logger, self.logger)

    def test_register_factory(self):
        """测试注册组件工厂函数"""
        def test_factory(config):
            return "test_component"
        
        self.registry.register_factory("test_component", test_factory)
        self.assertIn("test_component", self.registry._factories)
        self.assertEqual(self.registry._factories["test_component"], test_factory)

    def test_create_component_success(self):
        """测试成功创建组件"""
        def test_factory(config):
            return f"component_{config}"
        
        self.registry.register_factory("test_component", test_factory)
        component = self.registry.create_component("test_component", {"param": "value"})
        self.assertEqual(component, "component_{'param': 'value'}")
        self.assertIn("test_component", self.registry._components)

    def test_create_component_cached(self):
        """测试组件缓存机制"""
        def test_factory(config):
            return "cached_component"
        
        self.registry.register_factory("test_component", test_factory)
        
        # 第一次创建
        component1 = self.registry.create_component("test_component")
        # 第二次创建应该返回缓存的组件
        component2 = self.registry.create_component("test_component")
        
        self.assertEqual(component1, component2)
        self.assertEqual(component1, "cached_component")

    def test_create_component_factory_not_found(self):
        """测试工厂函数不存在的情况"""
        with patch.object(self.registry.logger, 'warning') as mock_warning:
            component = self.registry.create_component("nonexistent_component")
            self.assertIsNone(component)
            mock_warning.assert_called_once_with("未找到组件工厂: nonexistent_component")

    def test_create_component_factory_exception(self):
        """测试工厂函数抛出异常的情况"""
        def failing_factory(config):
            raise Exception("Factory error")
        
        self.registry.register_factory("failing_component", failing_factory)
        
        with patch.object(self.registry.logger, 'error') as mock_error:
            component = self.registry.create_component("failing_component")
            self.assertIsNone(component)
            mock_error.assert_called_once_with("创建组件 failing_component 失败: Factory error")

    def test_list_components(self):
        """测试列出所有组件"""
        def factory1(config):
            return "component1"
        def factory2(config):
            return "component2"
        
        self.registry.register_factory("component1", factory1)
        self.registry.register_factory("component2", factory2)
        
        components = self.registry.list_components()
        self.assertIn("component1", components)
        self.assertIn("component2", components)
        self.assertEqual(len(components), 2)


class TestAlertSystemComponentFactory(unittest.TestCase):
    """测试告警系统组件工厂"""

    def setUp(self):
        """设置测试环境"""
        if AlertSystemComponentFactory is None:
            self.skipTest(f"Cannot import AlertSystemComponentFactory: {IMPORT_ERROR}")
        self.registry = Mock(spec=ComponentFactoryRegistry)
        self.logger = Mock(spec=logging.Logger)
        self.factory = AlertSystemComponentFactory(self.registry, self.logger)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.PerformanceMonitor')
    def test_create_performance_monitor_success(self, mock_performance_monitor):
        """测试成功创建性能监控器"""
        mock_instance = Mock()
        mock_performance_monitor.return_value = mock_instance
        
        result = self.factory.create_performance_monitor({"interval": 60})
        
        mock_performance_monitor.assert_called_once_with({"interval": 60})
        self.assertEqual(result, mock_instance)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.PerformanceMonitor')
    def test_create_performance_monitor_import_error(self, mock_performance_monitor):
        """测试性能监控器导入错误"""
        mock_performance_monitor.side_effect = ImportError("Cannot import")
        
        result = self.factory.create_performance_monitor()
        
        self.assertIsNone(result)
        self.logger.warning.assert_called_once_with("PerformanceMonitor组件不可用")

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.PerformanceMonitor')
    def test_create_performance_monitor_with_empty_config(self, mock_performance_monitor):
        """测试使用空配置创建性能监控器"""
        mock_instance = Mock()
        mock_performance_monitor.return_value = mock_instance
        
        result = self.factory.create_performance_monitor(None)
        
        mock_performance_monitor.assert_called_once_with({})
        self.assertEqual(result, mock_instance)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.AlertManager')
    def test_create_alert_manager_success(self, mock_alert_manager):
        """测试成功创建告警管理器"""
        mock_instance = Mock()
        mock_alert_manager.return_value = mock_instance
        
        result = self.factory.create_alert_manager({"rules": []})
        
        mock_alert_manager.assert_called_once_with({"rules": []})
        self.assertEqual(result, mock_instance)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.AlertManager')
    def test_create_alert_manager_import_error(self, mock_alert_manager):
        """测试告警管理器导入错误"""
        mock_alert_manager.side_effect = ImportError("Cannot import")
        
        result = self.factory.create_alert_manager()
        
        self.assertIsNone(result)
        self.logger.warning.assert_called_once_with("AlertManager组件不可用")

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.NotificationManager')
    def test_create_notification_manager_success(self, mock_notification_manager):
        """测试成功创建通知管理器"""
        mock_instance = Mock()
        mock_notification_manager.return_value = mock_instance
        
        result = self.factory.create_notification_manager({"channels": []})
        
        mock_notification_manager.assert_called_once_with({"channels": []})
        self.assertEqual(result, mock_instance)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.NotificationManager')
    def test_create_notification_manager_import_error(self, mock_notification_manager):
        """测试通知管理器导入错误"""
        mock_notification_manager.side_effect = ImportError("Cannot import")
        
        result = self.factory.create_notification_manager()
        
        self.assertIsNone(result)
        self.logger.warning.assert_called_once_with("NotificationManager组件不可用")

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.TestExecutionMonitor')
    def test_create_test_monitor_success(self, mock_test_monitor):
        """测试成功创建测试监控器"""
        mock_instance = Mock()
        mock_test_monitor.return_value = mock_instance
        
        result = self.factory.create_test_monitor({"monitor_interval": 30})
        
        mock_test_monitor.assert_called_once_with({"monitor_interval": 30})
        self.assertEqual(result, mock_instance)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.TestExecutionMonitor')
    def test_create_test_monitor_import_error(self, mock_test_monitor):
        """测试测试监控器导入错误"""
        mock_test_monitor.side_effect = ImportError("Cannot import")
        
        result = self.factory.create_test_monitor()
        
        self.assertIsNone(result)
        self.logger.warning.assert_called_once_with("TestExecutionMonitor组件不可用")

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.TestExecutionManager')
    def test_create_test_execution_manager_success(self, mock_test_execution_manager):
        """测试成功创建测试执行管理器"""
        mock_test_monitor = Mock()
        mock_instance = Mock()
        self.registry.create_component.return_value = mock_test_monitor
        mock_test_execution_manager.return_value = mock_instance
        
        result = self.factory.create_test_execution_manager({"config": "test"})
        
        self.registry.create_component.assert_called_once_with('test_monitor', {"config": "test"})
        mock_test_execution_manager.assert_called_once_with(mock_test_monitor, self.logger)
        self.assertEqual(result, mock_instance)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.TestExecutionManager')
    def test_create_test_execution_manager_import_error(self, mock_test_execution_manager):
        """测试测试执行管理器导入错误"""
        mock_test_execution_manager.side_effect = ImportError("Cannot import")
        
        result = self.factory.create_test_execution_manager()
        
        self.assertIsNone(result)
        self.logger.warning.assert_called_once_with("TestExecutionManager组件不可用")

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.CachedMetricsCollector')
    def test_create_metrics_collector_success(self, mock_metrics_collector):
        """测试成功创建指标收集器"""
        mock_performance_monitor = Mock()
        mock_instance = Mock()
        self.registry.create_component.return_value = mock_performance_monitor
        mock_metrics_collector.return_value = mock_instance
        
        result = self.factory.create_metrics_collector({"cache_size": 1000})
        
        self.registry.create_component.assert_called_once_with('performance_monitor', {"cache_size": 1000})
        mock_metrics_collector.assert_called_once_with(mock_performance_monitor, self.logger)
        self.assertEqual(result, mock_instance)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.CachedMetricsCollector')
    def test_create_metrics_collector_import_error(self, mock_metrics_collector):
        """测试指标收集器导入错误"""
        mock_metrics_collector.side_effect = ImportError("Cannot import")
        
        result = self.factory.create_metrics_collector()
        
        self.assertIsNone(result)
        self.logger.warning.assert_called_once_with("CachedMetricsCollector组件不可用")

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.AlertCoordinator')
    def test_create_alert_coordinator_success(self, mock_alert_coordinator):
        """测试成功创建告警协调器"""
        mock_alert_manager = Mock()
        mock_instance = Mock()
        self.registry.create_component.return_value = mock_alert_manager
        mock_alert_coordinator.return_value = mock_instance
        
        result = self.factory.create_alert_coordinator({"coordination": True})
        
        self.registry.create_component.assert_called_once_with('alert_manager', {"coordination": True})
        mock_alert_coordinator.assert_called_once_with(mock_alert_manager, self.logger)
        self.assertEqual(result, mock_instance)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.AlertCoordinator')
    def test_create_alert_coordinator_import_error(self, mock_alert_coordinator):
        """测试告警协调器导入错误"""
        mock_alert_coordinator.side_effect = ImportError("Cannot import")
        
        result = self.factory.create_alert_coordinator()
        
        self.assertIsNone(result)
        self.logger.warning.assert_called_once_with("AlertCoordinator组件不可用")

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.MonitoringSystemCoordinator')
    def test_create_monitoring_system_coordinator_success(self, mock_coordinator):
        """测试成功创建监控系统协调器"""
        mock_instance = Mock()
        mock_coordinator.return_value = mock_instance
        
        result = self.factory.create_monitoring_system_coordinator({"systems": []})
        
        mock_coordinator.assert_called_once_with({"systems": []}, self.logger)
        self.assertEqual(result, mock_instance)

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.MonitoringSystemCoordinator')
    def test_create_monitoring_system_coordinator_import_error(self, mock_coordinator):
        """测试监控系统协调器导入错误"""
        mock_coordinator.side_effect = ImportError("Cannot import")
        
        result = self.factory.create_monitoring_system_coordinator()
        
        self.assertIsNone(result)
        self.logger.warning.assert_called_once_with("MonitoringSystemCoordinator组件不可用")

    @patch('src.infrastructure.resource.monitoring.alerts.component_factories.MonitoringSystemCoordinator')
    def test_create_monitoring_system_coordinator_with_empty_config(self, mock_coordinator):
        """测试使用空配置创建监控系统协调器"""
        mock_instance = Mock()
        mock_coordinator.return_value = mock_instance
        
        result = self.factory.create_monitoring_system_coordinator(None)
        
        mock_coordinator.assert_called_once_with({}, self.logger)
        self.assertEqual(result, mock_instance)


if __name__ == '__main__':
    unittest.main()
