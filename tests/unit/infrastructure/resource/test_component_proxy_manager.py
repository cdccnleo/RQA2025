"""
component_proxy_manager.py 测试模块

测试组件代理管理器的功能，提升覆盖率。
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
    from src.infrastructure.resource.monitoring.alerts.component_proxy_manager import ComponentProxyManager
    IMPORT_ERROR = None
except ImportError as e:
    ComponentProxyManager = None
    IMPORT_ERROR = str(e)


class TestComponentProxyManager(unittest.TestCase):
    """测试组件代理管理器类"""

    def setUp(self):
        """设置测试环境"""
        if ComponentProxyManager is None:
            self.skipTest(f"Cannot import ComponentProxyManager: {IMPORT_ERROR}")
        
        # 创建mock对象
        self.mock_registry = Mock()
        self.mock_logger = Mock(spec=logging.Logger)
        
        self.proxy_manager = ComponentProxyManager(self.mock_registry, self.mock_logger)

    def test_component_proxy_manager_initialization(self):
        """测试组件代理管理器初始化"""
        self.assertIsNotNone(self.proxy_manager)
        self.assertEqual(self.proxy_manager.registry, self.mock_registry)
        self.assertEqual(self.proxy_manager.logger, self.mock_logger)
        self.assertIsInstance(self.proxy_manager._components, dict)
        self.assertEqual(len(self.proxy_manager._components), 0)

    def test_get_alert_rule_manager_success(self):
        """测试成功获取告警规则管理器"""
        mock_alert_manager = Mock()
        mock_rule_manager = Mock()
        
        self.mock_registry.create_component.return_value = mock_alert_manager
        
        # Mock AlertRuleManager类
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.AlertRuleManager') as mock_rule_class:
            mock_rule_class.return_value = mock_rule_manager
            
            result = self.proxy_manager.get_alert_rule_manager()
            
            self.assertEqual(result, mock_rule_manager)
            self.mock_registry.create_component.assert_called_once_with('alert_manager')
            mock_rule_class.assert_called_once_with(mock_alert_manager, self.mock_logger)

    def test_get_alert_rule_manager_registry_returns_none(self):
        """测试注册表返回None的情况"""
        self.mock_registry.create_component.return_value = None
        
        result = self.proxy_manager.get_alert_rule_manager()
        
        self.assertIsNone(result)
        self.mock_registry.create_component.assert_called_once_with('alert_manager')

    def test_get_alert_rule_manager_import_error(self):
        """测试导入错误的情况"""
        mock_alert_manager = Mock()
        self.mock_registry.create_component.return_value = mock_alert_manager
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.AlertRuleManager', side_effect=ImportError("Cannot import")):
            result = self.proxy_manager.get_alert_rule_manager()
            
            self.assertIsNone(result)
            self.mock_logger.error.assert_called_once_with("AlertRuleManager不可用")

    def test_get_alert_rule_manager_cached(self):
        """测试缓存的告警规则管理器"""
        mock_rule_manager = Mock()
        self.proxy_manager._components['alert_rule_manager'] = mock_rule_manager
        
        result = self.proxy_manager.get_alert_rule_manager()
        
        self.assertEqual(result, mock_rule_manager)
        # 验证没有调用注册表，因为已缓存
        self.mock_registry.create_component.assert_not_called()

    def test_get_notification_channel_manager_success(self):
        """测试成功获取通知渠道管理器"""
        mock_notification_manager = Mock()
        mock_channel_manager = Mock()
        
        self.mock_registry.create_component.return_value = mock_notification_manager
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.NotificationChannelManager') as mock_channel_class:
            mock_channel_class.return_value = mock_channel_manager
            
            result = self.proxy_manager.get_notification_channel_manager()
            
            self.assertEqual(result, mock_channel_manager)
            self.mock_registry.create_component.assert_called_once_with('notification_manager')
            mock_channel_class.assert_called_once_with(mock_notification_manager, self.mock_logger)

    def test_get_notification_channel_manager_registry_returns_none(self):
        """测试通知渠道管理器注册表返回None"""
        self.mock_registry.create_component.return_value = None
        
        result = self.proxy_manager.get_notification_channel_manager()
        
        self.assertIsNone(result)

    def test_get_notification_channel_manager_import_error(self):
        """测试通知渠道管理器导入错误"""
        mock_notification_manager = Mock()
        self.mock_registry.create_component.return_value = mock_notification_manager
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.NotificationChannelManager', side_effect=ImportError("Cannot import")):
            result = self.proxy_manager.get_notification_channel_manager()
            
            self.assertIsNone(result)
            self.mock_logger.error.assert_called_once_with("NotificationChannelManager不可用")

    def test_get_test_execution_manager_success(self):
        """测试成功获取测试执行管理器"""
        mock_test_monitor = Mock()
        mock_execution_manager = Mock()
        
        self.mock_registry.create_component.return_value = mock_test_monitor
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.TestExecutionManager') as mock_execution_class:
            mock_execution_class.return_value = mock_execution_manager
            
            result = self.proxy_manager.get_test_execution_manager()
            
            self.assertEqual(result, mock_execution_manager)
            self.mock_registry.create_component.assert_called_once_with('test_monitor')
            mock_execution_class.assert_called_once_with(mock_test_monitor, self.mock_logger)

    def test_get_test_execution_manager_registry_returns_none(self):
        """测试测试执行管理器注册表返回None"""
        self.mock_registry.create_component.return_value = None
        
        result = self.proxy_manager.get_test_execution_manager()
        
        self.assertIsNone(result)

    def test_get_test_execution_manager_import_error(self):
        """测试测试执行管理器导入错误"""
        mock_test_monitor = Mock()
        self.mock_registry.create_component.return_value = mock_test_monitor
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.TestExecutionManager', side_effect=ImportError("Cannot import")):
            result = self.proxy_manager.get_test_execution_manager()
            
            self.assertIsNone(result)
            self.mock_logger.error.assert_called_once_with("TestExecutionManager不可用")

    def test_get_metrics_collector_success(self):
        """测试成功获取指标收集器"""
        mock_performance_monitor = Mock()
        mock_metrics_collector = Mock()
        
        self.mock_registry.create_component.return_value = mock_performance_monitor
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.CachedMetricsCollector') as mock_collector_class:
            mock_collector_class.return_value = mock_metrics_collector
            
            result = self.proxy_manager.get_metrics_collector()
            
            self.assertEqual(result, mock_metrics_collector)
            self.mock_registry.create_component.assert_called_once_with('performance_monitor')
            mock_collector_class.assert_called_once_with(mock_performance_monitor, self.mock_logger)

    def test_get_metrics_collector_registry_returns_none(self):
        """测试指标收集器注册表返回None"""
        self.mock_registry.create_component.return_value = None
        
        result = self.proxy_manager.get_metrics_collector()
        
        self.assertIsNone(result)

    def test_get_metrics_collector_import_error(self):
        """测试指标收集器导入错误"""
        mock_performance_monitor = Mock()
        self.mock_registry.create_component.return_value = mock_performance_monitor
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.CachedMetricsCollector', side_effect=ImportError("Cannot import")):
            result = self.proxy_manager.get_metrics_collector()
            
            self.assertIsNone(result)
            self.mock_logger.error.assert_called_once_with("CachedMetricsCollector不可用")

    def test_get_alert_coordinator_success(self):
        """测试成功获取告警协调器"""
        mock_alert_manager = Mock()
        mock_alert_coordinator = Mock()
        
        self.mock_registry.create_component.return_value = mock_alert_manager
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.AlertCoordinator') as mock_coordinator_class:
            mock_coordinator_class.return_value = mock_alert_coordinator
            
            result = self.proxy_manager.get_alert_coordinator()
            
            self.assertEqual(result, mock_alert_coordinator)
            self.mock_registry.create_component.assert_called_once_with('alert_manager')
            mock_coordinator_class.assert_called_once_with(mock_alert_manager, self.mock_logger)

    def test_get_alert_coordinator_registry_returns_none(self):
        """测试告警协调器注册表返回None"""
        self.mock_registry.create_component.return_value = None
        
        result = self.proxy_manager.get_alert_coordinator()
        
        self.assertIsNone(result)

    def test_get_alert_coordinator_import_error(self):
        """测试告警协调器导入错误"""
        mock_alert_manager = Mock()
        self.mock_registry.create_component.return_value = mock_alert_manager
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.AlertCoordinator', side_effect=ImportError("Cannot import")):
            result = self.proxy_manager.get_alert_coordinator()
            
            self.assertIsNone(result)
            self.mock_logger.error.assert_called_once_with("AlertCoordinator不可用")

    def test_component_caching_behavior(self):
        """测试组件缓存行为"""
        mock_alert_manager = Mock()
        mock_rule_manager = Mock()
        
        self.mock_registry.create_component.return_value = mock_alert_manager
        
        with patch('src.infrastructure.resource.monitoring.alerts.component_proxy_manager.AlertRuleManager') as mock_rule_class:
            mock_rule_class.return_value = mock_rule_manager
            
            # 第一次调用
            result1 = self.proxy_manager.get_alert_rule_manager()
            
            # 第二次调用应该返回缓存的组件
            result2 = self.proxy_manager.get_alert_rule_manager()
            
            self.assertEqual(result1, result2)
            self.assertEqual(result1, mock_rule_manager)
            
            # 验证注册表只被调用一次
            self.mock_registry.create_component.assert_called_once_with('alert_manager')
            mock_rule_class.assert_called_once()


if __name__ == '__main__':
    unittest.main()
