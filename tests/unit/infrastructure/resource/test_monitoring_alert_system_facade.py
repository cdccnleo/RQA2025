"""
monitoring_alert_system_facade.py 测试模块

测试监控告警系统门面类的功能，提升覆盖率。
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
    from src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade import (
        LegacyMonitoringAlertSystemFacade
    )
except ImportError as e:
    # 如果导入失败，我们跳过这些测试
    LegacyMonitoringAlertSystemFacade = None
    IMPORT_ERROR = str(e)


class TestLegacyMonitoringAlertSystemFacade(unittest.TestCase):
    """测试监控告警系统门面类"""

    def setUp(self):
        """设置测试环境"""
        if LegacyMonitoringAlertSystemFacade is None:
            self.skipTest(f"Cannot import LegacyMonitoringAlertSystemFacade: {IMPORT_ERROR}")
        
        # Mock所有依赖的组件
        with patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.ComponentFactoryRegistry') as mock_registry, \
             patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.AlertSystemComponentFactory') as mock_factory, \
             patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.ComponentProxyManager') as mock_proxy, \
             patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.SystemOperationsManager') as mock_ops:
            
            # 设置mock返回值
            self.mock_factory_registry = Mock()
            self.mock_component_factory = Mock()
            self.mock_component_proxy = Mock()
            self.mock_system_ops = Mock()
            
            mock_registry.return_value = self.mock_factory_registry
            mock_factory.return_value = self.mock_component_factory
            mock_proxy.return_value = self.mock_component_proxy
            mock_ops.return_value = self.mock_system_ops
            
            self.facade = LegacyMonitoringAlertSystemFacade()

    def test_facade_initialization(self):
        """测试门面类初始化"""
        self.assertIsNotNone(self.facade)
        self.assertIsNotNone(self.facade.logger)
        self.assertIsInstance(self.facade.config, dict)

    def test_facade_initialization_with_config(self):
        """测试使用配置初始化门面类"""
        config = {"test": "config"}
        with patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.ComponentFactoryRegistry'), \
             patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.AlertSystemComponentFactory'), \
             patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.ComponentProxyManager'), \
             patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.SystemOperationsManager') as mock_ops:
            
            facade = LegacyMonitoringAlertSystemFacade(config=config)
            self.assertEqual(facade.config, config)

    def test_facade_initialization_with_logger(self):
        """测试使用自定义日志器初始化门面类"""
        with patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.ComponentFactoryRegistry'), \
             patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.AlertSystemComponentFactory'), \
             patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.ComponentProxyManager'), \
             patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.SystemOperationsManager'):
            
            facade = LegacyMonitoringAlertSystemFacade(logger="test_logger")
            # 验证logger被设置
            self.assertIsNotNone(facade.logger)

    def test_component_management_initialization(self):
        """测试组件管理初始化"""
        # 验证组件管理相关属性被初始化
        self.assertIsNotNone(self.facade.factory_registry)
        self.assertIsNotNone(self.facade.component_factory)
        self.assertIsNotNone(self.facade.component_proxy)
        self.assertIsNotNone(self.facade.system_ops)

    def test_add_alert_rule(self):
        """测试添加告警规则"""
        mock_rule = Mock()
        mock_manager = Mock()
        mock_manager.add_alert_rule.return_value = True
        self.mock_component_proxy.get_alert_rule_manager.return_value = mock_manager
        
        result = self.facade.add_alert_rule(mock_rule)
        
        self.assertTrue(result)
        mock_manager.add_alert_rule.assert_called_once_with(mock_rule)

    def test_add_alert_rule_no_manager(self):
        """测试添加告警规则 - 无管理器"""
        mock_rule = Mock()
        self.mock_component_proxy.get_alert_rule_manager.return_value = None
        
        result = self.facade.add_alert_rule(mock_rule)
        
        self.assertFalse(result)

    def test_remove_alert_rule(self):
        """测试移除告警规则"""
        mock_manager = Mock()
        mock_manager.remove_alert_rule.return_value = True
        self.mock_component_proxy.get_alert_rule_manager.return_value = mock_manager
        
        result = self.facade.remove_alert_rule("test_rule")
        
        self.assertTrue(result)
        mock_manager.remove_alert_rule.assert_called_once_with("test_rule")

    def test_remove_alert_rule_no_manager(self):
        """测试移除告警规则 - 无管理器"""
        self.mock_component_proxy.get_alert_rule_manager.return_value = None
        
        result = self.facade.remove_alert_rule("test_rule")
        
        self.assertFalse(result)

    def test_get_alert_rules(self):
        """测试获取告警规则"""
        mock_rules = [Mock(), Mock()]
        mock_manager = Mock()
        mock_manager.get_alert_rules.return_value = mock_rules
        self.mock_component_proxy.get_alert_rule_manager.return_value = mock_manager
        
        result = self.facade.get_alert_rules()
        
        self.assertEqual(result, mock_rules)
        mock_manager.get_alert_rules.assert_called_once()

    def test_get_alert_rules_no_manager(self):
        """测试获取告警规则 - 无管理器"""
        self.mock_component_proxy.get_alert_rule_manager.return_value = None
        
        result = self.facade.get_alert_rules()
        
        self.assertEqual(result, [])

    def test_configure_notification_channel(self):
        """测试配置通知渠道"""
        mock_manager = Mock()
        mock_manager.configure_notification_channel.return_value = True
        self.mock_component_proxy.get_notification_channel_manager.return_value = mock_manager
        
        result = self.facade.configure_notification_channel("test_channel", {"key": "value"})
        
        self.assertTrue(result)
        mock_manager.configure_notification_channel.assert_called_once_with("test_channel", {"key": "value"})

    def test_configure_notification_channel_no_manager(self):
        """测试配置通知渠道 - 无管理器"""
        self.mock_component_proxy.get_notification_channel_manager.return_value = None
        
        result = self.facade.configure_notification_channel("test_channel", {"key": "value"})
        
        self.assertFalse(result)

    def test_get_notification_channels(self):
        """测试获取通知渠道"""
        mock_channels = {"channel1": True, "channel2": False}
        mock_manager = Mock()
        mock_manager.get_notification_channels.return_value = mock_channels
        self.mock_component_proxy.get_notification_channel_manager.return_value = mock_manager
        
        result = self.facade.get_notification_channels()
        
        self.assertEqual(result, mock_channels)
        mock_manager.get_notification_channels.assert_called_once()

    def test_get_notification_channels_no_manager(self):
        """测试获取通知渠道 - 无管理器"""
        self.mock_component_proxy.get_notification_channel_manager.return_value = None
        
        result = self.facade.get_notification_channels()
        
        self.assertEqual(result, {})

    def test_register_test(self):
        """测试注册测试"""
        mock_manager = Mock()
        mock_manager.register_test.return_value = "test_id"
        self.mock_component_proxy.get_test_execution_manager.return_value = mock_manager
        
        result = self.facade.register_test("test_id", "test_name", {"metadata": "value"})
        
        self.assertEqual(result, "test_id")
        mock_manager.register_test.assert_called_once_with("test_id", "test_name", {"metadata": "value"})

    def test_register_test_no_manager(self):
        """测试注册测试 - 无管理器"""
        self.mock_component_proxy.get_test_execution_manager.return_value = None
        
        result = self.facade.register_test("test_id", "test_name")
        
        self.assertEqual(result, "")

    def test_update_test_status(self):
        """测试更新测试状态"""
        mock_manager = Mock()
        mock_manager.update_test_status.return_value = True
        self.mock_component_proxy.get_test_execution_manager.return_value = mock_manager
        
        result = self.facade.update_test_status("test_id", "running", "message", {"metric": "value"})
        
        self.assertTrue(result)
        mock_manager.update_test_status.assert_called_once_with("test_id", "running", "message", {"metric": "value"})

    def test_update_test_status_no_manager(self):
        """测试更新测试状态 - 无管理器"""
        self.mock_component_proxy.get_test_execution_manager.return_value = None
        
        result = self.facade.update_test_status("test_id", "running")
        
        self.assertFalse(result)

    def test_get_active_tests(self):
        """测试获取活跃测试"""
        mock_tests = [Mock(), Mock()]
        mock_manager = Mock()
        mock_manager.get_active_tests.return_value = mock_tests
        self.mock_component_proxy.get_test_execution_manager.return_value = mock_manager
        
        result = self.facade.get_active_tests()
        
        self.assertEqual(result, mock_tests)
        mock_manager.get_active_tests.assert_called_once()

    def test_get_active_tests_no_manager(self):
        """测试获取活跃测试 - 无管理器"""
        self.mock_component_proxy.get_test_execution_manager.return_value = None
        
        result = self.facade.get_active_tests()
        
        self.assertEqual(result, [])

    def test_get_test_history(self):
        """测试获取测试历史"""
        mock_history = [Mock(), Mock()]
        mock_manager = Mock()
        mock_manager.get_test_history.return_value = mock_history
        self.mock_component_proxy.get_test_execution_manager.return_value = mock_manager
        
        result = self.facade.get_test_history(48)
        
        self.assertEqual(result, mock_history)
        mock_manager.get_test_history.assert_called_once_with(48)

    def test_get_test_history_no_manager(self):
        """测试获取测试历史 - 无管理器"""
        self.mock_component_proxy.get_test_execution_manager.return_value = None
        
        result = self.facade.get_test_history()
        
        self.assertEqual(result, [])

    def test_get_current_metrics(self):
        """测试获取当前指标"""
        mock_metrics = Mock()
        mock_collector = Mock()
        mock_collector.get_current_metrics.return_value = mock_metrics
        self.mock_component_proxy.get_metrics_collector.return_value = mock_collector
        
        result = self.facade.get_current_metrics()
        
        self.assertEqual(result, mock_metrics)
        mock_collector.get_current_metrics.assert_called_once()

    def test_get_current_metrics_no_collector(self):
        """测试获取当前指标 - 无收集器"""
        self.mock_component_proxy.get_metrics_collector.return_value = None
        
        # 需要mock AlertPerformanceMetrics类
        with patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.AlertPerformanceMetrics') as mock_metrics_class:
            mock_empty_metrics = Mock()
            mock_metrics_class.return_value = mock_empty_metrics
            
            result = self.facade.get_current_metrics()
            
            self.assertEqual(result, mock_empty_metrics)

    def test_get_metrics_history(self):
        """测试获取指标历史"""
        mock_history = [Mock(), Mock()]
        mock_collector = Mock()
        mock_collector.get_metrics_history.return_value = mock_history
        self.mock_component_proxy.get_metrics_collector.return_value = mock_collector
        
        result = self.facade.get_metrics_history(120)
        
        self.assertEqual(result, mock_history)
        mock_collector.get_metrics_history.assert_called_once_with(120)

    def test_get_metrics_history_no_collector(self):
        """测试获取指标历史 - 无收集器"""
        self.mock_component_proxy.get_metrics_collector.return_value = None
        
        result = self.facade.get_metrics_history()
        
        self.assertEqual(result, [])

    def test_get_average_metrics(self):
        """测试获取平均指标"""
        mock_avg_metrics = Mock()
        mock_collector = Mock()
        mock_collector.get_average_metrics.return_value = mock_avg_metrics
        self.mock_component_proxy.get_metrics_collector.return_value = mock_collector
        
        result = self.facade.get_average_metrics(180)
        
        self.assertEqual(result, mock_avg_metrics)
        mock_collector.get_average_metrics.assert_called_once_with(180)

    def test_get_average_metrics_no_collector(self):
        """测试获取平均指标 - 无收集器"""
        self.mock_component_proxy.get_metrics_collector.return_value = None
        
        # 需要mock AlertPerformanceMetrics类
        with patch('src.infrastructure.resource.monitoring.alerts.monitoring_alert_system_facade.AlertPerformanceMetrics') as mock_metrics_class:
            mock_empty_metrics = Mock()
            mock_metrics_class.return_value = mock_empty_metrics
            
            result = self.facade.get_average_metrics()
            
            self.assertEqual(result, mock_empty_metrics)

    def test_get_active_alerts(self):
        """测试获取活跃告警"""
        mock_alerts = [Mock(), Mock()]
        mock_coordinator = Mock()
        mock_coordinator.get_active_alerts.return_value = mock_alerts
        self.mock_component_proxy.get_alert_coordinator.return_value = mock_coordinator
        
        result = self.facade.get_active_alerts()
        
        self.assertEqual(result, mock_alerts)
        mock_coordinator.get_active_alerts.assert_called_once()

    def test_get_active_alerts_no_coordinator(self):
        """测试获取活跃告警 - 无协调器"""
        self.mock_component_proxy.get_alert_coordinator.return_value = None
        
        result = self.facade.get_active_alerts()
        
        self.assertEqual(result, [])

    def test_resolve_alert(self):
        """测试解决告警"""
        mock_coordinator = Mock()
        mock_coordinator.resolve_alert.return_value = True
        self.mock_component_proxy.get_alert_coordinator.return_value = mock_coordinator
        
        result = self.facade.resolve_alert("alert_id")
        
        self.assertTrue(result)
        mock_coordinator.resolve_alert.assert_called_once_with("alert_id")

    def test_resolve_alert_no_coordinator(self):
        """测试解决告警 - 无协调器"""
        self.mock_component_proxy.get_alert_coordinator.return_value = None
        
        result = self.facade.resolve_alert("alert_id")
        
        self.assertFalse(result)

    def test_get_alert_statistics(self):
        """测试获取告警统计"""
        mock_stats = {"total": 10, "active": 3}
        mock_coordinator = Mock()
        mock_coordinator.get_alert_statistics.return_value = mock_stats
        self.mock_component_proxy.get_alert_coordinator.return_value = mock_coordinator
        
        result = self.facade.get_alert_statistics()
        
        self.assertEqual(result, mock_stats)
        mock_coordinator.get_alert_statistics.assert_called_once()

    def test_get_alert_statistics_no_coordinator(self):
        """测试获取告警统计 - 无协调器"""
        self.mock_component_proxy.get_alert_coordinator.return_value = None
        
        result = self.facade.get_alert_statistics()
        
        self.assertEqual(result, {})

    def test_start(self):
        """测试启动系统"""
        self.mock_system_ops.start_system.return_value = True
        
        result = self.facade.start()
        
        self.assertTrue(result)
        self.mock_system_ops.start_system.assert_called_once()

    def test_stop(self):
        """测试停止系统"""
        self.mock_system_ops.stop_system.return_value = True
        
        result = self.facade.stop()
        
        self.assertTrue(result)
        self.mock_system_ops.stop_system.assert_called_once()

    def test_get_system_status(self):
        """测试获取系统状态"""
        mock_status = {"status": "running", "uptime": 3600}
        self.mock_system_ops.get_system_status.return_value = mock_status
        
        result = self.facade.get_system_status()
        
        self.assertEqual(result, mock_status)
        self.mock_system_ops.get_system_status.assert_called_once()

    def test_get_system_health_report(self):
        """测试获取系统健康报告"""
        mock_health_report = {"health": "good", "issues": []}
        self.mock_system_ops.get_system_health_report.return_value = mock_health_report
        
        result = self.facade.get_system_health_report()
        
        self.assertEqual(result, mock_health_report)
        self.mock_system_ops.get_system_health_report.assert_called_once()

    def test_update_configuration(self):
        """测试更新配置"""
        new_config = {"new_key": "new_value"}
        self.mock_system_ops.update_configuration.return_value = True
        
        result = self.facade.update_configuration(new_config)
        
        self.assertTrue(result)
        self.assertIn("new_key", self.facade.config)
        self.mock_system_ops.update_configuration.assert_called_once_with(new_config)

    def test_get_configuration(self):
        """测试获取配置"""
        original_config = {"key": "value"}
        self.facade.config = original_config.copy()
        
        result = self.facade.get_configuration()
        
        self.assertEqual(result, original_config)
        # 验证返回的是副本
        result["new_key"] = "new_value"
        self.assertNotIn("new_key", self.facade.config)


if __name__ == '__main__':
    unittest.main()