#!/usr/bin/env python3
"""
RQA2025 基础设施层监控协调器单元测试

测试 MonitoringCoordinator 的功能和行为。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
import time
from unittest.mock import Mock, patch
from datetime import datetime

from src.infrastructure.monitoring.services.monitoring_coordinator import MonitoringCoordinator


class TestMonitoringCoordinator(unittest.TestCase):
    """监控协调器测试类"""

    def setUp(self):
        """测试前准备"""
        self.coordinator = MonitoringCoordinator()
        self.mock_collector = Mock()
        self.mock_processor = Mock()
        self.mock_suggester = Mock()
        self.mock_manager = Mock()

    def tearDown(self):
        """测试后清理"""
        if self.coordinator.monitoring_active:
            self.coordinator.stop_monitoring()

    def test_initialization(self):
        """测试初始化"""
        self.assertFalse(self.coordinator.monitoring_active)
        self.assertIsNone(self.coordinator.monitoring_thread)
        self.assertIsNone(self.coordinator.start_time)
        self.assertIn('alert_thresholds', self.coordinator.config)
        self.assertEqual(self.coordinator.config['alert_thresholds']['cpu_usage_high'], 70)

    def test_set_components(self):
        """测试设置组件"""
        self.coordinator.set_components(
            metrics_collector=self.mock_collector,
            alert_processor=self.mock_processor,
            optimization_suggester=self.mock_suggester,
            data_manager=self.mock_manager
        )

        self.assertEqual(self.coordinator.metrics_collector, self.mock_collector)
        self.assertEqual(self.coordinator.alert_processor, self.mock_processor)
        self.assertEqual(self.coordinator.optimization_suggester, self.mock_suggester)
        self.assertEqual(self.coordinator.data_manager, self.mock_manager)

    def test_start_stop_monitoring(self):
        """测试启动和停止监控"""
        # 测试启动
        result = self.coordinator.start_monitoring()
        self.assertTrue(result)
        self.assertTrue(self.coordinator.monitoring_active)
        self.assertIsNotNone(self.coordinator.start_time)
        self.assertIsNotNone(self.coordinator.monitoring_thread)
        self.assertTrue(self.coordinator.monitoring_thread.is_alive())

        # 测试重复启动
        result2 = self.coordinator.start_monitoring()
        self.assertFalse(result2)  # 应该返回False

        # 测试停止
        result3 = self.coordinator.stop_monitoring()
        self.assertTrue(result3)
        self.assertFalse(self.coordinator.monitoring_active)

    def test_update_config(self):
        """测试更新配置"""
        new_config = {
            'interval_seconds': 120,
            'max_history_items': 2000
        }

        self.coordinator.update_config(new_config)

        self.assertEqual(self.coordinator.config['interval_seconds'], 120)
        self.assertEqual(self.coordinator.config['max_history_items'], 2000)
        # 原有配置应该保留
        self.assertEqual(self.coordinator.config['alert_thresholds']['cpu_usage_high'], 70)

    def test_get_monitoring_status(self):
        """测试获取监控状态"""
        status = self.coordinator.get_monitoring_status()

        expected_keys = [
            'active', 'start_time', 'uptime_seconds',
            'config', 'stats', 'components_status'
        ]

        for key in expected_keys:
            self.assertIn(key, status)

        self.assertFalse(status['active'])
        self.assertIsNone(status['start_time'])

        # 检查组件状态
        components_status = status['components_status']
        self.assertFalse(components_status['metrics_collector'])
        self.assertFalse(components_status['alert_processor'])
        self.assertFalse(components_status['optimization_suggester'])
        self.assertFalse(components_status['data_manager'])

    def test_get_health_status_healthy(self):
        """测试获取健康状态 - 健康"""
        health = self.coordinator.get_health_status()

        self.assertIn('status', health)
        self.assertIn('issues', health)
        self.assertIn('health_score', health)

        # 没有组件时应该是警告状态
        self.assertEqual(health['status'], 'warning')
        self.assertGreater(len(health['issues']), 0)

    def test_reset_stats(self):
        """测试重置统计"""
        # 先设置一些统计数据
        self.coordinator.monitoring_stats['cycles_completed'] = 10
        self.coordinator.monitoring_stats['alerts_generated'] = 5

        self.coordinator.reset_stats()

        self.assertEqual(self.coordinator.monitoring_stats['cycles_completed'], 0)
        self.assertEqual(self.coordinator.monitoring_stats['alerts_generated'], 0)
        self.assertEqual(self.coordinator.monitoring_stats['errors_encountered'], 0)

    def test_force_monitoring_cycle_without_components(self):
        """测试强制执行监控周期（无组件）"""
        result = self.coordinator.force_monitoring_cycle()

        # 即使没有组件，执行也应该成功，只是记录警告
        self.assertTrue(result['success'])
        self.assertIn('监控周期执行完成', result['message'])

    @patch('time.sleep')  # 避免测试中实际等待
    def test_force_monitoring_cycle_with_components(self, mock_sleep):
        """测试强制执行监控周期（有组件）"""
        # 设置模拟组件
        self.mock_collector.collect_all_metrics.return_value = {'test': 'data'}
        self.mock_processor.process_alerts.return_value = [{'id': 'test_alert'}]
        self.mock_suggester.generate_suggestions.return_value = [{'type': 'test'}]
        self.mock_manager.save_monitoring_data.return_value = None

        self.coordinator.set_components(
            metrics_collector=self.mock_collector,
            alert_processor=self.mock_processor,
            optimization_suggester=self.mock_suggester,
            data_manager=self.mock_manager
        )

        result = self.coordinator.force_monitoring_cycle()

        self.assertTrue(result['success'])
        self.assertIn('监控周期执行完成', result['message'])

        # 验证组件被调用
        self.mock_collector.collect_all_metrics.assert_called_once()
        self.mock_processor.process_alerts.assert_called_once()
        self.mock_suggester.generate_suggestions.assert_called_once()
        self.mock_manager.save_monitoring_data.assert_called_once()

    def test_monitoring_thread_creation(self):
        """测试监控线程创建"""
        self.coordinator.start_monitoring()

        self.assertIsNotNone(self.coordinator.monitoring_thread)
        self.assertTrue(self.coordinator.monitoring_thread.is_alive())
        self.assertEqual(self.coordinator.monitoring_thread.name, "MonitoringCoordinator")

        self.coordinator.stop_monitoring()


if __name__ == '__main__':
    unittest.main()