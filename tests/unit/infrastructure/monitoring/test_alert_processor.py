#!/usr/bin/env python3
"""
RQA2025 基础设施层告警处理器单元测试

测试 AlertProcessor 的功能和告警处理逻辑。
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from src.infrastructure.monitoring.services.alert_processor import (
    AlertProcessor, AlertSeverity, AlertStatus
)


class TestAlertProcessor(unittest.TestCase):
    """告警处理器测试类"""

    def setUp(self):
        """测试前准备"""
        self.processor = AlertProcessor()

    def test_initialization(self):
        """测试初始化"""
        self.assertIsInstance(self.processor.alert_thresholds, dict)
        self.assertIn('cpu_usage_high', self.processor.alert_thresholds)
        self.assertEqual(self.processor.alert_thresholds['cpu_usage_high'], 70)
        self.assertEqual(len(self.processor.active_alerts), 0)
        self.assertEqual(len(self.processor.alert_history), 0)

    def test_initialization_with_custom_thresholds(self):
        """测试使用自定义阈值初始化"""
        custom_thresholds = {
            'cpu_usage_high': 80,
            'memory_usage_high': 85
        }
        processor = AlertProcessor(custom_thresholds)

        self.assertEqual(processor.alert_thresholds['cpu_usage_high'], 80)
        self.assertEqual(processor.alert_thresholds['memory_usage_high'], 85)

    def test_process_alerts_empty_metrics(self):
        """测试处理空指标"""
        alerts = self.processor.process_alerts({})

        self.assertIsInstance(alerts, list)
        self.assertEqual(len(alerts), 0)

    def test_process_alerts_cpu_high(self):
        """测试CPU使用率过高告警"""
        metrics = {
            'system_metrics': {
                'cpu': {'usage_percent': 85}
            }
        }

        alerts = self.processor.process_alerts(metrics)

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]

        self.assertEqual(alert['type'], 'cpu_usage_high')
        self.assertEqual(alert['severity'], AlertSeverity.CRITICAL.value)
        self.assertIn('CPU使用率过高', alert['title'])
        self.assertIn('85', alert['description'])
        self.assertEqual(alert['current_value'], 85)
        self.assertEqual(alert['threshold'], 70)

    def test_process_alerts_memory_high(self):
        """测试内存使用率过高告警"""
        metrics = {
            'system_metrics': {
                'memory': {'usage_percent': 95}
            }
        }

        alerts = self.processor.process_alerts(metrics)

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]

        self.assertEqual(alert['type'], 'memory_usage_high')
        self.assertEqual(alert['severity'], AlertSeverity.CRITICAL.value)
        self.assertIn('内存使用率过高', alert['title'])

    def test_process_alerts_disk_high(self):
        """测试磁盘使用率过高告警"""
        metrics = {
            'system_metrics': {
                'disk': {'usage_percent': 96}
            }
        }

        alerts = self.processor.process_alerts(metrics)

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]

        self.assertEqual(alert['type'], 'disk_usage_high')
        self.assertEqual(alert['severity'], AlertSeverity.CRITICAL.value)
        self.assertIn('磁盘使用率过高', alert['title'])

    def test_process_alerts_response_time_high(self):
        """测试响应时间过高告警"""
        metrics = {
            'performance_metrics': {
                'response_time_ms': 1500
            }
        }

        alerts = self.processor.process_alerts(metrics)

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]

        self.assertEqual(alert['type'], 'response_time_high')
        self.assertEqual(alert['severity'], AlertSeverity.WARNING.value)
        self.assertIn('响应时间过高', alert['title'])

    def test_process_alerts_error_rate_high(self):
        """测试错误率过高告警"""
        metrics = {
            'performance_metrics': {
                'error_rate_percent': 8.0
            }
        }

        alerts = self.processor.process_alerts(metrics)

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]

        self.assertEqual(alert['type'], 'error_rate_high')
        self.assertEqual(alert['severity'], AlertSeverity.ERROR.value)
        self.assertIn('错误率过高', alert['title'])

    def test_process_alerts_coverage_low(self):
        """测试覆盖率过低告警"""
        metrics = {
            'test_coverage_metrics': {
                'overall_coverage': 65.0
            }
        }

        alerts = self.processor.process_alerts(metrics)

        self.assertEqual(len(alerts), 1)
        alert = alerts[0]

        self.assertEqual(alert['type'], 'coverage_low')
        self.assertEqual(alert['severity'], AlertSeverity.WARNING.value)
        self.assertIn('测试覆盖率过低', alert['title'])

    def test_alert_lifecycle(self):
        """测试告警生命周期"""
        # 创建告警
        metrics = {
            'system_metrics': {
                'cpu': {'usage_percent': 90}
            }
        }

        alerts = self.processor.process_alerts(metrics)
        self.assertEqual(len(alerts), 1)

        alert_id = alerts[0]['id']
        self.assertIn(alert_id, self.processor.active_alerts)

        # 确认告警
        result = self.processor.acknowledge_alert(alert_id)
        self.assertTrue(result)

        alert = self.processor.active_alerts[alert_id]
        self.assertEqual(alert['status'], AlertStatus.ACKNOWLEDGED.value)
        self.assertIsNotNone(alert['acknowledged_at'])

        # 解决告警
        result = self.processor.resolve_alert(alert_id)
        self.assertTrue(result)

        self.assertNotIn(alert_id, self.processor.active_alerts)

        # 检查历史记录
        self.assertEqual(len(self.processor.alert_history), 1)
        history_alert = self.processor.alert_history[0]
        self.assertEqual(history_alert['id'], alert_id)
        self.assertEqual(history_alert['status'], AlertStatus.RESOLVED.value)

    def test_acknowledge_nonexistent_alert(self):
        """测试确认不存在的告警"""
        result = self.processor.acknowledge_alert("nonexistent_id")
        self.assertFalse(result)

    def test_resolve_nonexistent_alert(self):
        """测试解决不存在的告警"""
        result = self.processor.resolve_alert("nonexistent_id")
        self.assertFalse(result)

    def test_get_active_alerts(self):
        """测试获取活跃告警"""
        # 创建一些告警
        metrics = {
            'system_metrics': {
                'cpu': {'usage_percent': 85},
                'memory': {'usage_percent': 90}
            }
        }

        self.processor.process_alerts(metrics)

        active_alerts = self.processor.get_active_alerts()
        self.assertEqual(len(active_alerts), 2)

        # 所有活跃告警的状态都应该是ACTIVE
        for alert in active_alerts:
            self.assertEqual(alert['status'], AlertStatus.ACTIVE.value)

    def test_get_alert_history(self):
        """测试获取告警历史"""
        # 创建并解决一个告警
        metrics = {'system_metrics': {'cpu': {'usage_percent': 85}}}
        alerts = self.processor.process_alerts(metrics)
        alert_id = alerts[0]['id']

        self.processor.acknowledge_alert(alert_id)
        self.processor.resolve_alert(alert_id)

        # 获取历史记录
        history = self.processor.get_alert_history()
        self.assertEqual(len(history), 1)

        # 获取限制数量的历史记录
        limited_history = self.processor.get_alert_history(limit=0)
        self.assertEqual(len(limited_history), 1)

    def test_get_alert_stats(self):
        """测试获取告警统计"""
        # 创建多个告警
        metrics = {
            'system_metrics': {
                'cpu': {'usage_percent': 85},  # WARNING
                'memory': {'usage_percent': 95}  # CRITICAL
            },
            'performance_metrics': {
                'error_rate_percent': 8.0  # ERROR
            }
        }

        self.processor.process_alerts(metrics)

        stats = self.processor.get_alert_stats()

        self.assertEqual(stats['total_generated'], 3)
        self.assertEqual(stats['active_count'], 3)
        self.assertEqual(stats['resolved_count'], 0)
        self.assertEqual(stats['acknowledged_count'], 0)

        # 检查严重程度统计
        self.assertEqual(stats['by_severity'][AlertSeverity.WARNING.value], 1)
        self.assertEqual(stats['by_severity'][AlertSeverity.CRITICAL.value], 1)
        self.assertEqual(stats['by_severity'][AlertSeverity.ERROR.value], 1)

    def test_history_updates_when_acknowledged_and_resolved(self):
        """测试确认和解决告警时历史记录更新"""
        metrics = {'system_metrics': {'cpu': {'usage_percent': 85}}}
        alerts = self.processor.process_alerts(metrics)
        alert_id = alerts[0]['id']

        self.processor.acknowledge_alert(alert_id)
        history_after_ack = self.processor.get_alert_history()
        self.assertEqual(history_after_ack[0]['status'], AlertStatus.ACKNOWLEDGED.value)
        self.assertIsNotNone(history_after_ack[0].get('acknowledged_at'))

        self.processor.resolve_alert(alert_id)
        history_after_resolve = self.processor.get_alert_history()
        self.assertEqual(history_after_resolve[0]['status'], AlertStatus.RESOLVED.value)
        self.assertIsNotNone(history_after_resolve[0].get('resolved_at'))

    def test_cleanup_old_alerts(self):
        """测试清理旧告警"""
        old_alert_id = "legacy_alert_1"
        new_alert_id = "recent_alert_1"
        self.processor.alert_history.extend([
            {
                'id': old_alert_id,
                'type': 'legacy_alert',
                'severity': AlertSeverity.WARNING.value,
                'title': 'legacy',
                'description': 'old alert',
                'current_value': 1,
                'threshold': 0,
                'metric_name': 'legacy',
                'created_at': datetime.now() - timedelta(days=31),
            },
            {
                'id': new_alert_id,
                'type': 'recent_alert',
                'severity': AlertSeverity.WARNING.value,
                'title': 'recent',
                'description': 'new alert',
                'current_value': 1,
                'threshold': 0,
                'metric_name': 'recent',
                'created_at': datetime.now(),
            },
        ])

        with patch('src.infrastructure.monitoring.services.alert_processor.logger') as mock_logger:
            self.processor.cleanup_old_alerts(days=30)
            mock_logger.info.assert_called_once_with('清理了 1 个过期告警')

        self.assertEqual(len(self.processor.alert_history), 1)
        self.assertEqual(self.processor.alert_history[0]['id'], new_alert_id)
        self.assertEqual(self.processor.alert_history[0]['type'], 'recent_alert')

    def test_alert_history_max_size_enforced(self):
        """测试告警历史大小限制"""
        self.processor.max_history_size = 5

        for i in range(10):
            self.processor._store_alert({
                'id': f'alert_{i}',
                'type': 'cpu_usage_high',
                'severity': AlertSeverity.WARNING.value,
                'status': AlertStatus.ACTIVE.value,
                'title': f'Alert {i}',
                'description': 'Test alert',
                'current_value': i,
                'threshold': 70,
                'metric_name': 'cpu_usage_percent',
                'created_at': datetime.now(),
                'updated_at': datetime.now(),
                'acknowledged_at': None,
                'resolved_at': None,
                'tags': {},
                'context': {},
            })

        self.assertEqual(len(self.processor.alert_history), 5)
        self.assertEqual(self.processor.alert_history[0]['id'], 'alert_5')

    def test_get_health_status(self):
        """测试获取健康状态"""
        health = self.processor.get_health_status()

        self.assertIn('status', health)
        self.assertIn('issues', health)
        self.assertIn('active_alerts', health)

        # 没有活跃告警时应该是健康状态
        self.assertEqual(health['active_alerts'], 0)
        self.assertEqual(health['status'], 'healthy')

    def test_get_health_status_with_alerts(self):
        """测试有告警时的健康状态"""
        # 创建严重告警
        metrics = {'system_metrics': {'cpu': {'usage_percent': 95}}}
        self.processor.process_alerts(metrics)

        health = self.processor.get_health_status()

        self.assertEqual(health['active_alerts'], 1)
        self.assertEqual(health['status'], 'warning')
        self.assertGreater(len(health['issues']), 0)

    def test_validate_rule_condition(self):
        """测试规则条件验证"""
        # 有效的条件
        valid_rule = Mock()
        valid_rule.condition = "cpu_usage > 80"
        valid_rule.action = "reduce_load"

        errors = self.processor.validate_rule_condition(valid_rule)
        self.assertEqual(len(errors), 0)

        # 无效的条件语法
        invalid_rule = Mock()
        invalid_rule.condition = "invalid condition syntax"
        invalid_rule.action = "reduce_load"

        errors = self.processor.validate_rule_condition(invalid_rule)
        self.assertGreater(len(errors), 0)

        # 不支持的动作
        unsupported_rule = Mock()
        unsupported_rule.condition = "cpu_usage > 80"
        unsupported_rule.action = "unsupported_action"

        errors = self.processor.validate_rule_condition(unsupported_rule)
        self.assertGreater(len(errors), 0)


if __name__ == '__main__':
    unittest.main()
    
