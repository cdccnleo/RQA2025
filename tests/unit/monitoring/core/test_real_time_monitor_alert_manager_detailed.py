#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor AlertManager详细测试
补充AlertManager类的详细功能和边界情况测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入模块
try:
    core_real_time_monitor_module = importlib.import_module('src.monitoring.core.real_time_monitor')
    AlertManager = getattr(core_real_time_monitor_module, 'AlertManager', None)
    AlertRule = getattr(core_real_time_monitor_module, 'AlertRule', None)
    Alert = getattr(core_real_time_monitor_module, 'Alert', None)
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    if AlertManager is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestAlertManagerDetailed:
    """测试AlertManager类的详细功能和边界情况"""

    @pytest.fixture
    def alert_manager(self):
        """创建AlertManager实例"""
        return AlertManager()

    @pytest.fixture
    def base_rule(self):
        """创建基础告警规则"""
        return AlertRule(
            name='test_rule',
            metric_name='test_metric',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='Test rule',
            enabled=True
        )

    def test_check_alerts_alert_structure_complete(self, alert_manager, base_rule):
        """测试告警结构完整性"""
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        alert_manager.add_rule(base_rule)
        
        alerts = alert_manager.check_alerts(metrics)
        
        assert len(alerts) == 1
        alert = alerts[0]
        assert alert.rule_name == 'test_rule'
        assert alert.metric_name == 'test_metric'
        assert alert.current_value == 90.0
        assert alert.threshold == 80.0
        assert alert.severity == 'warning'
        assert alert.resolved == False
        assert alert.resolved_at is None
        assert isinstance(alert.timestamp, datetime)
        assert isinstance(alert.message, str)

    def test_check_alerts_alert_message_format(self, alert_manager, base_rule):
        """测试告警消息格式"""
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        alert_manager.add_rule(base_rule)
        
        alerts = alert_manager.check_alerts(metrics)
        
        alert = alerts[0]
        assert base_rule.description in alert.message
        assert str(90.0) in alert.message
        assert base_rule.condition in alert.message
        assert str(base_rule.threshold) in alert.message

    def test_check_alerts_resolve_sets_resolved_flag(self, alert_manager, base_rule):
        """测试告警解决设置resolved标志"""
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        alert_manager.add_rule(base_rule)
        
        # 触发告警
        alerts1 = alert_manager.check_alerts(metrics)
        alert = alerts1[0]
        assert alert.resolved == False
        
        # 解决告警
        metrics['test_metric'].value = 50.0
        alert_manager.check_alerts(metrics)
        
        # 验证告警被标记为已解决
        assert alert.resolved == True

    def test_check_alerts_resolve_sets_resolved_at(self, alert_manager, base_rule):
        """测试告警解决设置resolved_at时间戳"""
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        alert_manager.add_rule(base_rule)
        
        # 触发告警
        alerts1 = alert_manager.check_alerts(metrics)
        alert = alerts1[0]
        assert alert.resolved_at is None
        
        # 解决告警
        with patch('src.monitoring.core.real_time_monitor.datetime') as mock_datetime:
            resolve_time = datetime(2024, 1, 1, 12, 0, 0)
            mock_datetime.now.return_value = resolve_time
            
            metrics['test_metric'].value = 50.0
            alert_manager.check_alerts(metrics)
            
            # 验证resolved_at被设置
            assert alert.resolved_at == resolve_time

    def test_check_alerts_resolve_removes_from_active(self, alert_manager, base_rule):
        """测试告警解决后从活跃告警中移除"""
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        alert_manager.add_rule(base_rule)
        
        # 触发告警
        alert_manager.check_alerts(metrics)
        assert len(alert_manager.active_alerts) == 1
        
        # 解决告警
        metrics['test_metric'].value = 50.0
        alert_manager.check_alerts(metrics)
        
        # 验证从活跃告警中移除
        assert len(alert_manager.active_alerts) == 0

    def test_check_alerts_multiple_rules_same_metric(self, alert_manager):
        """测试同一指标的多个规则"""
        rule1 = AlertRule(
            name='rule1', metric_name='cpu', condition='>', threshold=80.0,
            duration=60, severity='warning', description='Rule 1'
        )
        rule2 = AlertRule(
            name='rule2', metric_name='cpu', condition='>', threshold=90.0,
            duration=60, severity='critical', description='Rule 2'
        )
        
        alert_manager.add_rule(rule1)
        alert_manager.add_rule(rule2)
        
        metrics = {
            'cpu': MetricData(
                name='cpu',
                value=85.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = alert_manager.check_alerts(metrics)
        
        # 只有rule1应该触发（85 > 80 但不 > 90）
        assert len(alerts) == 1
        assert alerts[0].rule_name == 'rule1'

    def test_check_alerts_multiple_rules_different_metrics(self, alert_manager):
        """测试不同指标的多个规则"""
        rule1 = AlertRule(
            name='cpu_rule', metric_name='cpu', condition='>', threshold=80.0,
            duration=60, severity='warning', description='CPU rule'
        )
        rule2 = AlertRule(
            name='memory_rule', metric_name='memory', condition='>', threshold=70.0,
            duration=60, severity='warning', description='Memory rule'
        )
        
        alert_manager.add_rule(rule1)
        alert_manager.add_rule(rule2)
        
        metrics = {
            'cpu': MetricData('cpu', 85.0, datetime.now(), {}),
            'memory': MetricData('memory', 75.0, datetime.now(), {})
        }
        
        alerts = alert_manager.check_alerts(metrics)
        
        # 两个规则都应该触发
        assert len(alerts) == 2
        rule_names = {alert.rule_name for alert in alerts}
        assert 'cpu_rule' in rule_names
        assert 'memory_rule' in rule_names

    def test_check_alerts_alert_history_accumulation(self, alert_manager, base_rule):
        """测试告警历史累积"""
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        alert_manager.add_rule(base_rule)
        
        # 第一次触发
        alert_manager.check_alerts(metrics)
        assert len(alert_manager.alert_history) == 1
        
        # 解决告警
        metrics['test_metric'].value = 50.0
        alert_manager.check_alerts(metrics)
        assert len(alert_manager.alert_history) == 1  # 历史不减少
        
        # 再次触发
        metrics['test_metric'].value = 90.0
        alert_manager.check_alerts(metrics)
        assert len(alert_manager.alert_history) == 2  # 历史增加

    def test_get_alert_history_time_filtering(self, alert_manager, base_rule):
        """测试告警历史时间过滤"""
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        alert_manager.add_rule(base_rule)
        
        # 创建两个告警
        alert_manager.check_alerts(metrics)
        alert_manager.check_alerts(metrics)
        
        # 手动修改告警的时间戳以测试过滤
        current_time = datetime.now()
        if len(alert_manager.alert_history) >= 2:
            # 第一个告警：2小时前
            alert_manager.alert_history[0].timestamp = current_time - timedelta(hours=2)
            # 第二个告警：30分钟前
            alert_manager.alert_history[1].timestamp = current_time - timedelta(minutes=30)
            
            # 获取1小时内的历史（应该只有第二个）
            history = alert_manager.get_alert_history(hours=1)
            assert len(history) == 1
            
            # 获取3小时内的历史（应该有两个）
            history = alert_manager.get_alert_history(hours=3)
            assert len(history) == 2

    def test_get_alert_history_empty_when_no_alerts(self, alert_manager):
        """测试无告警时历史为空"""
        history = alert_manager.get_alert_history()
        assert len(history) == 0

    def test_get_alert_history_default_24_hours(self, alert_manager, base_rule):
        """测试默认获取24小时内的历史"""
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        alert_manager.add_rule(base_rule)
        alert_manager.check_alerts(metrics)
        
        # 不传参数应该默认24小时
        history_default = alert_manager.get_alert_history()
        history_24h = alert_manager.get_alert_history(hours=24)
        
        assert len(history_default) == len(history_24h)

    def test_check_alerts_condition_boundary_values(self, alert_manager):
        """测试条件边界值"""
        # 测试 > 条件：值刚好等于阈值（不触发）
        rule = AlertRule(
            name='boundary_test', metric_name='test', condition='>',
            threshold=80.0, duration=60, severity='warning', description='Test'
        )
        alert_manager.add_rule(rule)
        
        metrics = {
            'test': MetricData('test', 80.0, datetime.now(), {})
        }
        
        alerts = alert_manager.check_alerts(metrics)
        assert len(alerts) == 0  # 80.0 不大于 80.0

    def test_check_alerts_condition_boundary_values_greater_equal(self, alert_manager):
        """测试 >= 条件边界值"""
        rule = AlertRule(
            name='boundary_test', metric_name='test', condition='>=',
            threshold=80.0, duration=60, severity='warning', description='Test'
        )
        alert_manager.add_rule(rule)
        
        metrics = {
            'test': MetricData('test', 80.0, datetime.now(), {})
        }
        
        alerts = alert_manager.check_alerts(metrics)
        assert len(alerts) == 1  # 80.0 >= 80.0 应该触发

    def test_check_alerts_condition_boundary_values_less(self, alert_manager):
        """测试 < 条件边界值"""
        rule = AlertRule(
            name='boundary_test', metric_name='test', condition='<',
            threshold=80.0, duration=60, severity='warning', description='Test'
        )
        alert_manager.add_rule(rule)
        
        metrics = {
            'test': MetricData('test', 80.0, datetime.now(), {})
        }
        
        alerts = alert_manager.check_alerts(metrics)
        assert len(alerts) == 0  # 80.0 不小于 80.0

    def test_check_alerts_condition_boundary_values_less_equal(self, alert_manager):
        """测试 <= 条件边界值"""
        rule = AlertRule(
            name='boundary_test', metric_name='test', condition='<=',
            threshold=80.0, duration=60, severity='warning', description='Test'
        )
        alert_manager.add_rule(rule)
        
        metrics = {
            'test': MetricData('test', 80.0, datetime.now(), {})
        }
        
        alerts = alert_manager.check_alerts(metrics)
        assert len(alerts) == 1  # 80.0 <= 80.0 应该触发

    def test_check_alerts_condition_boundary_values_equal(self, alert_manager):
        """测试 == 条件边界值"""
        rule = AlertRule(
            name='boundary_test', metric_name='test', condition='==',
            threshold=80.0, duration=60, severity='warning', description='Test'
        )
        alert_manager.add_rule(rule)
        
        metrics = {
            'test': MetricData('test', 80.0, datetime.now(), {})
        }
        
        alerts = alert_manager.check_alerts(metrics)
        assert len(alerts) == 1  # 80.0 == 80.0 应该触发

    def test_check_alerts_multiple_callbacks(self, alert_manager, base_rule):
        """测试多个回调函数"""
        callback1 = Mock()
        callback2 = Mock()
        
        alert_manager.add_alert_callback(callback1)
        alert_manager.add_alert_callback(callback2)
        alert_manager.add_rule(base_rule)
        
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alert_manager.check_alerts(metrics)
        
        # 两个回调都应该被调用
        callback1.assert_called_once()
        callback2.assert_called_once()

    def test_check_alerts_callback_receives_alert_object(self, alert_manager, base_rule):
        """测试回调函数接收告警对象"""
        received_alert = None
        
        def test_callback(alert):
            nonlocal received_alert
            received_alert = alert
        
        alert_manager.add_alert_callback(test_callback)
        alert_manager.add_rule(base_rule)
        
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = alert_manager.check_alerts(metrics)
        
        # 验证回调接收的告警对象
        assert received_alert is not None
        assert received_alert == alerts[0]

    def test_check_alerts_one_callback_fails_others_still_called(self, alert_manager, base_rule):
        """测试一个回调失败不影响其他回调"""
        callback1 = Mock(side_effect=Exception("Callback 1 error"))
        callback2 = Mock()
        
        alert_manager.add_alert_callback(callback1)
        alert_manager.add_alert_callback(callback2)
        alert_manager.add_rule(base_rule)
        
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        with patch('src.monitoring.core.real_time_monitor.logger'):
            # 不应该抛出异常
            alerts = alert_manager.check_alerts(metrics)
            assert len(alerts) == 1
            
            # callback2应该仍然被调用
            callback2.assert_called_once()

    def test_get_active_alerts_returns_list_copy(self, alert_manager, base_rule):
        """测试get_active_alerts返回列表副本"""
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        alert_manager.add_rule(base_rule)
        alert_manager.check_alerts(metrics)
        
        active_alerts1 = alert_manager.get_active_alerts()
        active_alerts2 = alert_manager.get_active_alerts()
        
        # 应该是不同的列表对象（副本）
        assert active_alerts1 is not active_alerts2
        # 但内容应该相同
        assert len(active_alerts1) == len(active_alerts2) == 1

    def test_alert_key_format(self, alert_manager):
        """测试告警键格式（规则名_指标名）"""
        rule = AlertRule(
            name='test_rule', metric_name='test_metric', condition='>',
            threshold=80.0, duration=60, severity='warning', description='Test'
        )
        alert_manager.add_rule(rule)
        
        metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alert_manager.check_alerts(metrics)
        
        # 验证告警键格式
        expected_key = 'test_rule_test_metric'
        assert expected_key in alert_manager.active_alerts

    def test_check_alerts_float_precision(self, alert_manager):
        """测试浮点数精度问题"""
        rule = AlertRule(
            name='precision_test', metric_name='test', condition='>',
            threshold=80.0, duration=60, severity='warning', description='Test'
        )
        alert_manager.add_rule(rule)
        
        # 使用非常接近阈值的值
        metrics = {
            'test': MetricData(
                name='test',
                value=80.0000001,  # 略微大于阈值
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = alert_manager.check_alerts(metrics)
        assert len(alerts) == 1

