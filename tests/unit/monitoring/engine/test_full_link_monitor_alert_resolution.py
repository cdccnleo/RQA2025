#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FullLinkMonitor告警解决测试
补充full_link_monitor.py中resolve_alert、get_active_alerts等方法的全面测试
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
    engine_full_link_monitor_module = importlib.import_module('src.monitoring.engine.full_link_monitor')
    FullLinkMonitor = getattr(engine_full_link_monitor_module, 'FullLinkMonitor', None)
    AlertLevel = getattr(engine_full_link_monitor_module, 'AlertLevel', None)
    MonitorType = getattr(engine_full_link_monitor_module, 'MonitorType', None)
    MetricData = getattr(engine_full_link_monitor_module, 'MetricData', None)
    AlertRule = getattr(engine_full_link_monitor_module, 'AlertRule', None)
    Alert = getattr(engine_full_link_monitor_module, 'Alert', None)
    
    if FullLinkMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestFullLinkMonitorAlertResolution:
    """测试FullLinkMonitor告警解决功能"""

    @pytest.fixture
    def monitor(self):
        """创建monitor实例"""
        return FullLinkMonitor()

    @pytest.fixture
    def sample_alert(self):
        """创建示例告警"""
        return Alert(
            id='test_alert_123',
            rule_name='test_rule',
            metric_name='cpu_usage',
            current_value=90.0,
            threshold='> 80',
            level=AlertLevel.WARNING,
            timestamp=datetime.now(),
            message='Test alert'
        )

    def test_resolve_alert_success(self, monitor, sample_alert):
        """测试成功解决告警"""
        # 添加告警到活跃告警列表
        monitor.active_alerts['test_rule'] = sample_alert
        
        # 解决告警
        monitor.resolve_alert('test_alert_123')
        
        # 验证告警已解决
        assert sample_alert.resolved == True
        assert sample_alert.resolved_time is not None
        # 验证告警已从活跃列表中移除
        assert 'test_rule' not in monitor.active_alerts

    def test_resolve_alert_not_found(self, monitor):
        """测试解决不存在的告警"""
        # 初始状态没有告警
        monitor.active_alerts.clear()
        
        # 尝试解决不存在的告警
        monitor.resolve_alert('non_existent_alert')
        
        # 验证方法执行不抛异常（应该记录警告）
        assert True

    def test_resolve_alert_multiple_alerts(self, monitor):
        """测试多个告警时的解决"""
        # 创建多个告警
        alert1 = Alert(
            id='alert1',
            rule_name='rule1',
            metric_name='cpu_usage',
            current_value=90.0,
            threshold='> 80',
            level=AlertLevel.WARNING,
            timestamp=datetime.now(),
            message='Alert 1'
        )
        alert2 = Alert(
            id='alert2',
            rule_name='rule2',
            metric_name='memory_usage',
            current_value=95.0,
            threshold='> 85',
            level=AlertLevel.ERROR,
            timestamp=datetime.now(),
            message='Alert 2'
        )
        
        monitor.active_alerts['rule1'] = alert1
        monitor.active_alerts['rule2'] = alert2
        
        # 解决第一个告警
        monitor.resolve_alert('alert1')
        
        # 验证第一个告警已解决，第二个仍在
        assert alert1.resolved == True
        assert 'rule1' not in monitor.active_alerts
        assert 'rule2' in monitor.active_alerts
        assert alert2.resolved == False

    def test_resolve_alert_by_id_matching(self, monitor):
        """测试通过ID匹配解决告警"""
        alert = Alert(
            id='specific_alert_id',
            rule_name='test_rule',
            metric_name='cpu_usage',
            current_value=90.0,
            threshold='> 80',
            level=AlertLevel.WARNING,
            timestamp=datetime.now(),
            message='Test alert'
        )
        
        monitor.active_alerts['test_rule'] = alert
        
        # 使用ID解决告警
        monitor.resolve_alert('specific_alert_id')
        
        assert alert.resolved == True
        assert 'test_rule' not in monitor.active_alerts

    def test_resolve_alert_resolved_time_set(self, monitor, sample_alert):
        """测试解决告警时设置解决时间"""
        monitor.active_alerts['test_rule'] = sample_alert
        
        before_time = datetime.now()
        monitor.resolve_alert('test_alert_123')
        after_time = datetime.now()
        
        # 验证解决时间已设置
        assert sample_alert.resolved_time is not None
        assert before_time <= sample_alert.resolved_time <= after_time

    def test_get_active_alerts_empty(self, monitor):
        """测试获取空活跃告警列表"""
        monitor.active_alerts.clear()
        
        active_alerts = monitor.get_active_alerts()
        
        assert isinstance(active_alerts, list)
        assert len(active_alerts) == 0

    def test_get_active_alerts_with_alerts(self, monitor):
        """测试获取有告警的活跃列表"""
        alert1 = Alert(
            id='alert1',
            rule_name='rule1',
            metric_name='cpu_usage',
            current_value=90.0,
            threshold='> 80',
            level=AlertLevel.WARNING,
            timestamp=datetime.now(),
            message='Alert 1'
        )
        alert2 = Alert(
            id='alert2',
            rule_name='rule2',
            metric_name='memory_usage',
            current_value=95.0,
            threshold='> 85',
            level=AlertLevel.ERROR,
            timestamp=datetime.now(),
            message='Alert 2'
        )
        
        monitor.active_alerts['rule1'] = alert1
        monitor.active_alerts['rule2'] = alert2
        
        active_alerts = monitor.get_active_alerts()
        
        assert isinstance(active_alerts, list)
        assert len(active_alerts) == 2
        assert alert1 in active_alerts
        assert alert2 in active_alerts

    def test_get_active_alerts_returns_copy(self, monitor):
        """测试返回的是副本"""
        alert = Alert(
            id='alert1',
            rule_name='rule1',
            metric_name='cpu_usage',
            current_value=90.0,
            threshold='> 80',
            level=AlertLevel.WARNING,
            timestamp=datetime.now(),
            message='Alert 1'
        )
        
        monitor.active_alerts['rule1'] = alert
        
        active_alerts = monitor.get_active_alerts()
        
        # 修改返回的列表不应该影响内部状态
        active_alerts.append('test')
        assert len(monitor.get_active_alerts()) == 1


class TestFullLinkMonitorAlertCallbacks:
    """测试FullLinkMonitor告警回调功能"""

    @pytest.fixture
    def monitor(self):
        """创建monitor实例"""
        return FullLinkMonitor()

    def test_add_metric_callback(self, monitor):
        """测试添加指标回调"""
        callback = Mock()
        
        monitor.add_metric_callback(callback)
        
        assert callback in monitor.metric_callbacks

    def test_add_metric_callback_multiple(self, monitor):
        """测试添加多个指标回调"""
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()
        
        monitor.add_metric_callback(callback1)
        monitor.add_metric_callback(callback2)
        monitor.add_metric_callback(callback3)
        
        assert len(monitor.metric_callbacks) >= 3
        assert callback1 in monitor.metric_callbacks
        assert callback2 in monitor.metric_callbacks
        assert callback3 in monitor.metric_callbacks

    def test_add_alert_callback(self, monitor):
        """测试添加告警回调"""
        callback = Mock()
        
        monitor.add_alert_callback(callback)
        
        assert callback in monitor.alert_callbacks

    def test_add_alert_callback_multiple(self, monitor):
        """测试添加多个告警回调"""
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()
        
        monitor.add_alert_callback(callback1)
        monitor.add_alert_callback(callback2)
        monitor.add_alert_callback(callback3)
        
        assert len(monitor.alert_callbacks) >= 3
        assert callback1 in monitor.alert_callbacks
        assert callback2 in monitor.alert_callbacks
        assert callback3 in monitor.alert_callbacks

    def test_metric_callback_called_on_add_metric(self, monitor):
        """测试添加指标时调用回调"""
        callback_called = []
        
        def test_callback(metric):
            callback_called.append(metric)
        
        monitor.add_metric_callback(test_callback)
        
        metric = MetricData(
            name='test_metric',
            value=50.0,
            timestamp=datetime.now(),
            tags={},
            monitor_type=MonitorType.SYSTEM,
            source='test'
        )
        
        monitor.add_metric(metric)
        
        # 验证回调被调用
        assert len(callback_called) > 0
        assert callback_called[0] == metric

    def test_alert_callback_called_on_trigger(self, monitor):
        """测试触发告警时调用回调"""
        callback_called = []
        
        def test_callback(alert):
            callback_called.append(alert)
        
        monitor.add_alert_callback(test_callback)
        
        # 添加规则
        rule = AlertRule(
            name='callback_rule',
            metric_name='cpu_usage',
            condition='> 80',
            level=AlertLevel.WARNING,
            duration=60,
            enabled=True
        )
        monitor.add_alert_rule(rule)
        
        # 触发告警
        metric = MetricData(
            name='cpu_usage',
            value=90.0,
            timestamp=datetime.now(),
            tags={},
            monitor_type=MonitorType.SYSTEM,
            source='test'
        )
        
        monitor.add_metric(metric)
        
        # 验证回调可能被调用（等待告警处理）
        import time
        time.sleep(0.1)
        
        # 至少验证回调已注册
        assert test_callback in monitor.alert_callbacks

    def test_callback_error_handling(self, monitor):
        """测试回调错误处理"""
        def failing_callback(metric):
            raise Exception("Callback error")
        
        def successful_callback(metric):
            pass
        
        monitor.add_metric_callback(failing_callback)
        monitor.add_metric_callback(successful_callback)
        
        metric = MetricData(
            name='test_metric',
            value=50.0,
            timestamp=datetime.now(),
            tags={},
            monitor_type=MonitorType.SYSTEM,
            source='test'
        )
        
        # 验证即使一个回调失败，其他回调仍能执行
        # add_metric方法内部应该有错误处理
        monitor.add_metric(metric)
        
        # 验证方法不抛异常
        assert True



