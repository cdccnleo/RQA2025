#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor AlertManager测试
补充AlertManager类的方法测试
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


class TestAlertManager:
    """测试AlertManager类"""

    @pytest.fixture
    def alert_manager(self):
        """创建AlertManager实例"""
        return AlertManager()

    @pytest.fixture
    def sample_rule(self):
        """创建示例告警规则"""
        return AlertRule(
            name='cpu_high',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='CPU使用率过高',
            enabled=True
        )

    @pytest.fixture
    def sample_metrics(self):
        """创建示例指标"""
        return {
            'cpu_percent': MetricData(
                name='cpu_percent',
                value=85.0,
                timestamp=datetime.now(),
                tags={}
            )
        }

    def test_init(self, alert_manager):
        """测试初始化"""
        assert alert_manager.rules == {}
        assert alert_manager.active_alerts == {}
        assert alert_manager.alert_history == []
        assert alert_manager.alert_callbacks == []

    def test_add_rule(self, alert_manager, sample_rule):
        """测试添加告警规则"""
        alert_manager.add_rule(sample_rule)
        
        assert 'cpu_high' in alert_manager.rules
        assert alert_manager.rules['cpu_high'] == sample_rule

    def test_add_rule_multiple(self, alert_manager):
        """测试添加多个告警规则"""
        rule1 = AlertRule(
            name='rule1', metric_name='cpu', condition='>', threshold=80.0,
            duration=60, severity='warning', description='Rule 1'
        )
        rule2 = AlertRule(
            name='rule2', metric_name='memory', condition='>', threshold=70.0,
            duration=60, severity='warning', description='Rule 2'
        )
        
        alert_manager.add_rule(rule1)
        alert_manager.add_rule(rule2)
        
        assert len(alert_manager.rules) == 2

    def test_remove_rule(self, alert_manager, sample_rule):
        """测试移除告警规则"""
        alert_manager.add_rule(sample_rule)
        alert_manager.remove_rule('cpu_high')
        
        assert 'cpu_high' not in alert_manager.rules

    def test_remove_rule_nonexistent(self, alert_manager):
        """测试移除不存在的规则"""
        # 不应该抛出异常
        alert_manager.remove_rule('nonexistent')

    def test_add_alert_callback(self, alert_manager):
        """测试添加告警回调"""
        callback = Mock()
        
        alert_manager.add_alert_callback(callback)
        
        assert callback in alert_manager.alert_callbacks

    def test_check_alerts_condition_greater_than(self, alert_manager, sample_rule, sample_metrics):
        """测试检查告警条件 >"""
        alert_manager.add_rule(sample_rule)
        
        alerts = alert_manager.check_alerts(sample_metrics)
        
        assert len(alerts) == 1
        assert alerts[0].rule_name == 'cpu_high'

    def test_check_alerts_condition_less_than(self, alert_manager, sample_metrics):
        """测试检查告警条件 <"""
        rule = AlertRule(
            name='cpu_low', metric_name='cpu_percent', condition='<',
            threshold=10.0, duration=60, severity='info', description='CPU使用率过低'
        )
        alert_manager.add_rule(rule)
        
        # 修改指标值为5.0，小于阈值10.0
        sample_metrics['cpu_percent'].value = 5.0
        
        alerts = alert_manager.check_alerts(sample_metrics)
        
        assert len(alerts) == 1

    def test_check_alerts_condition_greater_equal(self, alert_manager, sample_rule, sample_metrics):
        """测试检查告警条件 >="""
        sample_rule.condition = '>='
        sample_rule.threshold = 85.0
        alert_manager.add_rule(sample_rule)
        
        # 值等于阈值，应该触发
        alerts = alert_manager.check_alerts(sample_metrics)
        
        assert len(alerts) == 1

    def test_check_alerts_condition_less_equal(self, alert_manager, sample_metrics):
        """测试检查告警条件 <="""
        rule = AlertRule(
            name='cpu_low', metric_name='cpu_percent', condition='<=',
            threshold=10.0, duration=60, severity='info', description='CPU使用率过低'
        )
        alert_manager.add_rule(rule)
        
        sample_metrics['cpu_percent'].value = 10.0  # 等于阈值
        
        alerts = alert_manager.check_alerts(sample_metrics)
        
        assert len(alerts) == 1

    def test_check_alerts_condition_equal(self, alert_manager, sample_metrics):
        """测试检查告警条件 =="""
        rule = AlertRule(
            name='cpu_exact', metric_name='cpu_percent', condition='==',
            threshold=85.0, duration=60, severity='info', description='CPU等于阈值'
        )
        alert_manager.add_rule(rule)
        
        alerts = alert_manager.check_alerts(sample_metrics)
        
        assert len(alerts) == 1

    def test_check_alerts_not_triggered(self, alert_manager, sample_rule, sample_metrics):
        """测试告警未触发"""
        sample_rule.threshold = 90.0  # 阈值高于当前值
        alert_manager.add_rule(sample_rule)
        
        alerts = alert_manager.check_alerts(sample_metrics)
        
        assert len(alerts) == 0

    def test_check_alerts_rule_disabled(self, alert_manager, sample_rule, sample_metrics):
        """测试规则禁用时不触发告警"""
        sample_rule.enabled = False
        alert_manager.add_rule(sample_rule)
        
        alerts = alert_manager.check_alerts(sample_metrics)
        
        assert len(alerts) == 0

    def test_check_alerts_metric_missing(self, alert_manager, sample_rule):
        """测试指标缺失时不触发告警"""
        alert_manager.add_rule(sample_rule)
        metrics = {}  # 空指标
        
        alerts = alert_manager.check_alerts(metrics)
        
        assert len(alerts) == 0

    def test_check_alerts_duplicate_alert(self, alert_manager, sample_rule, sample_metrics):
        """测试重复告警不创建新告警"""
        alert_manager.add_rule(sample_rule)
        
        # 第一次检查
        alerts1 = alert_manager.check_alerts(sample_metrics)
        assert len(alerts1) == 1
        
        # 第二次检查，应该不创建新告警
        alerts2 = alert_manager.check_alerts(sample_metrics)
        assert len(alerts2) == 0

    def test_check_alerts_callback_triggered(self, alert_manager, sample_rule, sample_metrics):
        """测试告警触发回调"""
        callback = Mock()
        alert_manager.add_alert_callback(callback)
        alert_manager.add_rule(sample_rule)
        
        alert_manager.check_alerts(sample_metrics)
        
        callback.assert_called_once()

    def test_check_alerts_callback_exception(self, alert_manager, sample_rule, sample_metrics):
        """测试告警回调异常处理"""
        def failing_callback(alert):
            raise Exception("Callback error")
        
        alert_manager.add_alert_callback(failing_callback)
        alert_manager.add_rule(sample_rule)
        
        # 不应该抛出异常
        alerts = alert_manager.check_alerts(sample_metrics)
        assert len(alerts) == 1

    def test_check_alerts_resolve_alert(self, alert_manager, sample_rule, sample_metrics):
        """测试告警解决"""
        alert_manager.add_rule(sample_rule)
        
        # 触发告警
        alerts1 = alert_manager.check_alerts(sample_metrics)
        assert len(alerts1) == 1
        assert len(alert_manager.active_alerts) == 1
        
        # 降低指标值，告警应该解决
        sample_metrics['cpu_percent'].value = 50.0
        alerts2 = alert_manager.check_alerts(sample_metrics)
        assert len(alerts2) == 0
        assert len(alert_manager.active_alerts) == 0

    def test_get_active_alerts(self, alert_manager, sample_rule, sample_metrics):
        """测试获取活跃告警"""
        alert_manager.add_rule(sample_rule)
        alert_manager.check_alerts(sample_metrics)
        
        active_alerts = alert_manager.get_active_alerts()
        
        assert len(active_alerts) == 1
        assert active_alerts[0].rule_name == 'cpu_high'

    def test_get_alert_history(self, alert_manager, sample_rule, sample_metrics):
        """测试获取告警历史"""
        alert_manager.add_rule(sample_rule)
        alert_manager.check_alerts(sample_metrics)
        
        assert len(alert_manager.alert_history) == 1
        assert alert_manager.alert_history[0].rule_name == 'cpu_high'

    def test_get_alert_history_with_hours_filter(self, alert_manager, sample_rule, sample_metrics):
        """测试获取告警历史按时间过滤"""
        alert_manager.add_rule(sample_rule)
        alert_manager.check_alerts(sample_metrics)
        
        # 获取24小时内的历史（默认）
        history_24h = alert_manager.get_alert_history(24)
        assert len(history_24h) == 1
        
        # 获取1小时内的历史
        history_1h = alert_manager.get_alert_history(1)
        assert len(history_1h) == 1
        
        # 获取0小时内的历史（应该为空）
        history_0h = alert_manager.get_alert_history(0)
        assert len(history_0h) == 0

