#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor默认告警规则测试
补充_setup_default_alert_rules方法的详细测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from datetime import datetime

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
    RealTimeMonitor = getattr(core_real_time_monitor_module, 'RealTimeMonitor', None)
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    AlertRule = getattr(core_real_time_monitor_module, 'AlertRule', None)
    if RealTimeMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestRealTimeMonitorDefaultAlertRules:
    """测试RealTimeMonitor类默认告警规则"""

    @pytest.fixture
    def monitor(self):
        """创建RealTimeMonitor实例"""
        return RealTimeMonitor()

    def test_default_rules_are_created(self, monitor):
        """测试默认告警规则被创建"""
        assert len(monitor.alert_manager.rules) > 0

    def test_default_rule_high_cpu_usage_exists(self, monitor):
        """测试high_cpu_usage规则存在"""
        assert 'high_cpu_usage' in monitor.alert_manager.rules

    def test_default_rule_high_cpu_usage_config(self, monitor):
        """测试high_cpu_usage规则配置"""
        rule = monitor.alert_manager.rules['high_cpu_usage']
        assert rule.metric_name == 'cpu_percent'
        assert rule.condition == '>'
        assert rule.threshold == 80.0
        assert rule.duration == 60
        assert rule.severity == 'warning'
        assert rule.description == 'CPU使用率过高'
        assert rule.enabled == True

    def test_default_rule_high_memory_usage_exists(self, monitor):
        """测试high_memory_usage规则存在"""
        assert 'high_memory_usage' in monitor.alert_manager.rules

    def test_default_rule_high_memory_usage_config(self, monitor):
        """测试high_memory_usage规则配置"""
        rule = monitor.alert_manager.rules['high_memory_usage']
        assert rule.metric_name == 'memory_percent'
        assert rule.condition == '>'
        assert rule.threshold == 85.0
        assert rule.duration == 60
        assert rule.severity == 'warning'
        assert rule.description == '内存使用率过高'
        assert rule.enabled == True

    def test_default_rule_low_memory_available_exists(self, monitor):
        """测试low_memory_available规则存在"""
        assert 'low_memory_available' in monitor.alert_manager.rules

    def test_default_rule_low_memory_available_config(self, monitor):
        """测试low_memory_available规则配置"""
        rule = monitor.alert_manager.rules['low_memory_available']
        assert rule.metric_name == 'memory_available_mb'
        assert rule.condition == '<'
        assert rule.threshold == 512.0
        assert rule.duration == 60
        assert rule.severity == 'error'
        assert rule.description == '可用内存不足'
        assert rule.enabled == True

    def test_default_rule_high_error_rate_exists(self, monitor):
        """测试high_error_rate规则存在"""
        assert 'high_error_rate' in monitor.alert_manager.rules

    def test_default_rule_high_error_rate_config(self, monitor):
        """测试high_error_rate规则配置"""
        rule = monitor.alert_manager.rules['high_error_rate']
        assert rule.metric_name == 'error_rate'
        assert rule.condition == '>'
        assert rule.threshold == 0.05
        assert rule.duration == 300
        assert rule.severity == 'error'
        assert rule.description == '错误率过高'
        assert rule.enabled == True

    def test_default_rules_count(self, monitor):
        """测试默认规则数量"""
        assert len(monitor.alert_manager.rules) == 4

    def test_default_rule_high_cpu_usage_can_trigger(self, monitor):
        """测试high_cpu_usage规则可以触发"""
        metrics = {
            'cpu_percent': MetricData(
                name='cpu_percent',
                value=85.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = monitor.alert_manager.check_alerts(metrics)
        
        # 应该触发告警
        assert len(alerts) == 1
        assert alerts[0].rule_name == 'high_cpu_usage'

    def test_default_rule_high_cpu_usage_not_trigger_below_threshold(self, monitor):
        """测试high_cpu_usage规则在阈值以下不触发"""
        metrics = {
            'cpu_percent': MetricData(
                name='cpu_percent',
                value=75.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = monitor.alert_manager.check_alerts(metrics)
        
        # 不应该触发告警
        assert len(alerts) == 0

    def test_default_rule_high_memory_usage_can_trigger(self, monitor):
        """测试high_memory_usage规则可以触发"""
        metrics = {
            'memory_percent': MetricData(
                name='memory_percent',
                value=90.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = monitor.alert_manager.check_alerts(metrics)
        
        # 应该触发告警
        assert len(alerts) == 1
        assert alerts[0].rule_name == 'high_memory_usage'

    def test_default_rule_high_memory_usage_not_trigger_below_threshold(self, monitor):
        """测试high_memory_usage规则在阈值以下不触发"""
        metrics = {
            'memory_percent': MetricData(
                name='memory_percent',
                value=80.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = monitor.alert_manager.check_alerts(metrics)
        
        # 不应该触发告警
        assert len(alerts) == 0

    def test_default_rule_low_memory_available_can_trigger(self, monitor):
        """测试low_memory_available规则可以触发"""
        metrics = {
            'memory_available_mb': MetricData(
                name='memory_available_mb',
                value=400.0,  # 小于512MB
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = monitor.alert_manager.check_alerts(metrics)
        
        # 应该触发告警
        assert len(alerts) == 1
        assert alerts[0].rule_name == 'low_memory_available'

    def test_default_rule_low_memory_available_not_trigger_above_threshold(self, monitor):
        """测试low_memory_available规则在阈值以上不触发"""
        metrics = {
            'memory_available_mb': MetricData(
                name='memory_available_mb',
                value=600.0,  # 大于512MB
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = monitor.alert_manager.check_alerts(metrics)
        
        # 不应该触发告警
        assert len(alerts) == 0

    def test_default_rule_high_error_rate_can_trigger(self, monitor):
        """测试high_error_rate规则可以触发"""
        metrics = {
            'error_rate': MetricData(
                name='error_rate',
                value=0.06,  # 大于0.05 (5%)
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = monitor.alert_manager.check_alerts(metrics)
        
        # 应该触发告警
        assert len(alerts) == 1
        assert alerts[0].rule_name == 'high_error_rate'

    def test_default_rule_high_error_rate_not_trigger_below_threshold(self, monitor):
        """测试high_error_rate规则在阈值以下不触发"""
        metrics = {
            'error_rate': MetricData(
                name='error_rate',
                value=0.04,  # 小于0.05 (5%)
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        alerts = monitor.alert_manager.check_alerts(metrics)
        
        # 不应该触发告警
        assert len(alerts) == 0

    def test_default_rules_all_enabled(self, monitor):
        """测试所有默认规则都启用"""
        for rule in monitor.alert_manager.rules.values():
            assert rule.enabled == True

    def test_default_rules_can_trigger_multiple_simultaneously(self, monitor):
        """测试多个默认规则可以同时触发"""
        metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 90.0, datetime.now(), {}),
            'memory_available_mb': MetricData('memory_available_mb', 400.0, datetime.now(), {}),
            'error_rate': MetricData('error_rate', 0.06, datetime.now(), {})
        }
        
        alerts = monitor.alert_manager.check_alerts(metrics)
        
        # 应该触发多个告警
        assert len(alerts) == 4
        rule_names = {alert.rule_name for alert in alerts}
        assert 'high_cpu_usage' in rule_names
        assert 'high_memory_usage' in rule_names
        assert 'low_memory_available' in rule_names
        assert 'high_error_rate' in rule_names

    def test_default_rules_severity_levels(self, monitor):
        """测试默认规则严重级别"""
        rule = monitor.alert_manager.rules['high_cpu_usage']
        assert rule.severity == 'warning'
        
        rule = monitor.alert_manager.rules['high_memory_usage']
        assert rule.severity == 'warning'
        
        rule = monitor.alert_manager.rules['low_memory_available']
        assert rule.severity == 'error'
        
        rule = monitor.alert_manager.rules['high_error_rate']
        assert rule.severity == 'error'

    def test_default_rules_duration_values(self, monitor):
        """测试默认规则持续时间值"""
        rule = monitor.alert_manager.rules['high_cpu_usage']
        assert rule.duration == 60
        
        rule = monitor.alert_manager.rules['high_memory_usage']
        assert rule.duration == 60
        
        rule = monitor.alert_manager.rules['low_memory_available']
        assert rule.duration == 60
        
        rule = monitor.alert_manager.rules['high_error_rate']
        assert rule.duration == 300  # 5分钟

    def test_default_rules_boundary_values(self, monitor):
        """测试默认规则边界值"""
        # CPU规则：刚好80%不应该触发（条件是>）
        metrics = {
            'cpu_percent': MetricData('cpu_percent', 80.0, datetime.now(), {})
        }
        alerts = monitor.alert_manager.check_alerts(metrics)
        assert len(alerts) == 0
        
        # 内存规则：刚好85%不应该触发（条件是>）
        metrics = {
            'memory_percent': MetricData('memory_percent', 85.0, datetime.now(), {})
        }
        alerts = monitor.alert_manager.check_alerts(metrics)
        assert len(alerts) == 0
        
        # 可用内存规则：刚好512MB不应该触发（条件是<）
        metrics = {
            'memory_available_mb': MetricData('memory_available_mb', 512.0, datetime.now(), {})
        }
        alerts = monitor.alert_manager.check_alerts(metrics)
        assert len(alerts) == 0
        
        # 错误率规则：刚好0.05不应该触发（条件是>）
        metrics = {
            'error_rate': MetricData('error_rate', 0.05, datetime.now(), {})
        }
        alerts = monitor.alert_manager.check_alerts(metrics)
        assert len(alerts) == 0

    def test_default_rules_just_above_boundary(self, monitor):
        """测试默认规则刚好超过边界值时触发"""
        # CPU规则：80.0001%应该触发
        metrics = {
            'cpu_percent': MetricData('cpu_percent', 80.0001, datetime.now(), {})
        }
        alerts = monitor.alert_manager.check_alerts(metrics)
        assert len(alerts) == 1
        
        # 内存规则：85.0001%应该触发
        metrics = {
            'memory_percent': MetricData('memory_percent', 85.0001, datetime.now(), {})
        }
        alerts = monitor.alert_manager.check_alerts(metrics)
        assert len(alerts) == 1
        
        # 可用内存规则：511.999MB应该触发
        metrics = {
            'memory_available_mb': MetricData('memory_available_mb', 511.999, datetime.now(), {})
        }
        alerts = monitor.alert_manager.check_alerts(metrics)
        assert len(alerts) == 1
        
        # 错误率规则：0.050001应该触发
        metrics = {
            'error_rate': MetricData('error_rate', 0.050001, datetime.now(), {})
        }
        alerts = monitor.alert_manager.check_alerts(metrics)
        assert len(alerts) == 1

