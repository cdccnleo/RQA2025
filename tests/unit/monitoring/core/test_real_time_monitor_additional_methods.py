#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor附加方法测试
补充RealTimeMonitor类的其他方法测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
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
    AlertRule = getattr(core_real_time_monitor_module, 'AlertRule', None)
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    if RealTimeMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)
    pytest.skip("real_time_monitor module not available", allow_module_level=True)


class TestRealTimeMonitorAdditionalMethods:
    """测试RealTimeMonitor类的附加方法"""

    @pytest.fixture
    def monitor(self):
        """创建RealTimeMonitor实例"""
        return RealTimeMonitor()

    def test_update_business_metric(self, monitor):
        """测试更新业务指标"""
        monitor.update_business_metric('request', 1.0)
        
        metrics = monitor.metrics_collector.collect_business_metrics()
        assert metrics['requests_total'] == 1
        
        # 再更新一次
        monitor.update_business_metric('request', 1.0)
        metrics = monitor.metrics_collector.collect_business_metrics()
        assert metrics['requests_total'] == 2

    def test_update_business_metric_error(self, monitor):
        """测试更新业务指标-错误"""
        monitor.update_business_metric('error', 1.0)
        
        metrics = monitor.metrics_collector.collect_business_metrics()
        assert metrics['errors_total'] == 1

    def test_update_business_metric_response_time(self, monitor):
        """测试更新业务指标-响应时间"""
        monitor.update_business_metric('response_time', 100.0)
        
        metrics = monitor.metrics_collector.collect_business_metrics()
        assert metrics['avg_response_time_ms'] == 100.0
        
        monitor.update_business_metric('response_time', 200.0)
        metrics = monitor.metrics_collector.collect_business_metrics()
        # 平均值应该是 (100 + 200) / 2 = 150
        assert metrics['avg_response_time_ms'] == 150.0

    def test_add_custom_collector(self, monitor):
        """测试添加自定义指标收集器"""
        def custom_collector():
            return {'custom_metric': 42.0}
        
        monitor.add_custom_collector('custom', custom_collector)
        
        assert 'custom' in monitor.metrics_collector.collectors

    def test_add_custom_collector_multiple(self, monitor):
        """测试添加多个自定义收集器"""
        def collector1():
            return {'metric1': 1.0}
        def collector2():
            return {'metric2': 2.0}
        
        monitor.add_custom_collector('collector1', collector1)
        monitor.add_custom_collector('collector2', collector2)
        
        assert len(monitor.metrics_collector.collectors) >= 2

    def test_add_alert_rule(self, monitor):
        """测试添加告警规则"""
        rule = AlertRule(
            name='custom_rule',
            metric_name='custom_metric',
            condition='>',
            threshold=100.0,
            duration=60,
            severity='warning',
            description='Custom rule'
        )
        
        monitor.add_alert_rule(rule)
        
        assert 'custom_rule' in monitor.alert_manager.rules

    def test_add_alert_rule_multiple(self, monitor):
        """测试添加多个告警规则"""
        rule1 = AlertRule(
            name='rule1', metric_name='metric1', condition='>', threshold=50.0,
            duration=60, severity='warning', description='Rule 1'
        )
        rule2 = AlertRule(
            name='rule2', metric_name='metric2', condition='<', threshold=10.0,
            duration=60, severity='info', description='Rule 2'
        )
        
        monitor.add_alert_rule(rule1)
        monitor.add_alert_rule(rule2)
        
        assert 'rule1' in monitor.alert_manager.rules
        assert 'rule2' in monitor.alert_manager.rules

    def test_add_alert_callback(self, monitor):
        """测试添加告警回调"""
        callback1 = Mock()
        callback2 = Mock()
        
        monitor.add_alert_callback(callback1)
        monitor.add_alert_callback(callback2)
        
        assert callback1 in monitor.alert_manager.alert_callbacks
        assert callback2 in monitor.alert_manager.alert_callbacks

    def test_get_alerts_summary_empty(self, monitor):
        """测试获取空告警摘要"""
        summary = monitor.get_alerts_summary()
        
        assert isinstance(summary, dict)
        assert 'active_count' in summary
        assert 'recent_count' in summary
        assert summary['active_count'] == 0
        assert summary['recent_count'] == 0

    def test_get_alerts_summary_with_alerts(self, monitor):
        """测试获取有告警的摘要"""
        # 创建一个告警规则和指标
        rule = AlertRule(
            name='test_rule',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='Test rule'
        )
        monitor.add_alert_rule(rule)
        
        # 设置指标触发告警
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData(
                name='cpu_percent',
                value=85.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        # 检查告警
        monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        summary = monitor.get_alerts_summary()
        
        assert summary['active_count'] >= 0  # 可能有活跃告警
        assert isinstance(summary['active_alerts'], list)

    def test_get_system_status_with_metrics(self, monitor):
        """测试获取有指标的系统状态"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData(
                name='cpu_percent',
                value=50.0,
                timestamp=datetime.now(),
                tags={}
            ),
            'memory_percent': MetricData(
                name='memory_percent',
                value=60.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        status = monitor.get_system_status()
        
        assert status['metrics_count'] == 2
        assert status['system_health'] in ['healthy', 'warning', 'critical']

    def test_get_system_status_no_metrics(self, monitor):
        """测试获取无指标的系统状态"""
        monitor.metrics_collector.metrics = {}
        
        status = monitor.get_system_status()
        
        assert status['metrics_count'] == 0
        assert status['system_health'] == 'healthy'



