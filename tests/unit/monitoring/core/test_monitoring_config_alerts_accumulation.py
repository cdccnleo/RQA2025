#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig告警累积测试
补充check_alerts方法中告警累积和多告警场景的测试
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
    monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    MonitoringSystem = getattr(monitoring_config_module, 'MonitoringSystem', None)
    if MonitoringSystem is None:
        pytest.skip("监控配置模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("监控配置模块导入失败", allow_module_level=True)


class TestMonitoringSystemAlertsAccumulation:
    """测试MonitoringSystem告警累积"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_check_alerts_extends_alerts_list(self, monitoring_system):
        """测试check_alerts会将新告警添加到self.alerts列表中"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        
        initial_count = len(monitoring_system.alerts)
        alerts = monitoring_system.check_alerts()
        final_count = len(monitoring_system.alerts)
        
        assert len(alerts) > 0
        assert final_count == initial_count + len(alerts)

    def test_check_alerts_multiple_calls_accumulate(self, monitoring_system):
        """测试多次调用check_alerts会累积告警"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        
        # 第一次调用
        alerts1 = monitoring_system.check_alerts()
        count1 = len(monitoring_system.alerts)
        
        # 第二次调用
        alerts2 = monitoring_system.check_alerts()
        count2 = len(monitoring_system.alerts)
        
        # 告警应该累积
        assert count2 == count1 + len(alerts2)
        assert count2 == 2 * len(alerts1)  # 两次相同的告警

    def test_check_alerts_multiple_alert_types_accumulate(self, monitoring_system):
        """测试多种告警类型会累积"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        monitoring_system.record_metric('memory_usage', 75.0)
        monitoring_system.record_metric('api_response_time', 1500.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 应该有三种类型的告警
        alert_types = {a['type'] for a in alerts}
        assert 'cpu_high' in alert_types
        assert 'memory_high' in alert_types
        assert 'api_slow' in alert_types
        assert len(alerts) == 3

    def test_check_alerts_same_metric_multiple_times(self, monitoring_system):
        """测试同一指标多次触发告警会累积"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        
        # 第一次调用
        alerts1 = monitoring_system.check_alerts()
        
        # 再次调用（指标值不变）
        alerts2 = monitoring_system.check_alerts()
        
        # 每次都会产生新告警
        assert len(alerts2) == len(alerts1)
        assert len(monitoring_system.alerts) == len(alerts1) + len(alerts2)

    def test_check_alerts_initial_empty_alerts(self, monitoring_system):
        """测试初始时alerts列表为空"""
        assert len(monitoring_system.alerts) == 0
        
        # 没有告警条件时
        alerts = monitoring_system.check_alerts()
        assert len(alerts) == 0
        assert len(monitoring_system.alerts) == 0

    def test_check_alerts_alerts_list_grows(self, monitoring_system):
        """测试alerts列表会增长"""
        # 多次添加告警条件
        for i in range(5):
            monitoring_system.record_metric('cpu_usage', 85.0)
            monitoring_system.check_alerts()
        
        # alerts列表应该增长
        assert len(monitoring_system.alerts) > 0

    def test_check_alerts_returned_alerts_match_added(self, monitoring_system):
        """测试返回的告警和添加到alerts列表中的告警一致"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        
        alerts = monitoring_system.check_alerts()
        alerts_in_system = monitoring_system.alerts[-len(alerts):]
        
        # 返回的告警应该和添加到系统中的告警一致
        assert len(alerts) == len(alerts_in_system)
        for alert in alerts:
            assert alert in alerts_in_system

    def test_check_alerts_multiple_metrics_same_time(self, monitoring_system):
        """测试同时记录多个指标并检查告警"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        monitoring_system.record_metric('memory_usage', 75.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 应该有两种类型的告警
        assert len(alerts) >= 2
        alert_types = {a['type'] for a in alerts}
        assert 'cpu_high' in alert_types
        assert 'memory_high' in alert_types

    def test_check_alerts_no_duplicate_in_return(self, monitoring_system):
        """测试单次check_alerts调用返回的告警不重复"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 单次调用返回的告警不应该重复
        alert_types = [a['type'] for a in alerts]
        assert len(alert_types) == len(set(alert_types))

    def test_check_alerts_alert_timestamp_set(self, monitoring_system):
        """测试告警的timestamp被设置"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 每个告警都应该有timestamp
        for alert in alerts:
            assert 'timestamp' in alert
            assert alert['timestamp'] is not None
            # 验证timestamp格式
            try:
                datetime.fromisoformat(alert['timestamp'])
            except ValueError:
                pytest.fail("Invalid timestamp format")

    def test_check_alerts_alert_severity_set(self, monitoring_system):
        """测试告警的severity被正确设置"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        monitoring_system.record_metric('memory_usage', 75.0)
        
        alerts = monitoring_system.check_alerts()
        
        # 验证severity
        for alert in alerts:
            assert 'severity' in alert
            if alert['type'] == 'cpu_high':
                assert alert['severity'] == 'critical'
            elif alert['type'] == 'memory_high':
                assert alert['severity'] == 'warning'

    def test_check_alerts_alert_message_format(self, monitoring_system):
        """测试告警消息格式正确"""
        monitoring_system.record_metric('cpu_usage', 85.5)
        
        alerts = monitoring_system.check_alerts()
        
        cpu_alert = [a for a in alerts if a['type'] == 'cpu_high'][0]
        assert 'message' in cpu_alert
        assert 'CPU使用率过高' in cpu_alert['message']
        assert '85.5' in cpu_alert['message']


