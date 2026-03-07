#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor系统状态详细测试
补充get_system_status和get_alerts_summary的详细测试
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
    RealTimeMonitor = getattr(core_real_time_monitor_module, 'RealTimeMonitor', None)
    AlertRule = getattr(core_real_time_monitor_module, 'AlertRule', None)
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    Alert = getattr(core_real_time_monitor_module, 'Alert', None)
    if RealTimeMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestRealTimeMonitorSystemStatusDetailed:
    """测试RealTimeMonitor类系统状态相关方法的详细功能"""

    @pytest.fixture
    def monitor(self):
        """创建RealTimeMonitor实例"""
        return RealTimeMonitor()

    def test_get_system_status_structure(self, monitor):
        """测试系统状态结构完整性"""
        status = monitor.get_system_status()
        
        assert isinstance(status, dict)
        assert 'timestamp' in status
        assert 'system_health' in status
        assert 'active_alerts' in status
        assert 'metrics_count' in status

    def test_get_system_status_timestamp_format(self, monitor):
        """测试系统状态时间戳格式"""
        status = monitor.get_system_status()
        
        # 验证时间戳是ISO格式字符串
        assert isinstance(status['timestamp'], str)
        # 应该能够解析为datetime
        from datetime import datetime
        datetime.fromisoformat(status['timestamp'].replace('Z', '+00:00'))

    def test_get_system_status_healthy_when_all_normal(self, monitor):
        """测试所有指标正常时系统健康状态为healthy"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 50.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 60.0, datetime.now(), {})
        }
        
        status = monitor.get_system_status()
        
        assert status['system_health'] == 'healthy'

    def test_get_system_status_critical_when_cpu_over_90(self, monitor):
        """测试CPU超过90%时系统健康状态为critical"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 95.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 60.0, datetime.now(), {})
        }
        
        status = monitor.get_system_status()
        
        assert status['system_health'] == 'critical'

    def test_get_system_status_critical_when_cpu_exactly_90(self, monitor):
        """测试CPU刚好90%时系统健康状态（应该不触发，因为条件是>90）"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 90.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 60.0, datetime.now(), {})
        }
        
        status = monitor.get_system_status()
        
        # 90.0 不大于 90，所以不应该是critical
        assert status['system_health'] != 'critical'

    def test_get_system_status_critical_cpu_priority_over_memory(self, monitor):
        """测试CPU优先级高于内存（CPU critical时即使内存正常也是critical）"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 95.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 95.0, datetime.now(), {})
        }
        
        status = monitor.get_system_status()
        
        # CPU priority: critical
        assert status['system_health'] == 'critical'

    def test_get_system_status_warning_when_memory_over_90(self, monitor):
        """测试内存超过90%时系统健康状态为warning（CPU正常）"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 80.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 95.0, datetime.now(), {})
        }
        
        status = monitor.get_system_status()
        
        assert status['system_health'] == 'warning'

    def test_get_system_status_warning_when_memory_exactly_90(self, monitor):
        """测试内存刚好90%时系统健康状态（应该不触发，因为条件是>90）"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 80.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 90.0, datetime.now(), {})
        }
        
        status = monitor.get_system_status()
        
        # 90.0 不大于 90，所以不应该是warning
        assert status['system_health'] == 'healthy'

    def test_get_system_status_missing_cpu_metric(self, monitor):
        """测试缺少CPU指标时使用默认值"""
        monitor.metrics_collector.metrics = {
            'memory_percent': MetricData('memory_percent', 60.0, datetime.now(), {})
        }
        
        status = monitor.get_system_status()
        
        # 缺少CPU指标应该不影响健康状态判断
        assert status['system_health'] in ['healthy', 'warning', 'critical']

    def test_get_system_status_missing_memory_metric(self, monitor):
        """测试缺少内存指标时使用默认值"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 50.0, datetime.now(), {})
        }
        
        status = monitor.get_system_status()
        
        # 缺少内存指标应该不影响健康状态判断
        assert status['system_health'] in ['healthy', 'warning', 'critical']

    def test_get_system_status_active_alerts_count(self, monitor):
        """测试系统状态中活跃告警数量"""
        # 添加告警规则并触发告警
        rule = AlertRule(
            name='test_rule',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='Test'
        )
        monitor.add_alert_rule(rule)
        
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {})
        }
        monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        status = monitor.get_system_status()
        
        assert 'active_alerts' in status
        assert isinstance(status['active_alerts'], int)
        assert status['active_alerts'] >= 0

    def test_get_system_status_metrics_count(self, monitor):
        """测试系统状态中指标数量"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 50.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 60.0, datetime.now(), {}),
            'disk_usage': MetricData('disk_usage', 70.0, datetime.now(), {})
        }
        
        status = monitor.get_system_status()
        
        assert status['metrics_count'] == 3

    def test_get_system_status_metrics_count_zero(self, monitor):
        """测试系统状态中指标数量为零"""
        monitor.metrics_collector.metrics = {}
        
        status = monitor.get_system_status()
        
        assert status['metrics_count'] == 0

    def test_get_alerts_summary_structure(self, monitor):
        """测试告警摘要结构完整性"""
        summary = monitor.get_alerts_summary()
        
        assert isinstance(summary, dict)
        assert 'active_count' in summary
        assert 'active_alerts' in summary
        assert 'recent_count' in summary
        assert 'recent_alerts' in summary

    def test_get_alerts_summary_active_alerts_structure(self, monitor):
        """测试告警摘要中活跃告警结构"""
        # 创建并触发一个告警
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
        
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {})
        }
        monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        summary = monitor.get_alerts_summary()
        
        assert isinstance(summary['active_alerts'], list)
        if len(summary['active_alerts']) > 0:
            alert = summary['active_alerts'][0]
            assert 'rule_name' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'timestamp' in alert

    def test_get_alerts_summary_recent_alerts_structure(self, monitor):
        """测试告警摘要中最近告警结构"""
        # 创建并触发告警（会进入历史）
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
        
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {})
        }
        monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        summary = monitor.get_alerts_summary()
        
        assert isinstance(summary['recent_alerts'], list)
        if len(summary['recent_alerts']) > 0:
            alert = summary['recent_alerts'][0]
            assert 'rule_name' in alert
            assert 'severity' in alert
            assert 'message' in alert
            assert 'resolved' in alert
            assert 'timestamp' in alert

    def test_get_alerts_summary_recent_alerts_limit_10(self, monitor):
        """测试告警摘要中最近告警最多显示10个"""
        # 创建规则
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
        
        # 触发多个告警（通过多次检查，每次触发新的告警）
        for i in range(15):
            monitor.metrics_collector.metrics = {
                'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now() - timedelta(minutes=15-i), {})
            }
            # 先解决上一个告警
            if i > 0:
                monitor.metrics_collector.metrics['cpu_percent'].value = 50.0
                monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
            # 再触发新告警
            monitor.metrics_collector.metrics['cpu_percent'].value = 85.0
            monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        summary = monitor.get_alerts_summary()
        
        # 最近告警应该最多10个
        assert len(summary['recent_alerts']) <= 10

    def test_get_alerts_summary_active_count_accuracy(self, monitor):
        """测试告警摘要中活跃告警数量准确性"""
        # 添加多个告警规则
        rule1 = AlertRule(
            name='rule1', metric_name='cpu_percent', condition='>',
            threshold=80.0, duration=60, severity='warning', description='Rule 1'
        )
        rule2 = AlertRule(
            name='rule2', metric_name='memory_percent', condition='>',
            threshold=70.0, duration=60, severity='warning', description='Rule 2'
        )
        
        monitor.add_alert_rule(rule1)
        monitor.add_alert_rule(rule2)
        
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 75.0, datetime.now(), {})
        }
        monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        summary = monitor.get_alerts_summary()
        
        assert summary['active_count'] == len(summary['active_alerts'])

    def test_get_alerts_summary_recent_count_accuracy(self, monitor):
        """测试告警摘要中最近告警数量准确性"""
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
        
        # 触发一个告警
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {})
        }
        monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        summary = monitor.get_alerts_summary()
        
        # recent_count应该反映1小时内历史告警的总数（可能大于显示的10个）
        assert summary['recent_count'] >= len(summary['recent_alerts'])

    def test_get_alerts_summary_timestamp_iso_format(self, monitor):
        """测试告警摘要中时间戳为ISO格式"""
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
        
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {})
        }
        monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        summary = monitor.get_alerts_summary()
        
        if len(summary['active_alerts']) > 0:
            timestamp = summary['active_alerts'][0]['timestamp']
            assert isinstance(timestamp, str)
            # 应该能够解析
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))

    def test_get_alerts_summary_resolved_flag(self, monitor):
        """测试告警摘要中已解决标志"""
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
        
        # 触发告警
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {})
        }
        monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        # 解决告警
        monitor.metrics_collector.metrics['cpu_percent'].value = 50.0
        monitor.alert_manager.check_alerts(monitor.metrics_collector.metrics)
        
        summary = monitor.get_alerts_summary()
        
        # 在recent_alerts中应该能看到已解决的告警
        if len(summary['recent_alerts']) > 0:
            # 最近告警应该包含resolved字段
            assert 'resolved' in summary['recent_alerts'][0]

    def test_get_alerts_summary_empty_when_no_alerts(self, monitor):
        """测试无告警时告警摘要为空"""
        summary = monitor.get_alerts_summary()
        
        assert summary['active_count'] == 0
        assert summary['recent_count'] == 0
        assert len(summary['active_alerts']) == 0
        assert len(summary['recent_alerts']) == 0

    def test_get_current_metrics_returns_copy(self, monitor):
        """测试get_current_metrics返回副本"""
        monitor.metrics_collector.metrics = {
            'test_metric': MetricData('test_metric', 42.0, datetime.now(), {})
        }
        
        metrics1 = monitor.get_current_metrics()
        metrics2 = monitor.get_current_metrics()
        
        # 应该是不同的对象（副本）
        assert metrics1 is not metrics2
        # 但内容应该相同
        assert metrics1 == metrics2
        
        # 修改副本不应该影响原始数据
        metrics1['new_metric'] = MetricData('new_metric', 1.0, datetime.now(), {})
        assert 'new_metric' not in monitor.metrics_collector.metrics

