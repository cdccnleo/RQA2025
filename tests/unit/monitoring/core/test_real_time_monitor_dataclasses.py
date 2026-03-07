#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor dataclass测试
补充MetricData、AlertRule、Alert这些dataclass的详细测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from datetime import datetime, timedelta
from typing import Dict, Any

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
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    AlertRule = getattr(core_real_time_monitor_module, 'AlertRule', None)
    Alert = getattr(core_real_time_monitor_module, 'Alert', None)
    if MetricData is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMetricDataDataclass:
    """测试MetricData dataclass"""

    def test_metric_data_creation(self):
        """测试创建MetricData实例"""
        metric = MetricData(
            name='cpu_percent',
            value=85.5,
            timestamp=datetime.now(),
            tags={'type': 'system'},
            metadata={'source': 'psutil'}
        )
        
        assert metric.name == 'cpu_percent'
        assert metric.value == 85.5
        assert isinstance(metric.timestamp, datetime)
        assert metric.tags == {'type': 'system'}
        assert metric.metadata == {'source': 'psutil'}

    def test_metric_data_default_tags(self):
        """测试MetricData默认tags为空字典"""
        metric = MetricData(
            name='test_metric',
            value=42.0,
            timestamp=datetime.now()
        )
        
        assert metric.tags == {}
        assert isinstance(metric.tags, dict)

    def test_metric_data_default_metadata(self):
        """测试MetricData默认metadata为空字典"""
        metric = MetricData(
            name='test_metric',
            value=42.0,
            timestamp=datetime.now()
        )
        
        assert metric.metadata == {}
        assert isinstance(metric.metadata, dict)

    def test_metric_data_with_empty_tags(self):
        """测试MetricData使用空tags字典"""
        metric = MetricData(
            name='test_metric',
            value=42.0,
            timestamp=datetime.now(),
            tags={}
        )
        
        assert metric.tags == {}

    def test_metric_data_with_empty_metadata(self):
        """测试MetricData使用空metadata字典"""
        metric = MetricData(
            name='test_metric',
            value=42.0,
            timestamp=datetime.now(),
            metadata={}
        )
        
        assert metric.metadata == {}

    def test_metric_data_immutable_fields(self):
        """测试MetricData字段可以被修改（dataclass默认可变）"""
        metric = MetricData(
            name='test_metric',
            value=42.0,
            timestamp=datetime.now()
        )
        
        # dataclass默认是可变的
        metric.value = 50.0
        assert metric.value == 50.0
        
        metric.tags['new_key'] = 'new_value'
        assert 'new_key' in metric.tags

    def test_metric_data_timestamp_required(self):
        """测试MetricData的timestamp字段是必需的"""
        # timestamp字段在类型注解中不是Optional，所以不能为None
        # 但在实际运行时，Python不会强制类型检查
        # 这个测试主要验证timestamp字段的存在性
        metric = MetricData(
            name='test',
            value=1.0,
            timestamp=datetime.now()
        )
        assert metric.timestamp is not None
        assert isinstance(metric.timestamp, datetime)

    def test_metric_data_negative_value(self):
        """测试MetricData可以接受负数值"""
        metric = MetricData(
            name='temperature',
            value=-10.5,
            timestamp=datetime.now()
        )
        
        assert metric.value == -10.5

    def test_metric_data_zero_value(self):
        """测试MetricData可以接受零值"""
        metric = MetricData(
            name='count',
            value=0.0,
            timestamp=datetime.now()
        )
        
        assert metric.value == 0.0

    def test_metric_data_large_value(self):
        """测试MetricData可以接受很大的值"""
        metric = MetricData(
            name='large_metric',
            value=1e10,
            timestamp=datetime.now()
        )
        
        assert metric.value == 1e10


class TestAlertRuleDataclass:
    """测试AlertRule dataclass"""

    def test_alert_rule_creation(self):
        """测试创建AlertRule实例"""
        rule = AlertRule(
            name='cpu_high',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='CPU使用率过高',
            enabled=True
        )
        
        assert rule.name == 'cpu_high'
        assert rule.metric_name == 'cpu_percent'
        assert rule.condition == '>'
        assert rule.threshold == 80.0
        assert rule.duration == 60
        assert rule.severity == 'warning'
        assert rule.description == 'CPU使用率过高'
        assert rule.enabled == True

    def test_alert_rule_default_enabled(self):
        """测试AlertRule默认enabled为True"""
        rule = AlertRule(
            name='test_rule',
            metric_name='test_metric',
            condition='>',
            threshold=50.0,
            duration=60,
            severity='info',
            description='Test rule'
        )
        
        assert rule.enabled == True

    def test_alert_rule_disabled(self):
        """测试AlertRule可以设置为禁用"""
        rule = AlertRule(
            name='test_rule',
            metric_name='test_metric',
            condition='>',
            threshold=50.0,
            duration=60,
            severity='info',
            description='Test rule',
            enabled=False
        )
        
        assert rule.enabled == False

    def test_alert_rule_condition_types(self):
        """测试AlertRule支持所有条件类型"""
        conditions = ['>', '<', '>=', '<=', '==']
        
        for condition in conditions:
            rule = AlertRule(
                name=f'rule_{condition}',
                metric_name='test',
                condition=condition,
                threshold=50.0,
                duration=60,
                severity='info',
                description='Test'
            )
            assert rule.condition == condition

    def test_alert_rule_negative_threshold(self):
        """测试AlertRule可以接受负阈值"""
        rule = AlertRule(
            name='temp_low',
            metric_name='temperature',
            condition='<',
            threshold=-10.0,
            duration=60,
            severity='warning',
            description='Temperature too low'
        )
        
        assert rule.threshold == -10.0

    def test_alert_rule_zero_threshold(self):
        """测试AlertRule可以接受零阈值"""
        rule = AlertRule(
            name='zero_rule',
            metric_name='count',
            condition='==',
            threshold=0.0,
            duration=60,
            severity='info',
            description='Count is zero'
        )
        
        assert rule.threshold == 0.0

    def test_alert_rule_duration_positive(self):
        """测试AlertRule duration必须为正数"""
        rule = AlertRule(
            name='test_rule',
            metric_name='test',
            condition='>',
            threshold=50.0,
            duration=30,
            severity='info',
            description='Test'
        )
        
        assert rule.duration > 0

    def test_alert_rule_severity_levels(self):
        """测试AlertRule支持不同严重级别"""
        severities = ['info', 'warning', 'error', 'critical']
        
        for severity in severities:
            rule = AlertRule(
                name=f'rule_{severity}',
                metric_name='test',
                condition='>',
                threshold=50.0,
                duration=60,
                severity=severity,
                description='Test'
            )
            assert rule.severity == severity


class TestAlertDataclass:
    """测试Alert dataclass"""

    def test_alert_creation(self):
        """测试创建Alert实例"""
        now = datetime.now()
        alert = Alert(
            rule_name='cpu_high',
            metric_name='cpu_percent',
            current_value=85.0,
            threshold=80.0,
            severity='warning',
            message='CPU使用率过高: 85.0 > 80.0',
            timestamp=now
        )
        
        assert alert.rule_name == 'cpu_high'
        assert alert.metric_name == 'cpu_percent'
        assert alert.current_value == 85.0
        assert alert.threshold == 80.0
        assert alert.severity == 'warning'
        assert alert.message == 'CPU使用率过高: 85.0 > 80.0'
        assert alert.timestamp == now

    def test_alert_default_resolved(self):
        """测试Alert默认resolved为False"""
        alert = Alert(
            rule_name='test_rule',
            metric_name='test_metric',
            current_value=50.0,
            threshold=40.0,
            severity='info',
            message='Test alert',
            timestamp=datetime.now()
        )
        
        assert alert.resolved == False

    def test_alert_default_resolved_at(self):
        """测试Alert默认resolved_at为None"""
        alert = Alert(
            rule_name='test_rule',
            metric_name='test_metric',
            current_value=50.0,
            threshold=40.0,
            severity='info',
            message='Test alert',
            timestamp=datetime.now()
        )
        
        assert alert.resolved_at is None

    def test_alert_resolved(self):
        """测试Alert可以设置为已解决"""
        alert = Alert(
            rule_name='test_rule',
            metric_name='test_metric',
            current_value=50.0,
            threshold=40.0,
            severity='info',
            message='Test alert',
            timestamp=datetime.now(),
            resolved=True
        )
        
        assert alert.resolved == True

    def test_alert_resolved_at_set(self):
        """测试Alert可以设置resolved_at时间"""
        now = datetime.now()
        alert = Alert(
            rule_name='test_rule',
            metric_name='test_metric',
            current_value=50.0,
            threshold=40.0,
            severity='info',
            message='Test alert',
            timestamp=datetime.now() - timedelta(minutes=10),
            resolved=True,
            resolved_at=now
        )
        
        assert alert.resolved_at == now

    def test_alert_current_value_equals_threshold(self):
        """测试Alert的current_value可以等于threshold"""
        alert = Alert(
            rule_name='test_rule',
            metric_name='test_metric',
            current_value=50.0,
            threshold=50.0,
            severity='info',
            message='Value equals threshold',
            timestamp=datetime.now()
        )
        
        assert alert.current_value == alert.threshold

    def test_alert_current_value_greater_than_threshold(self):
        """测试Alert的current_value可以大于threshold"""
        alert = Alert(
            rule_name='test_rule',
            metric_name='test_metric',
            current_value=60.0,
            threshold=50.0,
            severity='warning',
            message='Value exceeds threshold',
            timestamp=datetime.now()
        )
        
        assert alert.current_value > alert.threshold

    def test_alert_current_value_less_than_threshold(self):
        """测试Alert的current_value可以小于threshold（异常情况）"""
        alert = Alert(
            rule_name='test_rule',
            metric_name='test_metric',
            current_value=40.0,
            threshold=50.0,
            severity='info',
            message='Value below threshold',
            timestamp=datetime.now()
        )
        
        assert alert.current_value < alert.threshold

    def test_alert_empty_message(self):
        """测试Alert可以接受空消息"""
        alert = Alert(
            rule_name='test_rule',
            metric_name='test_metric',
            current_value=50.0,
            threshold=40.0,
            severity='info',
            message='',
            timestamp=datetime.now()
        )
        
        assert alert.message == ''

    def test_alert_long_message(self):
        """测试Alert可以接受很长的消息"""
        long_message = 'A' * 1000
        alert = Alert(
            rule_name='test_rule',
            metric_name='test_metric',
            current_value=50.0,
            threshold=40.0,
            severity='info',
            message=long_message,
            timestamp=datetime.now()
        )
        
        assert len(alert.message) == 1000

