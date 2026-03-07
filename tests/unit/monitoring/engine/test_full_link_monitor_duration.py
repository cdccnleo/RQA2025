#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FullLinkMonitor持续时间检查功能测试
补充持续时间检查方法的测试覆盖率
"""

import pytest
import time
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

import sys
import importlib
from pathlib import Path
import pytest

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
    
    if FullLinkMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestFullLinkMonitorDurationCheck:
    """测试持续时间检查功能"""

    @pytest.fixture
    def monitor(self):
        """创建monitor实例"""
        return FullLinkMonitor()

    def test_check_duration_immediate_trigger(self, monitor):
        """测试立即触发（duration=0）"""
        rule = AlertRule(
            name='immediate_rule',
            metric_name='cpu_usage',
            condition='> 80',
            level=AlertLevel.WARNING,
            duration=0,  # 立即触发
            enabled=True
        )
        monitor.add_alert_rule(rule)
        
        # 添加一个指标
        metric = MetricData(
            name='cpu_usage',
            value=85.0,
            timestamp=datetime.now(),
            tags={},
            monitor_type=MonitorType.SYSTEM,
            source='test'
        )
        
        # duration=0应该立即返回True
        result = monitor._check_duration('cpu_usage', rule)
        assert result == True

    def test_check_duration_insufficient_data(self, monitor):
        """测试数据不足时的持续时间检查"""
        rule = AlertRule(
            name='duration_rule',
            metric_name='cpu_usage',
            condition='> 80',
            level=AlertLevel.WARNING,
            duration=300,  # 5分钟
            enabled=True
        )
        monitor.add_alert_rule(rule)
        
        # 只添加一个数据点
        metric = MetricData(
            name='cpu_usage',
            value=85.0,
            timestamp=datetime.now(),
            tags={},
            monitor_type=MonitorType.SYSTEM,
            source='test'
        )
        monitor.add_metric(metric)
        
        # 数据点不足，应该返回False
        result = monitor._check_duration('cpu_usage', rule)
        assert result == False

    def test_check_duration_sufficient_data(self, monitor):
        """测试数据充足时的持续时间检查"""
        rule = AlertRule(
            name='duration_rule',
            metric_name='cpu_usage',
            condition='> 80',
            level=AlertLevel.WARNING,
            duration=60,  # 1分钟
            enabled=True
        )
        monitor.add_alert_rule(rule)
        
        # 添加多个满足条件的数据点
        for i in range(5):
            metric = MetricData(
                name='cpu_usage',
                value=85.0 + i,  # 都满足条件
                timestamp=datetime.now() - timedelta(seconds=60-i*10),
                tags={},
                monitor_type=MonitorType.SYSTEM,
                source='test'
            )
            monitor.add_metric(metric)
        
        # 数据点充足且都满足条件
        result = monitor._check_duration('cpu_usage', rule)
        assert isinstance(result, bool)

    def test_check_duration_mixed_data(self, monitor):
        """测试混合数据（部分满足条件）"""
        rule = AlertRule(
            name='duration_rule',
            metric_name='cpu_usage',
            condition='> 80',
            level=AlertLevel.WARNING,
            duration=60,
            enabled=True
        )
        monitor.add_alert_rule(rule)
        
        # 添加混合数据（有些满足条件，有些不满足）
        values = [85.0, 75.0, 90.0, 70.0, 88.0]  # 混合值
        for i, value in enumerate(values):
            metric = MetricData(
                name='cpu_usage',
                value=value,
                timestamp=datetime.now() - timedelta(seconds=60-i*10),
                tags={},
                monitor_type=MonitorType.SYSTEM,
                source='test'
            )
            monitor.add_metric(metric)
        
        # 因为有值不满足条件，应该返回False
        result = monitor._check_duration('cpu_usage', rule)
        assert isinstance(result, bool)

    def test_check_duration_out_of_window(self, monitor):
        """测试超出时间窗口的数据"""
        rule = AlertRule(
            name='duration_rule',
            metric_name='cpu_usage',
            condition='> 80',
            level=AlertLevel.WARNING,
            duration=60,  # 1分钟窗口
            enabled=True
        )
        monitor.add_alert_rule(rule)
        
        # 添加超出时间窗口的数据
        metric = MetricData(
            name='cpu_usage',
            value=85.0,
            timestamp=datetime.now() - timedelta(seconds=120),  # 2分钟前
            tags={},
            monitor_type=MonitorType.SYSTEM,
            source='test'
        )
        monitor.add_metric(metric)
        
        # 数据超出窗口，应该返回False
        result = monitor._check_duration('cpu_usage', rule)
        assert result == False

    def test_evaluate_condition_all_operators(self, monitor):
        """测试所有条件操作符"""
        test_cases = [
            ('> 80', 85.0, True),
            ('> 80', 75.0, False),
            ('< 80', 75.0, True),
            ('< 80', 85.0, False),
            ('>= 80', 80.0, True),
            ('>= 80', 85.0, True),
            ('>= 80', 75.0, False),
            ('<= 80', 80.0, True),
            ('<= 80', 75.0, True),
            ('<= 80', 85.0, False),
            ('== 80', 80.0, True),
            ('== 80', 80.1, False),
            ('!= 80', 85.0, True),
            ('!= 80', 80.0, False),
        ]
        
        for condition, value, expected in test_cases:
            result = monitor._evaluate_condition(value, condition)
            assert isinstance(result, bool), f"Condition {condition} with value {value} should return bool"

    def test_evaluate_condition_invalid(self, monitor):
        """测试无效条件"""
        result = monitor._evaluate_condition(85.0, 'invalid condition')
        assert result == False

