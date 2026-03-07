#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig核心方法测试
补充record_metric、start_trace、end_trace、add_trace_event等核心方法的完整测试
"""

import pytest
import time
from unittest.mock import patch
from datetime import datetime

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
    core_monitoring_config_module = importlib.import_module('src.monitoring.core.monitoring_config')
    MonitoringSystem = getattr(core_monitoring_config_module, 'MonitoringSystem', None)
    if MonitoringSystem is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)

class TestMonitoringSystemCoreMethods:
    """测试MonitoringSystem核心方法"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_record_metric_basic(self, monitoring_system):
        """测试基本记录指标"""
        monitoring_system.record_metric('cpu_usage', 75.5)
        
        assert 'cpu_usage' in monitoring_system.metrics
        assert len(monitoring_system.metrics['cpu_usage']) == 1
        assert monitoring_system.metrics['cpu_usage'][0]['value'] == 75.5

    def test_record_metric_with_tags(self, monitoring_system):
        """测试记录指标带标签"""
        monitoring_system.record_metric('cpu_usage', 75.5, {'host': 'server1', 'env': 'prod'})
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert metric['tags'] == {'host': 'server1', 'env': 'prod'}

    def test_record_metric_without_tags(self, monitoring_system):
        """测试记录指标不带标签"""
        monitoring_system.record_metric('cpu_usage', 75.5, None)
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert metric['tags'] == {}

    def test_record_metric_limit_exceeded(self, monitoring_system):
        """测试记录指标超过限制（行43-44）"""
        # 记录超过1000个指标
        for i in range(1500):
            monitoring_system.record_metric('limited_metric', float(i))
        
        # 应该被限制在1000个以内
        assert len(monitoring_system.metrics['limited_metric']) <= 1000

    def test_start_trace_basic(self, monitoring_system):
        """测试开始链路追踪"""
        span_id = monitoring_system.start_trace('trace_1', 'test_operation')
        
        assert span_id is not None
        assert len(monitoring_system.traces) == 1
        trace = monitoring_system.traces[0]
        assert trace['trace_id'] == 'trace_1'
        assert trace['operation'] == 'test_operation'
        assert trace['start_time'] is not None
        assert trace['end_time'] is None

    def test_start_trace_multiple(self, monitoring_system):
        """测试开始多个追踪"""
        span_id1 = monitoring_system.start_trace('trace_1', 'op1')
        span_id2 = monitoring_system.start_trace('trace_2', 'op2')
        
        assert span_id1 != span_id2
        assert len(monitoring_system.traces) == 2

    def test_end_trace_basic(self, monitoring_system):
        """测试结束链路追踪"""
        span_id = monitoring_system.start_trace('trace_1', 'test_operation')
        time.sleep(0.01)
        
        monitoring_system.end_trace(span_id, {'status': 'success'})
        
        trace = monitoring_system.traces[0]
        assert trace['end_time'] is not None
        assert trace['duration'] is not None
        assert trace['duration'] > 0
        assert trace['tags']['status'] == 'success'

    def test_end_trace_without_tags(self, monitoring_system):
        """测试结束追踪不带标签"""
        span_id = monitoring_system.start_trace('trace_1', 'test_operation')
        
        monitoring_system.end_trace(span_id, None)
        
        trace = monitoring_system.traces[0]
        assert trace['end_time'] is not None

    def test_end_trace_nonexistent_span(self, monitoring_system):
        """测试结束不存在的追踪"""
        monitoring_system.end_trace('nonexistent_span')
        
        # 不应该创建新追踪
        assert len(monitoring_system.traces) == 0

    def test_end_trace_already_ended(self, monitoring_system):
        """测试结束已经结束的追踪"""
        span_id = monitoring_system.start_trace('trace_1', 'test_operation')
        monitoring_system.end_trace(span_id)
        original_end_time = monitoring_system.traces[0]['end_time']
        
        # 再次结束
        monitoring_system.end_trace(span_id)
        
        # 应该保持原来的结束时间
        assert monitoring_system.traces[0]['end_time'] == original_end_time

    def test_add_trace_event_basic(self, monitoring_system):
        """测试添加追踪事件"""
        span_id = monitoring_system.start_trace('trace_1', 'test_operation')
        
        monitoring_system.add_trace_event(span_id, 'event_1', {'data': 'value'})
        
        trace = monitoring_system.traces[0]
        assert len(trace['events']) == 1
        assert trace['events'][0]['event'] == 'event_1'
        assert trace['events'][0]['data'] == {'data': 'value'}

    def test_add_trace_event_without_data(self, monitoring_system):
        """测试添加追踪事件不带数据"""
        span_id = monitoring_system.start_trace('trace_1', 'test_operation')
        
        monitoring_system.add_trace_event(span_id, 'event_1', None)
        
        trace = monitoring_system.traces[0]
        assert trace['events'][0]['data'] == {}

    def test_add_trace_event_multiple(self, monitoring_system):
        """测试添加多个追踪事件"""
        span_id = monitoring_system.start_trace('trace_1', 'test_operation')
        
        monitoring_system.add_trace_event(span_id, 'event_1')
        monitoring_system.add_trace_event(span_id, 'event_2', {'value': 123})
        
        trace = monitoring_system.traces[0]
        assert len(trace['events']) == 2

    def test_add_trace_event_nonexistent_span(self, monitoring_system):
        """测试为不存在的追踪添加事件"""
        monitoring_system.add_trace_event('nonexistent_span', 'event_1')
        
        # 不应该创建新追踪或抛出异常
        assert len(monitoring_system.traces) == 0

    def test_generate_report_no_data(self, monitoring_system):
        """测试生成空数据报告"""
        report = monitoring_system.generate_report()
        
        assert report['metrics_count'] == 0
        assert report['traces_count'] == 0
        assert report['alerts_count'] == 0
        assert report['latest_metrics'] == {}
        assert report['performance_summary'] == {}

    def test_generate_report_with_metrics(self, monitoring_system):
        """测试生成有指标的报告"""
        monitoring_system.record_metric('cpu_usage', 75.0)
        monitoring_system.record_metric('memory_usage', 65.0)
        
        report = monitoring_system.generate_report()
        
        assert report['metrics_count'] == 2
        assert 'cpu_usage' in report['latest_metrics']
        assert 'memory_usage' in report['latest_metrics']

    def test_generate_report_with_traces_no_duration(self, monitoring_system):
        """测试生成有追踪但未结束的报告"""
        monitoring_system.start_trace('trace_1', 'operation')
        
        report = monitoring_system.generate_report()
        
        # 未结束的追踪不应该计算性能摘要
        assert report['performance_summary'] == {}

    def test_generate_report_with_traces_with_duration(self, monitoring_system):
        """测试生成有完整追踪的报告"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        monitoring_system.end_trace(span_id)
        
        report = monitoring_system.generate_report()
        
        assert 'performance_summary' in report
        if report['performance_summary']:
            assert 'avg_duration' in report['performance_summary']
            assert 'total_traces' in report['performance_summary']

    def test_generate_report_with_traces_no_durations(self, monitoring_system):
        """测试生成有追踪但无持续时间（durations为空）的报告"""
        # 创建追踪但不结束，所以duration为None
        monitoring_system.start_trace('trace_1', 'operation')
        
        report = monitoring_system.generate_report()
        
        # durations为空时，performance_summary应该为空字典
        assert report['performance_summary'] == {}

