#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig链路追踪边界情况测试
补充start_trace、end_trace、add_trace_event方法的边界情况
"""

import pytest
from unittest.mock import patch
import time

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

class TestMonitoringSystemTraceEdgeCases:
    """测试MonitoringSystem链路追踪边界情况"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_end_trace_nonexistent_span_id(self, monitoring_system):
        """测试结束不存在的span_id"""
        # 尝试结束一个不存在的span_id
        monitoring_system.end_trace('nonexistent_span', {'status': 'completed'})
        
        # 应该不会报错，但也不会更新任何追踪
        assert len(monitoring_system.traces) == 0

    def test_end_trace_already_ended(self, monitoring_system):
        """测试结束已经结束的追踪"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        monitoring_system.end_trace(span_id)
        
        # 再次结束同一个追踪
        monitoring_system.end_trace(span_id, {'status': 'retry'})
        
        # 找到对应的追踪
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        # 第一次结束的时间应该保留
        assert trace['end_time'] is not None

    def test_end_trace_multiple_traces_same_operation(self, monitoring_system):
        """测试多个追踪使用相同操作名"""
        span_id1 = monitoring_system.start_trace('trace_1', 'same_operation')
        span_id2 = monitoring_system.start_trace('trace_2', 'same_operation')
        
        monitoring_system.end_trace(span_id1)
        monitoring_system.end_trace(span_id2)
        
        # 两个追踪都应该结束
        trace1 = next((t for t in monitoring_system.traces if t['span_id'] == span_id1), None)
        trace2 = next((t for t in monitoring_system.traces if t['span_id'] == span_id2), None)
        
        assert trace1 is not None
        assert trace2 is not None
        assert trace1['end_time'] is not None
        assert trace2['end_time'] is not None

    def test_add_trace_event_nonexistent_span_id(self, monitoring_system):
        """测试为不存在的span_id添加事件"""
        # 尝试为不存在的span_id添加事件
        monitoring_system.add_trace_event('nonexistent_span', 'test_event', {'data': 'test'})
        
        # 应该不会报错，但也不会添加任何事件
        assert len(monitoring_system.traces) == 0

    def test_add_trace_event_without_data(self, monitoring_system):
        """测试添加事件不带data参数"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        monitoring_system.add_trace_event(span_id, 'test_event')
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 1
        assert trace['events'][0]['data'] == {}

    def test_add_trace_event_with_none_data(self, monitoring_system):
        """测试添加事件data为None"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        monitoring_system.add_trace_event(span_id, 'test_event', None)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 1
        assert trace['events'][0]['data'] == {}

    def test_add_trace_event_multiple_events(self, monitoring_system):
        """测试为同一追踪添加多个事件"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        monitoring_system.add_trace_event(span_id, 'event1', {'step': 1})
        monitoring_system.add_trace_event(span_id, 'event2', {'step': 2})
        monitoring_system.add_trace_event(span_id, 'event3', {'step': 3})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 3

    def test_add_trace_event_after_end_trace(self, monitoring_system):
        """测试在结束追踪后添加事件"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        monitoring_system.end_trace(span_id)
        
        # 结束后仍然可以添加事件
        monitoring_system.add_trace_event(span_id, 'late_event', {'after': 'end'})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 1

    def test_end_trace_with_empty_tags(self, monitoring_system):
        """测试结束追踪时tags为空"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        monitoring_system.end_trace(span_id, {})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert trace['tags'] == {}

    def test_end_trace_with_none_tags(self, monitoring_system):
        """测试结束追踪时tags为None"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        monitoring_system.end_trace(span_id, None)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert trace['tags'] == {}

    def test_start_trace_empty_operation(self, monitoring_system):
        """测试开始追踪时operation为空字符串"""
        span_id = monitoring_system.start_trace('trace_1', '')
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert trace['operation'] == ''

    def test_trace_duration_calculation(self, monitoring_system):
        """测试追踪持续时间计算"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        # 模拟等待
        time.sleep(0.1)
        
        monitoring_system.end_trace(span_id)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert trace['duration'] is not None
        assert trace['duration'] >= 0.1



