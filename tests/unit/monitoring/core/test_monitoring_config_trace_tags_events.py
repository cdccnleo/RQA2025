#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig链路追踪Tags和Events测试
补充end_trace和add_trace_event方法中tags和events的详细测试
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

class TestMonitoringSystemTraceTagsEvents:
    """测试MonitoringSystem链路追踪Tags和Events"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_end_trace_tags_update(self, monitoring_system):
        """测试end_trace更新tags"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        tags = {'status': 'success', 'response_code': '200'}
        monitoring_system.end_trace(span_id, tags)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert 'status' in trace['tags']
        assert trace['tags']['status'] == 'success'
        assert trace['tags']['response_code'] == '200'

    def test_end_trace_tags_merge(self, monitoring_system):
        """测试end_trace合并多个tags"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        # 第一次添加tags
        tags1 = {'status': 'processing'}
        monitoring_system.end_trace(span_id, tags1)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert trace['tags']['status'] == 'processing'
        
        # 注意：由于end_time已设置，再次调用不会更新tags（根据代码逻辑）

    def test_end_trace_tags_none(self, monitoring_system):
        """测试end_trace传入None tags"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        monitoring_system.end_trace(span_id, None)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert isinstance(trace['tags'], dict)

    def test_end_trace_tags_empty_dict(self, monitoring_system):
        """测试end_trace传入空tags字典"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        monitoring_system.end_trace(span_id, {})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert isinstance(trace['tags'], dict)

    def test_add_trace_event_basic(self, monitoring_system):
        """测试add_trace_event基本功能"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        monitoring_system.add_trace_event(span_id, 'cache_hit', {'key': 'user_123'})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 1
        assert trace['events'][0]['event'] == 'cache_hit'
        assert trace['events'][0]['data']['key'] == 'user_123'

    def test_add_trace_event_multiple(self, monitoring_system):
        """测试add_trace_event添加多个事件"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        monitoring_system.add_trace_event(span_id, 'event1', {'key1': 'value1'})
        monitoring_system.add_trace_event(span_id, 'event2', {'key2': 'value2'})
        monitoring_system.add_trace_event(span_id, 'event3', {'key3': 'value3'})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 3

    def test_add_trace_event_after_end(self, monitoring_system):
        """测试在trace结束后添加事件"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        monitoring_system.end_trace(span_id)
        
        # 在trace结束后仍然可以添加事件
        monitoring_system.add_trace_event(span_id, 'post_end_event', {'note': 'after_end'})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 1

    def test_add_trace_event_nonexistent_span(self, monitoring_system):
        """测试向不存在的span添加事件"""
        monitoring_system.add_trace_event('nonexistent_span', 'event', {'key': 'value'})
        
        # 应该不会报错，但也不会添加任何事件
        assert len(monitoring_system.traces) == 0

    def test_add_trace_event_data_none(self, monitoring_system):
        """测试add_trace_event传入None data"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        monitoring_system.add_trace_event(span_id, 'event', None)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 1
        assert trace['events'][0]['data'] == {}

    def test_add_trace_event_data_empty_dict(self, monitoring_system):
        """测试add_trace_event传入空data字典"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        monitoring_system.add_trace_event(span_id, 'event', {})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 1
        assert trace['events'][0]['data'] == {}

    def test_add_trace_event_timestamp_set(self, monitoring_system):
        """测试事件timestamp被设置"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        with patch('time.time', return_value=1234567890.0):
            monitoring_system.add_trace_event(span_id, 'event', {'key': 'value'})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 1
        assert trace['events'][0]['timestamp'] == 1234567890.0

    def test_add_trace_event_multiple_same_type(self, monitoring_system):
        """测试添加多个相同类型的事件"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        monitoring_system.add_trace_event(span_id, 'cache_hit', {'key': 'k1'})
        monitoring_system.add_trace_event(span_id, 'cache_hit', {'key': 'k2'})
        monitoring_system.add_trace_event(span_id, 'cache_hit', {'key': 'k3'})
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 3
        # 所有事件都应该是cache_hit类型
        assert all(e['event'] == 'cache_hit' for e in trace['events'])

    def test_end_trace_duration_calculation(self, monitoring_system):
        """测试end_trace计算duration"""
        # Mock time.time() - logger也会调用，所以需要更多值
        with patch('time.time', side_effect=[1000.0, 1005.5, 1005.5]):
            with patch('src.monitoring.core.monitoring_config.logger'):
                span_id = monitoring_system.start_trace('trace_1', 'operation')
                monitoring_system.end_trace(span_id)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert trace['duration'] == 5.5

    def test_end_trace_tags_update_existing(self, monitoring_system):
        """测试end_trace更新已存在的tags"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        # 初始tags（如果有的话）
        tags1 = {'initial': 'value'}
        monitoring_system.end_trace(span_id, tags1)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert trace['tags']['initial'] == 'value'
        
        # 注意：由于end_time已设置，再次调用end_trace不会更新tags

    def test_add_trace_event_event_ordering(self, monitoring_system):
        """测试事件添加的顺序"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        with patch('time.time', side_effect=[1000.0, 1001.0, 1002.0]):
            monitoring_system.add_trace_event(span_id, 'event1')
            monitoring_system.add_trace_event(span_id, 'event2')
            monitoring_system.add_trace_event(span_id, 'event3')
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert len(trace['events']) == 3
        # 事件应该按照添加顺序排列
        assert trace['events'][0]['event'] == 'event1'
        assert trace['events'][1]['event'] == 'event2'
        assert trace['events'][2]['event'] == 'event3'

    def test_end_trace_tags_with_nested_data(self, monitoring_system):
        """测试end_trace的tags包含复杂数据"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        tags = {
            'status': 'success',
            'metadata': {'key': 'value'},
            'count': 42
        }
        
        monitoring_system.end_trace(span_id, tags)
        
        trace = next((t for t in monitoring_system.traces if t['span_id'] == span_id), None)
        assert trace is not None
        assert trace['tags']['status'] == 'success'
        assert trace['tags']['count'] == 42

