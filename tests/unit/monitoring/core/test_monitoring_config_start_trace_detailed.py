#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig开始链路追踪详细测试
补充start_trace方法的详细测试，包括span_id生成、operation处理、边界情况等
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

class TestMonitoringSystemStartTraceDetailed:
    """测试MonitoringSystem开始链路追踪的详细功能"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_start_trace_returns_span_id(self, monitoring_system):
        """测试start_trace返回span_id"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        assert span_id is not None
        assert isinstance(span_id, str)
        assert len(span_id) > 0

    def test_start_trace_span_id_format(self, monitoring_system):
        """测试span_id格式"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        # span_id格式应该是 trace_id-index
        assert span_id.startswith('trace_1-')
        assert span_id.count('-') == 1

    def test_start_trace_span_id_index(self, monitoring_system):
        """测试span_id中的index递增"""
        span_id1 = monitoring_system.start_trace('trace_1', 'operation')
        span_id2 = monitoring_system.start_trace('trace_2', 'operation')
        span_id3 = monitoring_system.start_trace('trace_3', 'operation')
        
        # 每个trace的index应该基于traces列表的长度
        assert span_id1 == 'trace_1-0'
        assert span_id2 == 'trace_2-1'
        assert span_id3 == 'trace_3-2'

    def test_start_trace_creates_trace_entry(self, monitoring_system):
        """测试start_trace创建追踪条目"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        assert len(monitoring_system.traces) == 1
        trace = monitoring_system.traces[0]
        assert trace['span_id'] == span_id

    def test_start_trace_trace_structure(self, monitoring_system):
        """测试trace结构完整性"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        trace = monitoring_system.traces[0]
        
        # 验证所有必需字段存在
        assert 'trace_id' in trace
        assert 'span_id' in trace
        assert 'operation' in trace
        assert 'start_time' in trace
        assert 'end_time' in trace
        assert 'duration' in trace
        assert 'tags' in trace
        assert 'events' in trace

    def test_start_trace_trace_values(self, monitoring_system):
        """测试trace字段值"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        trace = monitoring_system.traces[0]
        
        assert trace['trace_id'] == 'trace_1'
        assert trace['span_id'] == span_id
        assert trace['operation'] == 'operation'
        assert trace['end_time'] is None
        assert trace['duration'] is None
        assert trace['tags'] == {}
        assert trace['events'] == []

    def test_start_trace_start_time_set(self, monitoring_system):
        """测试start_time被设置"""
        with patch('time.time', return_value=1234567890.0):
            span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        trace = monitoring_system.traces[0]
        assert trace['start_time'] == 1234567890.0

    def test_start_trace_multiple_traces(self, monitoring_system):
        """测试创建多个traces"""
        span_id1 = monitoring_system.start_trace('trace_1', 'op1')
        span_id2 = monitoring_system.start_trace('trace_2', 'op2')
        span_id3 = monitoring_system.start_trace('trace_3', 'op3')
        
        assert len(monitoring_system.traces) == 3
        assert monitoring_system.traces[0]['span_id'] == span_id1
        assert monitoring_system.traces[1]['span_id'] == span_id2
        assert monitoring_system.traces[2]['span_id'] == span_id3

    def test_start_trace_same_trace_id_different_span(self, monitoring_system):
        """测试相同trace_id创建不同的span"""
        span_id1 = monitoring_system.start_trace('same_trace', 'op1')
        span_id2 = monitoring_system.start_trace('same_trace', 'op2')
        
        # 应该有2个不同的span_id
        assert span_id1 != span_id2
        assert span_id1.startswith('same_trace-')
        assert span_id2.startswith('same_trace-')

    def test_start_trace_empty_operation(self, monitoring_system):
        """测试空字符串operation"""
        span_id = monitoring_system.start_trace('trace_1', '')
        
        trace = monitoring_system.traces[0]
        assert trace['operation'] == ''

    def test_start_trace_special_characters_in_trace_id(self, monitoring_system):
        """测试trace_id包含特殊字符"""
        special_trace_id = 'trace.with-special_chars@123'
        span_id = monitoring_system.start_trace(special_trace_id, 'operation')
        
        trace = monitoring_system.traces[0]
        assert trace['trace_id'] == special_trace_id
        assert span_id.startswith(special_trace_id + '-')

    def test_start_trace_special_characters_in_operation(self, monitoring_system):
        """测试operation包含特殊字符"""
        special_operation = 'operation.with-special_chars@123'
        span_id = monitoring_system.start_trace('trace_1', special_operation)
        
        trace = monitoring_system.traces[0]
        assert trace['operation'] == special_operation

    def test_start_trace_empty_trace_id(self, monitoring_system):
        """测试空字符串trace_id"""
        span_id = monitoring_system.start_trace('', 'operation')
        
        trace = monitoring_system.traces[0]
        assert trace['trace_id'] == ''
        assert span_id.startswith('-')

    def test_start_trace_unicode_characters(self, monitoring_system):
        """测试unicode字符"""
        unicode_trace_id = '追踪_测试'
        unicode_operation = '操作_测试'
        
        span_id = monitoring_system.start_trace(unicode_trace_id, unicode_operation)
        
        trace = monitoring_system.traces[0]
        assert trace['trace_id'] == unicode_trace_id
        assert trace['operation'] == unicode_operation
        assert unicode_trace_id in span_id

    def test_start_trace_long_strings(self, monitoring_system):
        """测试很长的字符串"""
        long_trace_id = 'a' * 1000
        long_operation = 'b' * 1000
        
        span_id = monitoring_system.start_trace(long_trace_id, long_operation)
        
        trace = monitoring_system.traces[0]
        assert trace['trace_id'] == long_trace_id
        assert trace['operation'] == long_operation

    def test_start_trace_tags_initialized_empty(self, monitoring_system):
        """测试tags初始化为空字典"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        trace = monitoring_system.traces[0]
        assert trace['tags'] == {}
        assert isinstance(trace['tags'], dict)

    def test_start_trace_events_initialized_empty(self, monitoring_system):
        """测试events初始化为空列表"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        trace = monitoring_system.traces[0]
        assert trace['events'] == []
        assert isinstance(trace['events'], list)

    def test_start_trace_end_time_initialized_none(self, monitoring_system):
        """测试end_time初始化为None"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        trace = monitoring_system.traces[0]
        assert trace['end_time'] is None

    def test_start_trace_duration_initialized_none(self, monitoring_system):
        """测试duration初始化为None"""
        span_id = monitoring_system.start_trace('trace_1', 'operation')
        
        trace = monitoring_system.traces[0]
        assert trace['duration'] is None

    def test_start_trace_index_based_on_existing_traces(self, monitoring_system):
        """测试span_id的index基于现有traces数量"""
        # 创建一些traces
        for i in range(5):
            monitoring_system.start_trace(f'trace_{i}', 'operation')
        
        # 新的trace的index应该是5
        span_id = monitoring_system.start_trace('new_trace', 'operation')
        assert span_id == 'new_trace-5'

    def test_start_trace_concurrent_traces(self, monitoring_system):
        """测试并发创建traces（模拟）"""
        span_ids = []
        for i in range(10):
            span_id = monitoring_system.start_trace(f'trace_{i}', f'op_{i}')
            span_ids.append(span_id)
        
        # 验证所有span_id都是唯一的
        assert len(set(span_ids)) == len(span_ids)
        
        # 验证所有traces都被创建
        assert len(monitoring_system.traces) == 10

    def test_start_trace_trace_order(self, monitoring_system):
        """测试traces的创建顺序"""
        span_ids = []
        for i in range(5):
            span_id = monitoring_system.start_trace(f'trace_{i}', f'op_{i}')
            span_ids.append(span_id)
        
        # 验证traces按创建顺序排列
        for i, trace in enumerate(monitoring_system.traces):
            assert trace['span_id'] == span_ids[i]


