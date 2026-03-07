#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试优化引擎组件
"""

import importlib
import sys
from datetime import datetime
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def optimization_engine_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.optimization_engine"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def engine(optimization_engine_module):
    """创建OptimizationEngine实例"""
    return optimization_engine_module.OptimizationEngine()


def test_initialization(engine):
    """测试初始化"""
    assert engine.optimization_suggestions == []


def test_generate_suggestions(engine):
    """测试生成优化建议"""
    coverage_data = {'coverage_percent': 75.0}
    performance_data = {'response_time_ms': 15.0, 'memory_usage_mb': 2048, 'throughput_tps': 500}
    
    suggestions = engine.generate_suggestions(coverage_data, performance_data)
    
    assert len(suggestions) > 0
    assert len(engine.optimization_suggestions) > 0
    
    # 验证建议类型
    suggestion_types = {s['type'] for s in suggestions}
    assert 'coverage_improvement' in suggestion_types
    assert 'performance_optimization' in suggestion_types
    assert 'memory_optimization' in suggestion_types
    assert 'throughput_optimization' in suggestion_types


def test_generate_coverage_suggestions_low_coverage(engine):
    """测试生成覆盖率建议（低覆盖率）"""
    coverage_data = {'coverage_percent': 70.0}
    
    suggestions = engine._generate_coverage_suggestions(coverage_data)
    
    assert len(suggestions) == 1
    assert suggestions[0]['type'] == 'coverage_improvement'
    assert suggestions[0]['priority'] == 'high'
    assert '70.0%' in suggestions[0]['description']


def test_generate_coverage_suggestions_medium_coverage(engine):
    """测试生成覆盖率建议（中等覆盖率）"""
    coverage_data = {'coverage_percent': 85.0}
    
    suggestions = engine._generate_coverage_suggestions(coverage_data)
    
    assert len(suggestions) == 1
    assert suggestions[0]['type'] == 'coverage_improvement'
    assert suggestions[0]['priority'] == 'medium'
    assert '85.0%' in suggestions[0]['description']


def test_generate_coverage_suggestions_high_coverage(engine):
    """测试生成覆盖率建议（高覆盖率）"""
    coverage_data = {'coverage_percent': 95.0}
    
    suggestions = engine._generate_coverage_suggestions(coverage_data)
    
    assert len(suggestions) == 0


def test_generate_coverage_suggestions_missing_key(engine, capsys):
    """测试生成覆盖率建议（缺少键，使用默认值0）"""
    coverage_data = {}
    
    suggestions = engine._generate_coverage_suggestions(coverage_data)
    
    # 当缺少键时，get返回0，0 < 80，所以会生成建议
    assert len(suggestions) == 1
    assert suggestions[0]['priority'] == 'high'


def test_generate_coverage_suggestions_exception(engine, capsys):
    """测试生成覆盖率建议（异常）"""
    # 创建一个会导致异常的数据
    coverage_data = {'coverage_percent': object()}  # 无法格式化的对象
    
    suggestions = engine._generate_coverage_suggestions(coverage_data)
    
    assert len(suggestions) == 0


def test_check_response_time_suggestions_slow(engine):
    """测试检查响应时间建议（慢响应）"""
    performance_data = {'response_time_ms': 15.0}
    
    suggestions = engine._check_response_time_suggestions(performance_data)
    
    assert len(suggestions) == 1
    assert suggestions[0]['type'] == 'performance_optimization'
    assert suggestions[0]['priority'] == 'medium'
    assert '15.0' in suggestions[0]['description']


def test_check_response_time_suggestions_fast(engine):
    """测试检查响应时间建议（快响应）"""
    performance_data = {'response_time_ms': 5.0}
    
    suggestions = engine._check_response_time_suggestions(performance_data)
    
    assert len(suggestions) == 0


def test_check_response_time_suggestions_missing_key(engine):
    """测试检查响应时间建议（缺少键）"""
    performance_data = {}
    
    suggestions = engine._check_response_time_suggestions(performance_data)
    
    assert len(suggestions) == 0


def test_check_memory_usage_suggestions_high(engine):
    """测试检查内存使用建议（高内存）"""
    performance_data = {'memory_usage_mb': 2048}  # 2GB
    
    suggestions = engine._check_memory_usage_suggestions(performance_data)
    
    assert len(suggestions) == 1
    assert suggestions[0]['type'] == 'memory_optimization'
    assert suggestions[0]['priority'] == 'medium'
    assert '2.0' in suggestions[0]['description']


def test_check_memory_usage_suggestions_low(engine):
    """测试检查内存使用建议（低内存）"""
    performance_data = {'memory_usage_mb': 512}  # 0.5GB
    
    suggestions = engine._check_memory_usage_suggestions(performance_data)
    
    assert len(suggestions) == 0


def test_check_memory_usage_suggestions_missing_key(engine):
    """测试检查内存使用建议（缺少键）"""
    performance_data = {}
    
    suggestions = engine._check_memory_usage_suggestions(performance_data)
    
    assert len(suggestions) == 0


def test_check_throughput_suggestions_low(engine):
    """测试检查吞吐量建议（低吞吐量）"""
    performance_data = {'throughput_tps': 500}
    
    suggestions = engine._check_throughput_suggestions(performance_data)
    
    assert len(suggestions) == 1
    assert suggestions[0]['type'] == 'throughput_optimization'
    assert suggestions[0]['priority'] == 'medium'
    assert '500' in suggestions[0]['description']


def test_check_throughput_suggestions_high(engine):
    """测试检查吞吐量建议（高吞吐量）"""
    performance_data = {'throughput_tps': 2000}
    
    suggestions = engine._check_throughput_suggestions(performance_data)
    
    assert len(suggestions) == 0


def test_check_throughput_suggestions_missing_key(engine):
    """测试检查吞吐量建议（缺少键，使用默认值0）"""
    performance_data = {}
    
    suggestions = engine._check_throughput_suggestions(performance_data)
    
    # 当缺少键时，get返回0，0 < 1000，所以会生成建议
    assert len(suggestions) == 1
    assert suggestions[0]['type'] == 'throughput_optimization'


def test_generate_performance_suggestions(engine):
    """测试生成性能建议"""
    performance_data = {
        'response_time_ms': 15.0,
        'memory_usage_mb': 2048,
        'throughput_tps': 500
    }
    
    suggestions = engine._generate_performance_suggestions(performance_data)
    
    assert len(suggestions) == 3  # 响应时间、内存、吞吐量各一个


def test_generate_performance_suggestions_exception(engine, capsys):
    """测试生成性能建议（异常）"""
    # 创建一个会导致除零错误的数据
    performance_data = {'throughput_tps': 0}
    
    # 模拟_check_throughput_suggestions抛出异常
    def failing_check(*args, **kwargs):
        raise ZeroDivisionError("Division error")
    
    with patch.object(engine, '_check_throughput_suggestions', side_effect=failing_check):
        suggestions = engine._generate_performance_suggestions(performance_data)
        
        # 应该返回部分建议，即使有异常
        assert isinstance(suggestions, list)


def test_print_suggestions(engine, capsys):
    """测试打印优化建议"""
    suggestions = [
        {'priority': 'high', 'title': 'High Priority'},
        {'priority': 'medium', 'title': 'Medium Priority'},
        {'priority': 'low', 'title': 'Low Priority'},
        {'priority': 'unknown', 'title': 'Unknown Priority'}
    ]
    
    engine._print_suggestions(suggestions)
    
    captured = capsys.readouterr()
    assert 'HIGH' in captured.out
    assert 'MEDIUM' in captured.out
    assert 'LOW' in captured.out


def test_get_suggestions_history_no_limit(engine):
    """测试获取建议历史（无限制）"""
    # 添加一些建议
    engine.optimization_suggestions = [
        {'type': 'test1'},
        {'type': 'test2'},
        {'type': 'test3'}
    ]
    
    history = engine.get_suggestions_history()
    
    assert len(history) == 3
    assert history == engine.optimization_suggestions.copy()


def test_get_suggestions_history_with_limit(engine):
    """测试获取建议历史（有限制）"""
    engine.optimization_suggestions = [
        {'type': 'test1'},
        {'type': 'test2'},
        {'type': 'test3'},
        {'type': 'test4'},
        {'type': 'test5'}
    ]
    
    history = engine.get_suggestions_history(limit=3)
    
    assert len(history) == 3
    assert history == engine.optimization_suggestions[-3:]


def test_get_suggestions_history_zero_limit(engine):
    """测试获取建议历史（限制为0）"""
    engine.optimization_suggestions = [{'type': 'test'}]
    
    history = engine.get_suggestions_history(limit=0)
    
    assert history == []


def test_clear_suggestions_history(engine):
    """测试清空建议历史"""
    engine.optimization_suggestions = [
        {'type': 'test1'},
        {'type': 'test2'}
    ]
    
    engine.clear_suggestions_history()
    
    assert len(engine.optimization_suggestions) == 0


def test_get_latest_suggestions(engine):
    """测试获取最新建议"""
    engine.optimization_suggestions = [
        {'type': 'test1'},
        {'type': 'test2'},
        {'type': 'test3'},
        {'type': 'test4'},
        {'type': 'test5'}
    ]
    
    latest = engine.get_latest_suggestions(count=3)
    
    assert len(latest) == 3
    assert latest == engine.optimization_suggestions[-3:]


def test_get_latest_suggestions_zero_count(engine):
    """测试获取最新建议（数量为0）"""
    engine.optimization_suggestions = [{'type': 'test'}]
    
    latest = engine.get_latest_suggestions(count=0)
    
    assert latest == []


def test_get_latest_suggestions_empty(engine):
    """测试获取最新建议（空历史）"""
    latest = engine.get_latest_suggestions(count=3)
    
    assert latest == []


def test_generate_suggestions_saves_to_history(engine):
    """测试生成建议保存到历史"""
    coverage_data = {'coverage_percent': 70.0}
    performance_data = {}
    
    initial_count = len(engine.optimization_suggestions)
    suggestions = engine.generate_suggestions(coverage_data, performance_data)
    
    assert len(engine.optimization_suggestions) == initial_count + len(suggestions)


def test_generate_suggestions_prints_output(engine, capsys):
    """测试生成建议打印输出"""
    coverage_data = {'coverage_percent': 70.0}
    performance_data = {}
    
    engine.generate_suggestions(coverage_data, performance_data)
    
    captured = capsys.readouterr()
    assert '生成优化建议' in captured.out or 'HIGH' in captured.out

