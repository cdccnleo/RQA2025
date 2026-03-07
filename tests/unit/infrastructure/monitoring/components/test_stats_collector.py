#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试统计收集器组件
"""

import importlib
import sys
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def stats_collector_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.stats_collector"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def mock_config():
    """创建模拟配置对象"""
    class MockLoggerPoolStatsConfig:
        pass
    
    return MockLoggerPoolStatsConfig()


@pytest.fixture
def collector(stats_collector_module, mock_config):
    """创建StatsCollector实例"""
    # Patch monitor_performance装饰器以避免依赖
    with patch.object(stats_collector_module, 'monitor_performance', lambda component, operation: lambda func: func):
        collector = stats_collector_module.StatsCollector(
            pool_name="test_pool",
            config=mock_config
        )
        return collector, stats_collector_module


def test_initialization(collector):
    """测试初始化"""
    collector_instance, module = collector
    assert collector_instance.pool_name == "test_pool"
    assert collector_instance.current_stats is None
    assert collector_instance.history_stats == []
    assert collector_instance.max_history_size == 1000
    assert collector_instance.access_times == []
    assert collector_instance.max_access_times_size == 1000


def test_initialization_default_pool_name(stats_collector_module, mock_config):
    """测试初始化（默认池名称）"""
    with patch.object(stats_collector_module, 'monitor_performance', lambda component, operation: lambda func: func):
        collector = stats_collector_module.StatsCollector(config=mock_config)
        assert collector.pool_name == "default_pool"


def test_collect_stats_success(collector):
    """测试收集统计信息（成功）"""
    collector_instance, module = collector
    
    stats = collector_instance.collect_stats()
    
    assert stats is not None
    assert stats['pool_name'] == "test_pool"
    assert 'pool_size' in stats
    assert 'hit_rate' in stats
    assert 'timestamp' in stats
    assert collector_instance.current_stats == stats
    assert len(collector_instance.history_stats) == 1


def test_collect_stats_exception(collector, monkeypatch):
    """测试收集统计信息（异常）"""
    collector_instance, module = collector
    
    # 模拟_collect_mock_stats抛出异常
    def failing_collect():
        raise RuntimeError("Collection error")
    
    monkeypatch.setattr(collector_instance, '_collect_mock_stats', failing_collect)
    
    stats = collector_instance.collect_stats()
    
    assert stats is None


def test_get_current_stats(collector):
    """测试获取当前统计信息"""
    collector_instance, module = collector
    
    # 初始状态应该为None
    assert collector_instance.get_current_stats() is None
    
    # 收集统计后应该有值
    stats = collector_instance.collect_stats()
    assert collector_instance.get_current_stats() == stats


def test_get_history_stats(collector):
    """测试获取历史统计信息"""
    collector_instance, module = collector
    
    # 收集多个统计
    for _ in range(5):
        collector_instance.collect_stats()
    
    # 获取所有历史
    history = collector_instance.get_history_stats(limit=0)
    assert len(history) == 5
    
    # 获取限制数量的历史
    history_limited = collector_instance.get_history_stats(limit=3)
    assert len(history_limited) == 3


def test_get_history_stats_empty(collector):
    """测试获取历史统计信息（空历史）"""
    collector_instance, module = collector
    
    history = collector_instance.get_history_stats()
    assert history == []


def test_get_access_times(collector):
    """测试获取访问时间记录"""
    collector_instance, module = collector
    
    # 记录一些访问时间
    for i in range(5):
        collector_instance.record_access_time(0.001 * (i + 1))
    
    # 获取所有访问时间
    access_times = collector_instance.get_access_times(limit=0)
    assert len(access_times) == 5
    
    # 获取限制数量的访问时间
    access_times_limited = collector_instance.get_access_times(limit=3)
    assert len(access_times_limited) == 3


def test_get_access_times_empty(collector):
    """测试获取访问时间记录（空记录）"""
    collector_instance, module = collector
    
    access_times = collector_instance.get_access_times()
    assert access_times == []


def test_record_access_time(collector):
    """测试记录访问时间"""
    collector_instance, module = collector
    
    collector_instance.record_access_time(0.005)
    
    assert len(collector_instance.access_times) == 1
    assert collector_instance.access_times[0] == 0.005


def test_record_access_time_max_size(collector):
    """测试记录访问时间（最大大小限制）"""
    collector_instance, module = collector
    
    # 设置较小的最大大小
    collector_instance.max_access_times_size = 3
    
    # 记录超过最大大小的访问时间
    for i in range(5):
        collector_instance.record_access_time(0.001 * (i + 1))
    
    # 应该只保留最后3个
    assert len(collector_instance.access_times) == 3
    assert collector_instance.access_times[0] == 0.003  # 第3个


def test_calculate_percentiles_empty_data(collector):
    """测试计算百分位数（空数据）"""
    collector_instance, module = collector
    
    result = collector_instance.calculate_percentiles([], [50, 95, 99])
    assert result == {}


def test_calculate_percentiles_success(collector):
    """测试计算百分位数（成功）"""
    collector_instance, module = collector
    
    # 创建测试数据：1到100的列表
    data = list(range(1, 101))
    percentiles = [50, 95, 99]
    
    result = collector_instance.calculate_percentiles(data, percentiles)
    
    assert 'p50' in result
    assert 'p95' in result
    assert 'p99' in result
    assert result['p50'] == 50  # 中位数
    assert result['p95'] == 95
    assert result['p99'] == 99


def test_calculate_percentiles_invalid_percentile(collector):
    """测试计算百分位数（无效百分位）"""
    collector_instance, module = collector
    
    data = [1, 2, 3, 4, 5]
    percentiles = [50, 150, -10]  # 150和-10是无效的
    
    result = collector_instance.calculate_percentiles(data, percentiles)
    
    assert 'p50' in result
    assert 'p150' in result
    assert result['p150'] == 0.0  # 无效百分位应该返回0.0


def test_calculate_percentiles_exception(collector, monkeypatch):
    """测试计算百分位数（异常）"""
    collector_instance, module = collector
    
    # 模拟sorted函数抛出异常
    import builtins
    original_sorted = builtins.sorted
    
    def failing_sorted(*args, **kwargs):
        raise RuntimeError("Sort error")
    
    monkeypatch.setattr(builtins, 'sorted', failing_sorted)
    
    try:
        result = collector_instance.calculate_percentiles([1, 2, 3], [50])
        assert result == {}
    finally:
        # 恢复原始sorted函数
        monkeypatch.setattr(builtins, 'sorted', original_sorted)


def test_analyze_trends_insufficient_data(collector):
    """测试分析趋势（数据不足）"""
    collector_instance, module = collector
    
    # 没有历史数据
    result = collector_instance.analyze_trends('pool_size')
    assert result['trend'] == 'insufficient_data'
    
    # 只有一条历史数据
    collector_instance.collect_stats()
    result = collector_instance.analyze_trends('pool_size')
    assert result['trend'] == 'insufficient_data'


def test_analyze_trends_increasing(collector):
    """测试分析趋势（上升趋势）"""
    collector_instance, module = collector
    
    # 创建上升趋势的数据
    for i in range(10):
        stats = {
            'pool_name': 'test_pool',
            'pool_size': 10 + i * 2,  # 从10增加到28
            'hit_rate': 0.8 + i * 0.01
        }
        collector_instance._add_to_history(stats)
    
    result = collector_instance.analyze_trends('pool_size', window_size=10)
    
    assert result['trend'] == 'increasing'
    assert result['slope'] > 0
    assert 'current_value' in result
    assert 'change_percent' in result


def test_analyze_trends_decreasing(collector):
    """测试分析趋势（下降趋势）"""
    collector_instance, module = collector
    
    # 创建下降趋势的数据
    for i in range(10):
        stats = {
            'pool_name': 'test_pool',
            'pool_size': 30 - i * 2,  # 从30减少到12
            'hit_rate': 0.9 - i * 0.01
        }
        collector_instance._add_to_history(stats)
    
    result = collector_instance.analyze_trends('pool_size', window_size=10)
    
    assert result['trend'] == 'decreasing'
    assert result['slope'] < 0


def test_analyze_trends_stable(collector):
    """测试分析趋势（稳定趋势）"""
    collector_instance, module = collector
    
    # 创建稳定趋势的数据
    for i in range(10):
        stats = {
            'pool_name': 'test_pool',
            'pool_size': 20,  # 保持稳定
            'hit_rate': 0.8
        }
        collector_instance._add_to_history(stats)
    
    result = collector_instance.analyze_trends('pool_size', window_size=10)
    
    assert result['trend'] == 'stable'
    assert abs(result['slope']) <= 0.01


def test_analyze_trends_metric_not_found(collector):
    """测试分析趋势（指标不存在）"""
    collector_instance, module = collector
    
    # 创建不包含目标指标的数据
    for i in range(10):
        stats = {
            'pool_name': 'test_pool',
            'other_metric': i
        }
        collector_instance._add_to_history(stats)
    
    result = collector_instance.analyze_trends('non_existent_metric', window_size=10)
    assert result['trend'] == 'insufficient_data'


def test_analyze_trends_exception(collector, monkeypatch):
    """测试分析趋势（异常）"""
    collector_instance, module = collector
    
    # 添加一些历史数据
    collector_instance.collect_stats()
    collector_instance.collect_stats()
    
    # 模拟_calculate_slope抛出异常
    def failing_slope(*args, **kwargs):
        raise RuntimeError("Slope error")
    
    monkeypatch.setattr(collector_instance, '_calculate_slope', failing_slope)
    
    result = collector_instance.analyze_trends('pool_size')
    assert result['trend'] == 'error'
    assert 'error' in result


def test_collect_mock_stats(collector):
    """测试收集模拟统计信息"""
    collector_instance, module = collector
    
    stats = collector_instance._collect_mock_stats()
    
    assert stats['pool_name'] == "test_pool"
    assert 'pool_size' in stats
    assert 'max_size' in stats
    assert 'created_count' in stats
    assert 'hit_count' in stats
    assert 'hit_rate' in stats
    assert 'memory_usage_mb' in stats
    assert 'avg_access_time' in stats
    assert 'timestamp' in stats
    assert 'collection_time' in stats


def test_add_to_history(collector):
    """测试添加到历史记录"""
    collector_instance, module = collector
    
    stats = {'test': 'data'}
    collector_instance._add_to_history(stats)
    
    assert len(collector_instance.history_stats) == 1
    assert collector_instance.history_stats[0] == stats


def test_add_to_history_max_size(collector):
    """测试添加到历史记录（最大大小限制）"""
    collector_instance, module = collector
    
    # 设置较小的最大大小
    collector_instance.max_history_size = 3
    
    # 添加超过最大大小的记录
    for i in range(5):
        collector_instance._add_to_history({'index': i})
    
    # 应该只保留最后3个
    assert len(collector_instance.history_stats) == 3
    assert collector_instance.history_stats[0]['index'] == 2  # 第3个


def test_calculate_slope_insufficient_data(collector):
    """测试计算斜率（数据不足）"""
    collector_instance, module = collector
    
    # 空数据
    slope = collector_instance._calculate_slope([])
    assert slope == 0.0
    
    # 只有一个数据点
    slope = collector_instance._calculate_slope([1.0])
    assert slope == 0.0


def test_calculate_slope_increasing(collector):
    """测试计算斜率（上升）"""
    collector_instance, module = collector
    
    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    slope = collector_instance._calculate_slope(values)
    
    assert slope > 0


def test_calculate_slope_decreasing(collector):
    """测试计算斜率（下降）"""
    collector_instance, module = collector
    
    values = [5.0, 4.0, 3.0, 2.0, 1.0]
    slope = collector_instance._calculate_slope(values)
    
    assert slope < 0


def test_calculate_slope_zero_denominator(collector):
    """测试计算斜率（分母为零）"""
    collector_instance, module = collector
    
    # 所有值相同的情况可能导致分母为零
    values = [1.0, 1.0, 1.0]
    slope = collector_instance._calculate_slope(values)
    
    # 应该返回0.0而不是抛出异常
    assert slope == 0.0


def test_analyze_trends_with_window_size(collector):
    """测试分析趋势（自定义窗口大小）"""
    collector_instance, module = collector
    
    # 创建20条历史数据
    for i in range(20):
        stats = {
            'pool_name': 'test_pool',
            'pool_size': 10 + i
        }
        collector_instance._add_to_history(stats)
    
    # 使用较小的窗口大小
    result = collector_instance.analyze_trends('pool_size', window_size=5)
    
    assert result['trend'] == 'increasing'
    # 应该只分析最后5条数据


def test_analyze_trends_change_percent_zero_initial(collector):
    """测试分析趋势（初始值为零）"""
    collector_instance, module = collector
    
    # 创建从0开始的数据
    for i in range(5):
        stats = {
            'pool_name': 'test_pool',
            'pool_size': i  # 从0开始
        }
        collector_instance._add_to_history(stats)
    
    result = collector_instance.analyze_trends('pool_size', window_size=5)
    
    # change_percent应该为0（因为初始值为0）
    assert result['change_percent'] == 0


def test_get_history_stats_negative_limit(collector):
    """测试获取历史统计信息（负限制）"""
    collector_instance, module = collector
    
    # 收集一些统计
    for _ in range(5):
        collector_instance.collect_stats()
    
    # 负限制应该返回所有历史
    history = collector_instance.get_history_stats(limit=-1)
    assert len(history) == 5


def test_get_access_times_negative_limit(collector):
    """测试获取访问时间记录（负限制）"""
    collector_instance, module = collector
    
    # 记录一些访问时间
    for i in range(5):
        collector_instance.record_access_time(0.001 * (i + 1))
    
    # 负限制应该返回所有记录
    access_times = collector_instance.get_access_times(limit=-1)
    assert len(access_times) == 5

