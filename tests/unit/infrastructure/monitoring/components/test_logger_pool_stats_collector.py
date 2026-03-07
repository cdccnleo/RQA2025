#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Logger池统计收集器组件
"""

import importlib
import sys
import time
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest


@pytest.fixture
def stats_collector_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.logger_pool_stats_collector"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def collector_without_pool(stats_collector_module, monkeypatch):
    """创建没有Logger池的收集器实例"""
    # 模拟POOL_AVAILABLE为False
    monkeypatch.setattr(stats_collector_module, "POOL_AVAILABLE", False)
    collector = stats_collector_module.LoggerPoolStatsCollector(pool_name="test_pool")
    return collector, stats_collector_module


@pytest.fixture
def collector_with_pool(stats_collector_module, monkeypatch):
    """创建有Logger池的收集器实例"""
    mock_pool = Mock()
    mock_pool.get_stats.return_value = {
        'pool_size': 5,
        'max_size': 10,
        'created_count': 20,
        'hit_count': 15,
        'hit_rate': 0.75,
        'loggers': [Mock(), Mock(), Mock()],
        'usage_stats': {
            'logger1': {'access_count': 10},
            'logger2': {'access_count': 5}
        }
    }
    
    def mock_get_logger_pool():
        return mock_pool
    
    monkeypatch.setattr(stats_collector_module, "POOL_AVAILABLE", True)
    monkeypatch.setattr(stats_collector_module, "get_logger_pool", mock_get_logger_pool)
    
    collector = stats_collector_module.LoggerPoolStatsCollector(pool_name="test_pool")
    return collector, stats_collector_module, mock_pool


def test_initialization_without_pool(collector_without_pool):
    """测试在没有Logger池时初始化"""
    collector, module = collector_without_pool
    assert collector.pool_name == "test_pool"
    assert collector.logger_pool is None
    assert collector.access_times == []
    assert collector.history_stats == []
    assert collector.max_access_times_size == 1000
    assert collector.max_history_size == 1000


def test_initialization_with_pool(collector_with_pool):
    """测试在有Logger池时初始化"""
    collector, module, mock_pool = collector_with_pool
    assert collector.pool_name == "test_pool"
    assert collector.logger_pool is not None
    assert collector.logger_pool == mock_pool


def test_collect_current_stats_without_pool(collector_without_pool):
    """测试在没有Logger池时收集统计信息"""
    collector, module = collector_without_pool
    stats = collector.collect_current_stats()
    
    assert stats is not None
    assert stats.pool_size == 10  # mock值
    assert stats.max_size == 100
    assert stats.hit_rate == 0.8
    # 验证stats被添加到历史记录
    assert len(collector.history_stats) == 1
    assert collector.history_stats[0] == stats


def test_collect_current_stats_with_pool(collector_with_pool):
    """测试在有Logger池时收集统计信息"""
    collector, module, mock_pool = collector_with_pool
    
    # 添加一些访问时间
    collector.record_access_time(0.001)
    collector.record_access_time(0.002)
    
    stats = collector.collect_current_stats()
    
    assert stats is not None
    assert stats.pool_size == 5
    assert stats.max_size == 10
    assert stats.created_count == 20
    assert stats.hit_count == 15
    assert stats.hit_rate == 0.75
    assert stats.logger_count == 3
    assert stats.total_access_count == 15  # 10 + 5
    assert stats.avg_access_time == pytest.approx(0.0015)  # (0.001 + 0.002) / 2
    assert stats.timestamp > 0
    assert len(collector.history_stats) == 1


def test_collect_current_stats_exception(collector_with_pool, monkeypatch):
    """测试收集统计信息时发生异常"""
    collector, module, mock_pool = collector_with_pool
    mock_pool.get_stats.side_effect = RuntimeError("Pool error")
    
    stats = collector.collect_current_stats()
    assert stats is None


def test_record_access_time(collector_without_pool):
    """测试记录访问时间"""
    collector, module = collector_without_pool
    
    collector.record_access_time(0.001)
    collector.record_access_time(0.002)
    collector.record_access_time(0.003)
    
    assert len(collector.access_times) == 3
    assert collector.access_times == [0.001, 0.002, 0.003]


def test_record_access_time_max_size_limit(collector_without_pool):
    """测试访问时间列表的最大大小限制"""
    collector, module = collector_without_pool
    collector.max_access_times_size = 3
    
    # 添加4个访问时间，应该只保留最后3个
    collector.record_access_time(0.001)
    collector.record_access_time(0.002)
    collector.record_access_time(0.003)
    collector.record_access_time(0.004)
    
    assert len(collector.access_times) == 3
    assert collector.access_times == [0.002, 0.003, 0.004]


def test_calculate_avg_access_time_empty(collector_without_pool):
    """测试计算平均访问时间（空列表）"""
    collector, module = collector_without_pool
    avg_time = collector._calculate_avg_access_time()
    assert avg_time == 0.0


def test_calculate_avg_access_time_with_data(collector_without_pool):
    """测试计算平均访问时间（有数据）"""
    collector, module = collector_without_pool
    collector.record_access_time(0.001)
    collector.record_access_time(0.002)
    collector.record_access_time(0.003)
    
    avg_time = collector._calculate_avg_access_time()
    assert avg_time == pytest.approx(0.002)  # (0.001 + 0.002 + 0.003) / 3


def test_estimate_memory_usage(collector_without_pool):
    """测试估算内存使用量"""
    collector, module = collector_without_pool
    
    # 添加一些历史记录
    mock_stats = Mock()
    collector.history_stats = [mock_stats] * 10
    
    pool_stats = {'pool_size': 5}
    memory = collector._estimate_memory_usage(pool_stats)
    
    # 5 * 2.0 + 10 * 0.1 = 10 + 1 = 11 MB
    assert memory == pytest.approx(11.0)


def test_estimate_memory_usage_exception(collector_without_pool, monkeypatch):
    """测试估算内存使用量时发生异常"""
    collector, module = collector_without_pool
    
    # 模拟pool_stats.get抛出异常
    def failing_get(*args, **kwargs):
        raise RuntimeError("Get error")
    
    pool_stats = Mock()
    pool_stats.get = failing_get
    
    memory = collector._estimate_memory_usage(pool_stats)
    assert memory == 0.0


def test_add_to_history(collector_without_pool):
    """测试添加统计到历史记录"""
    collector, module = collector_without_pool
    
    stats1 = collector.collect_current_stats()
    assert stats1 is not None
    
    stats2 = collector.collect_current_stats()
    assert stats2 is not None
    
    assert len(collector.history_stats) == 2
    # 验证stats对象被正确添加
    assert collector.history_stats[0].timestamp == stats1.timestamp
    assert collector.history_stats[1].timestamp == stats2.timestamp


def test_add_to_history_max_size_limit(collector_without_pool):
    """测试历史记录的最大大小限制"""
    collector, module = collector_without_pool
    collector.max_history_size = 3
    
    # 添加4条记录，应该只保留最后3条
    collector.collect_current_stats()
    collector.collect_current_stats()
    collector.collect_current_stats()
    stats4 = collector.collect_current_stats()
    
    assert len(collector.history_stats) == 3
    # 验证最后一条记录是stats4
    assert collector.history_stats[-1].timestamp == stats4.timestamp


def test_create_mock_stats(collector_without_pool):
    """测试创建模拟统计数据"""
    collector, module = collector_without_pool
    stats = collector._create_mock_stats()
    
    assert stats.pool_size == 10
    assert stats.max_size == 100
    assert stats.created_count == 50
    assert stats.hit_count == 40
    assert stats.hit_rate == 0.8
    assert stats.logger_count == 10
    assert stats.total_access_count == 200
    assert stats.avg_access_time == 0.001
    assert stats.memory_usage_mb == 20.0
    assert stats.timestamp > 0


def test_get_history_stats_no_limit(collector_without_pool):
    """测试获取历史统计数据（无限制）"""
    collector, module = collector_without_pool
    
    stats1 = collector.collect_current_stats()
    assert stats1 is not None
    
    stats2 = collector.collect_current_stats()
    assert stats2 is not None
    
    history = collector.get_history_stats()
    assert len(history) == 2
    # 验证返回的是副本
    assert history is not collector.history_stats
    # 验证内容正确
    assert history[0].timestamp == stats1.timestamp
    assert history[1].timestamp == stats2.timestamp


def test_get_history_stats_with_limit(collector_without_pool):
    """测试获取历史统计数据（有限制）"""
    collector, module = collector_without_pool
    
    collector.collect_current_stats()
    stats2 = collector.collect_current_stats()
    stats3 = collector.collect_current_stats()
    
    assert len(collector.history_stats) == 3
    
    history = collector.get_history_stats(limit=2)
    assert len(history) == 2
    # 应该返回最后2条记录
    assert history[0].timestamp == stats2.timestamp
    assert history[1].timestamp == stats3.timestamp


def test_get_history_stats_zero_limit(collector_without_pool):
    """测试获取历史统计数据（限制为0）"""
    collector, module = collector_without_pool
    
    collector.collect_current_stats()
    collector.collect_current_stats()
    
    history = collector.get_history_stats(limit=0)
    assert history == []


def test_get_current_access_times(collector_without_pool):
    """测试获取当前访问时间列表"""
    collector, module = collector_without_pool
    
    collector.record_access_time(0.001)
    collector.record_access_time(0.002)
    
    access_times = collector.get_current_access_times()
    assert access_times == [0.001, 0.002]
    # 验证返回的是副本
    assert access_times is not collector.access_times


def test_clear_history(collector_without_pool):
    """测试清空历史记录"""
    collector, module = collector_without_pool
    
    stats = collector.collect_current_stats()
    assert stats is not None
    collector.record_access_time(0.001)
    
    assert len(collector.history_stats) > 0
    assert len(collector.access_times) > 0
    
    collector.clear_history()
    
    assert len(collector.history_stats) == 0
    assert len(collector.access_times) == 0


def test_collect_current_stats_with_empty_pool_stats(collector_with_pool):
    """测试收集统计信息时pool_stats为空"""
    collector, module, mock_pool = collector_with_pool
    mock_pool.get_stats.return_value = {}
    
    stats = collector.collect_current_stats()
    
    assert stats is not None
    assert stats.pool_size == 0
    assert stats.max_size == 0
    assert stats.logger_count == 0
    assert stats.total_access_count == 0


def test_collect_current_stats_with_missing_usage_stats(collector_with_pool):
    """测试收集统计信息时usage_stats缺失"""
    collector, module, mock_pool = collector_with_pool
    mock_pool.get_stats.return_value = {
        'pool_size': 5,
        'max_size': 10,
        'loggers': []
    }
    
    stats = collector.collect_current_stats()
    
    assert stats is not None
    assert stats.total_access_count == 0


def test_fallback_logger_pool_stats_definition(stats_collector_module, monkeypatch):
    """测试LoggerPoolStats fallback定义"""
    import builtins
    original_import = builtins.__import__
    
    # 模拟导入失败
    def failing_import(name, globals=None, locals=None, fromlist=(), level=0):
        if "logger_pool_monitor" in name:
            raise ImportError("Module not found")
        return original_import(name, globals, locals, fromlist, level)
    
    monkeypatch.setattr(builtins, "__import__", failing_import)
    
    # 重新导入模块
    module_name = "src.infrastructure.monitoring.components.logger_pool_stats_collector"
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    module = importlib.import_module(module_name)
    
    # 验证fallback LoggerPoolStats存在
    assert hasattr(module, "LoggerPoolStats")
    LoggerPoolStats = module.LoggerPoolStats
    
    # 验证可以创建实例
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=3,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    assert stats.pool_size == 5
    assert stats.hit_rate == 0.75

