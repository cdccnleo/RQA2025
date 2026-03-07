#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
补充测试性能监控组件 - 覆盖未覆盖的代码路径
"""

import importlib
import sys
import time
import threading
from unittest.mock import Mock, patch, MagicMock

import pytest

from src.infrastructure.monitoring.components.performance_monitor import (
    PerformanceMonitor,
    PerformanceMetrics,
    ComponentPerformanceStats,
    PerformanceContext,
    monitor_performance,
    global_performance_monitor
)


@pytest.fixture
def performance_monitor_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.performance_monitor"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def monitor(performance_monitor_module):
    """创建PerformanceMonitor实例（禁用自动监控）"""
    module = performance_monitor_module
    return module.PerformanceMonitor(enable_auto_monitoring=False)


@pytest.fixture
def PerformanceMetrics(performance_monitor_module):
    """获取PerformanceMetrics类"""
    return performance_monitor_module.PerformanceMetrics


@pytest.fixture
def ComponentPerformanceStats(performance_monitor_module):
    """获取ComponentPerformanceStats类"""
    return performance_monitor_module.ComponentPerformanceStats


@pytest.fixture
def PerformanceContext(performance_monitor_module):
    """获取PerformanceContext类"""
    return performance_monitor_module.PerformanceContext


def test_performance_metrics_complete_zero_duration(monitor, PerformanceMetrics):
    """测试PerformanceMetrics.complete()处理duration_ms <= 0的情况"""
    metrics = PerformanceMetrics(
        component_name="TestComp",
        operation_name="test_op",
        start_time=time.time()
    )
    
    # 模拟duration_ms <= 0的情况
    # 需要mock start_tick和perf_counter
    original_start_tick = metrics.start_tick
    with patch('time.perf_counter', return_value=original_start_tick):  # 返回相同值，duration为0
        metrics.complete(success=True)
        
        assert metrics.duration_ms == 0.01  # 应该被设置为0.01


def test_performance_metrics_complete_with_error_message(monitor, PerformanceMetrics):
    """测试PerformanceMetrics.complete()设置error_message"""
    metrics = PerformanceMetrics(
        component_name="TestComp",
        operation_name="test_op",
        start_time=time.time()
    )
    
    metrics.complete(success=False, error_message="Test error")
    
    assert metrics.success is False
    assert metrics.error_message == "Test error"


def test_get_recent_metrics_limit_zero(monitor, PerformanceMetrics):
    """测试get_recent_metrics() limit=0的情况"""
    # 添加一些指标
    for i in range(5):
        metrics = PerformanceMetrics(
            component_name="TestComp",
            operation_name=f"op_{i}",
            start_time=time.time()
        )
        metrics.complete(success=True)
        monitor.record_metrics(metrics)
    
    # limit=0应该返回所有指标
    recent = monitor.get_recent_metrics(limit=0)
    assert len(recent) == 5


def test_get_recent_metrics_with_component_filter(monitor, PerformanceMetrics):
    """测试get_recent_metrics()带组件名称过滤"""
    # 添加不同组件的指标
    for comp in ["CompA", "CompB", "CompA"]:
        metrics = PerformanceMetrics(
            component_name=comp,
            operation_name="test_op",
            start_time=time.time()
        )
        metrics.complete(success=True)
        monitor.record_metrics(metrics)
    
    # 只获取CompA的指标
    recent = monitor.get_recent_metrics(component_name="CompA", limit=10)
    assert len(recent) == 2
    assert all(m.component_name == "CompA" for m in recent)


def test_get_performance_summary_empty_stats(monitor):
    """测试get_performance_summary()空统计的情况"""
    summary = monitor.get_performance_summary()
    
    assert summary['total_components'] == 0
    assert summary['total_operations'] == 0
    assert summary['components'] == {}
    assert 'system_health' in summary
    assert 'generated_at' in summary


def test_get_performance_summary_with_zero_operations(monitor, ComponentPerformanceStats, PerformanceMetrics):
    """测试get_performance_summary()组件操作数为0但success_rate为0的情况"""
    # 创建一个stats，有操作但全部失败
    # 注意：_calculate_system_health需要至少有一个avg_response_time_ms > 0的stats
    stats = ComponentPerformanceStats(component_name="EmptyComp")
    
    # 创建一个失败的metrics并更新stats
    failed_metrics = PerformanceMetrics(
        component_name="EmptyComp",
        operation_name="test_op",
        start_time=time.time()
    )
    failed_metrics.complete(success=False, error_message="Test error")
    failed_metrics.duration_ms = 100.0
    
    # 通过update方法更新stats，这样error_rate会被正确计算
    stats.update(failed_metrics)
    
    monitor.component_stats["EmptyComp"] = stats
    
    summary = monitor.get_performance_summary()
    
    assert summary['total_components'] == 1
    assert summary['total_operations'] == 1
    assert summary['components']['EmptyComp']['success_rate'] == 0
    assert summary['components']['EmptyComp']['error_rate'] == 1.0  # 1个操作全部失败


def test_detect_performance_anomalies_high_error_rate_high_severity(monitor, PerformanceMetrics):
    """测试detect_performance_anomalies()高错误率（高严重性）"""
    # 创建高错误率组件（>50%）
    for i in range(20):
        metrics = PerformanceMetrics(
            component_name="HighErrorComp",
            operation_name=f"op_{i}",
            start_time=time.time()
        )
        metrics.complete(success=i < 5)  # 75%错误率
        monitor.record_metrics(metrics)
    
    anomalies = monitor.detect_performance_anomalies()
    
    high_error_anomalies = [a for a in anomalies if a['type'] == 'high_error_rate' and a['component'] == 'HighErrorComp']
    assert len(high_error_anomalies) == 1
    assert high_error_anomalies[0]['severity'] == 'high'


def test_detect_performance_anomalies_high_error_rate_medium_severity(monitor, PerformanceMetrics):
    """测试detect_performance_anomalies()高错误率（中等严重性）"""
    # 创建中等错误率组件（10-50%）
    for i in range(20):
        metrics = PerformanceMetrics(
            component_name="MediumErrorComp",
            operation_name=f"op_{i}",
            start_time=time.time()
        )
        metrics.complete(success=i < 15)  # 25%错误率
        monitor.record_metrics(metrics)
    
    anomalies = monitor.detect_performance_anomalies()
    
    medium_error_anomalies = [a for a in anomalies if a['type'] == 'high_error_rate' and a['component'] == 'MediumErrorComp']
    assert len(medium_error_anomalies) == 1
    assert medium_error_anomalies[0]['severity'] == 'medium'


def test_detect_performance_anomalies_slow_response_high_severity(monitor, PerformanceMetrics):
    """测试detect_performance_anomalies()慢响应（高严重性）"""
    # 创建非常慢的组件（>5000ms）
    metrics = PerformanceMetrics(
        component_name="VerySlowComp",
        operation_name="slow_op",
        start_time=time.time(),
        duration_ms=6000.0
    )
    metrics.complete(success=True)
    monitor.record_metrics(metrics)
    
    anomalies = monitor.detect_performance_anomalies()
    
    slow_anomalies = [a for a in anomalies if a['type'] == 'slow_response' and a['component'] == 'VerySlowComp']
    assert len(slow_anomalies) == 1
    assert slow_anomalies[0]['severity'] == 'high'


def test_detect_performance_anomalies_slow_response_medium_severity(monitor, PerformanceMetrics):
    """测试detect_performance_anomalies()慢响应（中等严重性）"""
    # 创建中等慢的组件（1000-5000ms）
    metrics = PerformanceMetrics(
        component_name="MediumSlowComp",
        operation_name="slow_op",
        start_time=time.time(),
        duration_ms=2000.0
    )
    metrics.complete(success=True)
    monitor.record_metrics(metrics)
    
    anomalies = monitor.detect_performance_anomalies()
    
    slow_anomalies = [a for a in anomalies if a['type'] == 'slow_response' and a['component'] == 'MediumSlowComp']
    assert len(slow_anomalies) == 1
    assert slow_anomalies[0]['severity'] == 'medium'


def test_detect_performance_anomalies_high_memory_usage(monitor, PerformanceMetrics):
    """测试detect_performance_anomalies()高内存使用"""
    # 创建高内存使用的组件
    # 需要多个指标来更新stats.memory_usage_mb
    for i in range(5):
        metrics = PerformanceMetrics(
            component_name="MemoryHogComp",
            operation_name=f"memory_op_{i}",
            start_time=time.time()
        )
        # complete()会设置memory_usage_mb，但我们需要确保stats中也有这个值
        metrics.complete(success=True)
        # 直接设置一个高内存值
        metrics.memory_usage_mb = 600.0  # 超过500MB阈值
        monitor.record_metrics(metrics)
    
    # 手动更新stats的memory_usage_mb
    if "MemoryHogComp" in monitor.component_stats:
        monitor.component_stats["MemoryHogComp"].memory_usage_mb = 600.0
    
    anomalies = monitor.detect_performance_anomalies()
    
    memory_anomalies = [a for a in anomalies if a['type'] == 'high_memory_usage' and a['component'] == 'MemoryHogComp']
    assert len(memory_anomalies) == 1
    assert memory_anomalies[0]['severity'] == 'medium'


def test_generate_performance_recommendations_error_handling(monitor, PerformanceMetrics):
    """测试generate_performance_recommendations()错误处理建议"""
    # 创建高错误率组件
    for i in range(20):
        metrics = PerformanceMetrics(
            component_name="ErrorProneComp",
            operation_name=f"op_{i}",
            start_time=time.time()
        )
        metrics.complete(success=i < 5)  # 75%错误率
        monitor.record_metrics(metrics)
    
    recommendations = monitor.generate_performance_recommendations()
    
    error_recommendations = [r for r in recommendations if r['type'] == 'error_handling' and r['component'] == 'ErrorProneComp']
    assert len(error_recommendations) > 0


def test_generate_performance_recommendations_performance_optimization(monitor, PerformanceMetrics):
    """测试generate_performance_recommendations()性能优化建议"""
    # 创建慢响应组件
    metrics = PerformanceMetrics(
        component_name="SlowComp",
        operation_name="slow_op",
        start_time=time.time(),
        duration_ms=2000.0
    )
    metrics.complete(success=True)
    monitor.record_metrics(metrics)
    
    recommendations = monitor.generate_performance_recommendations()
    
    perf_recommendations = [r for r in recommendations if r['type'] == 'performance_optimization' and r['component'] == 'SlowComp']
    assert len(perf_recommendations) > 0


def test_generate_performance_recommendations_memory_optimization(monitor, PerformanceMetrics):
    """测试generate_performance_recommendations()内存优化建议"""
    # 创建高内存使用组件
    metrics = PerformanceMetrics(
        component_name="MemoryComp",
        operation_name="memory_op",
        start_time=time.time()
    )
    metrics.memory_usage_mb = 600.0
    metrics.complete(success=True)
    monitor.record_metrics(metrics)
    
    recommendations = monitor.generate_performance_recommendations()
    
    memory_recommendations = [r for r in recommendations if r['type'] == 'memory_optimization' and r['component'] == 'MemoryComp']
    assert len(memory_recommendations) > 0


def test_calculate_system_health_empty(monitor):
    """测试_calculate_system_health()空统计的情况"""
    health = monitor._calculate_system_health()
    
    assert health['status'] == 'unknown'
    assert health['score'] == 0.0


def test_calculate_system_health_excellent(monitor, PerformanceMetrics):
    """测试_calculate_system_health()优秀状态"""
    # 创建高性能组件
    for i in range(10):
        metrics = PerformanceMetrics(
            component_name="ExcellentComp",
            operation_name=f"op_{i}",
            start_time=time.time(),
            duration_ms=50.0  # 快速响应
        )
        metrics.complete(success=True)
        monitor.record_metrics(metrics)
    
    health = monitor._calculate_system_health()
    
    assert health['status'] == 'excellent' or health['status'] == 'good'
    assert health['score'] >= 70


def test_calculate_system_health_poor(monitor, PerformanceMetrics):
    """测试_calculate_system_health()差状态"""
    # 创建低性能组件
    for i in range(10):
        metrics = PerformanceMetrics(
            component_name="PoorComp",
            operation_name=f"op_{i}",
            start_time=time.time(),
            duration_ms=2000.0  # 慢响应
        )
        metrics.complete(success=i < 3)  # 70%错误率
        monitor.record_metrics(metrics)
    
    health = monitor._calculate_system_health()
    
    assert health['status'] in ['poor', 'fair']
    assert health['score'] < 70


def test_start_auto_monitoring(monitor):
    """测试start_auto_monitoring()"""
    monitor.start_auto_monitoring()
    
    assert monitor.monitoring_active is True
    assert monitor.auto_monitor_thread is not None
    assert monitor.auto_monitor_thread.is_alive()
    
    monitor.stop_auto_monitoring()


def test_stop_auto_monitoring(monitor):
    """测试stop_auto_monitoring()"""
    monitor.start_auto_monitoring()
    assert monitor.monitoring_active is True
    
    monitor.stop_auto_monitoring()
    
    assert monitor.monitoring_active is False
    if monitor.auto_monitor_thread:
        # 等待线程结束
        monitor.auto_monitor_thread.join(timeout=1.0)


def test_performance_context_enter_exit(monitor, PerformanceContext):
    """测试PerformanceContext上下文管理器"""
    context = monitor.monitor_operation("TestComp", "test_op")
    
    assert isinstance(context, PerformanceContext)
    
    with context:
        assert context.metrics is not None
        assert context.metrics.component_name == "TestComp"
        assert context.metrics.operation_name == "test_op"
    
    # 检查是否记录了指标
    assert len(monitor.metrics_history) == 1
    assert monitor.metrics_history[0].success is True


def test_performance_context_exception(monitor, PerformanceContext):
    """测试PerformanceContext异常处理"""
    context = monitor.monitor_operation("TestComp", "failing_op")
    
    try:
        with context:
            raise ValueError("Test exception")
    except ValueError:
        pass
    
    # 检查是否记录了失败的指标
    assert len(monitor.metrics_history) == 1
    assert monitor.metrics_history[0].success is False
    assert monitor.metrics_history[0].error_message == "Test exception"


def test_record_metrics_max_history_size(monitor, PerformanceMetrics):
    """测试record_metrics()达到最大历史大小限制"""
    monitor.max_history_size = 5
    
    # 添加超过限制的指标
    for i in range(10):
        metrics = PerformanceMetrics(
            component_name="TestComp",
            operation_name=f"op_{i}",
            start_time=time.time()
        )
        metrics.complete(success=True)
        monitor.record_metrics(metrics)
    
    # 应该只保留最新的5个
    assert len(monitor.metrics_history) == 5
    assert monitor.metrics_history[0].operation_name == "op_5"
    assert monitor.metrics_history[-1].operation_name == "op_9"


def test_get_all_component_stats_copy(monitor, PerformanceMetrics):
    """测试get_all_component_stats()返回副本"""
    # 添加一些指标
    metrics = PerformanceMetrics(
        component_name="TestComp",
        operation_name="test_op",
        start_time=time.time()
    )
    metrics.complete(success=True)
    monitor.record_metrics(metrics)
    
    stats1 = monitor.get_all_component_stats()
    stats2 = monitor.get_all_component_stats()
    
    # 应该是不同的对象（副本）
    assert stats1 is not stats2
    # 但内容应该相同
    assert stats1 == stats2

