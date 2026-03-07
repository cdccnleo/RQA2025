#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试Logger池指标导出器组件
"""

import importlib
import sys
import time
from unittest.mock import Mock, patch

import pytest


@pytest.fixture
def metrics_exporter_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.logger_pool_metrics_exporter"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def exporter(metrics_exporter_module):
    """创建LoggerPoolMetricsExporter实例"""
    return metrics_exporter_module.LoggerPoolMetricsExporter(pool_name="test_pool")


@pytest.fixture
def sample_stats(metrics_exporter_module):
    """创建示例统计数据"""
    LoggerPoolStats = metrics_exporter_module.LoggerPoolStats
    return LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )


def test_initialization(exporter):
    """测试初始化"""
    assert exporter.pool_name == "test_pool"


def test_export_prometheus_metrics_success(exporter, sample_stats):
    """测试成功导出Prometheus格式指标"""
    result = exporter.export_prometheus_metrics(sample_stats)
    
    assert isinstance(result, str)
    assert len(result) > 0
    assert 'logger_pool_size' in result
    assert 'logger_pool_max_size' in result
    assert 'logger_pool_logger_count' in result
    assert 'logger_pool_created_count' in result
    assert 'logger_pool_hit_count' in result
    assert 'logger_pool_hit_rate' in result
    assert 'logger_pool_total_access_count' in result
    assert 'logger_pool_avg_access_time_seconds' in result
    assert 'logger_pool_memory_usage_mb' in result
    assert 'pool="test_pool"' in result


def test_export_prometheus_metrics_empty_stats(exporter):
    """测试导出空统计信息"""
    result = exporter.export_prometheus_metrics(None)
    assert result == ""


def test_export_prometheus_metrics_pool_name_with_dash(exporter, sample_stats):
    """测试池名称包含连字符的情况"""
    exporter.pool_name = "test-pool-name"
    result = exporter.export_prometheus_metrics(sample_stats)
    
    # 连字符应该被替换为下划线
    assert 'pool="test_pool_name"' in result


def test_generate_core_metrics(exporter, sample_stats):
    """测试生成核心指标"""
    metrics = exporter._generate_core_metrics(sample_stats, "test_pool")
    
    # 每个指标包含3行：HELP、TYPE、metric
    assert len(metrics) == 9  # 3个指标 * 3行
    assert '# HELP logger_pool_size' in metrics[0]
    assert '# TYPE logger_pool_size gauge' in metrics[1]
    assert 'logger_pool_size{pool="test_pool"} 5' in metrics[2]


def test_generate_performance_metrics(exporter, sample_stats):
    """测试生成性能指标"""
    metrics = exporter._generate_performance_metrics(sample_stats, "test_pool")
    
    # 每个指标包含3行：HELP、TYPE、metric
    assert len(metrics) == 15  # 5个指标 * 3行
    assert 'logger_pool_created_count' in metrics[0]
    assert 'logger_pool_hit_count' in metrics[3]
    assert 'logger_pool_hit_rate' in metrics[6]
    assert 'logger_pool_total_access_count' in metrics[9]
    assert 'logger_pool_avg_access_time_seconds' in metrics[12]


def test_generate_memory_metrics(exporter, sample_stats):
    """测试生成内存指标"""
    metrics = exporter._generate_memory_metrics(sample_stats, "test_pool")
    
    # 每个指标包含3行：HELP、TYPE、metric
    assert len(metrics) == 3  # 1个指标 * 3行
    assert 'logger_pool_memory_usage_mb' in metrics[0]
    assert '10.0' in metrics[2]


def test_export_json_metrics_success(exporter, sample_stats):
    """测试成功导出JSON格式指标"""
    result = exporter.export_json_metrics(sample_stats)
    
    assert isinstance(result, dict)
    assert result['pool_name'] == "test_pool"
    assert result['timestamp'] == sample_stats.timestamp
    assert 'metrics' in result
    assert 'derived_metrics' in result
    assert result['metrics']['pool_size'] == 5
    assert result['metrics']['hit_rate'] == 0.75
    assert result['derived_metrics']['pool_utilization'] == pytest.approx(0.5)  # 5/10
    assert result['derived_metrics']['miss_rate'] == pytest.approx(0.25)  # 1 - 0.75


def test_export_json_metrics_empty_stats(exporter):
    """测试导出空统计信息"""
    result = exporter.export_json_metrics(None)
    assert result == {}


def test_export_json_metrics_zero_max_size(exporter, metrics_exporter_module):
    """测试max_size为0时的导出"""
    LoggerPoolStats = metrics_exporter_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=0,  # 零值
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    result = exporter.export_json_metrics(stats)
    assert result['derived_metrics']['pool_utilization'] == 0
    assert result['derived_metrics']['memory_efficiency_score'] == 0


def test_export_summary_report_success(exporter, sample_stats):
    """测试成功导出汇总报告"""
    alert_status = {
        'hit_rate_low': False,
        'pool_usage_high': False,
        'memory_high': False
    }
    
    result = exporter.export_summary_report(sample_stats, alert_status)
    
    assert isinstance(result, dict)
    assert result['pool_name'] == "test_pool"
    assert 'current_stats' in result
    assert 'performance_metrics' in result
    assert 'alert_status' in result
    assert 'recommendations' in result
    assert result['current_stats']['pool_size'] == 5
    assert result['current_stats']['avg_access_time_ms'] == pytest.approx(1.0)  # 0.001 * 1000


def test_export_summary_report_empty_stats(exporter):
    """测试导出空统计信息"""
    result = exporter.export_summary_report(None)
    assert result == {}


def test_export_summary_report_no_alert_status(exporter, sample_stats):
    """测试没有告警状态时的导出"""
    result = exporter.export_summary_report(sample_stats)
    
    assert result['alert_status'] == {}
    assert isinstance(result['recommendations'], list)


def test_export_summary_report_zero_max_size(exporter, metrics_exporter_module):
    """测试max_size为0时的汇总报告"""
    LoggerPoolStats = metrics_exporter_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=0,
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    result = exporter.export_summary_report(stats)
    assert result['performance_metrics']['pool_utilization'] == 0


def test_calculate_performance_score_normal(exporter, sample_stats):
    """测试计算性能评分（正常情况）"""
    score = exporter._calculate_performance_score(sample_stats)
    
    assert 0 <= score <= 100
    assert score > 0


def test_calculate_performance_score_high_hit_rate(exporter, metrics_exporter_module):
    """测试计算性能评分（高命中率）"""
    LoggerPoolStats = metrics_exporter_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.95,  # 高命中率
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,  # 低访问时间
        memory_usage_mb=5.0,  # 低内存使用
        timestamp=time.time()
    )
    
    score = exporter._calculate_performance_score(stats)
    assert score > 80  # 应该得到高分


def test_calculate_performance_score_exception(exporter, sample_stats, monkeypatch):
    """测试计算性能评分时发生异常"""
    # 创建一个会抛出异常的stats对象
    class FailingStats:
        def __getattribute__(self, name):
            if name == 'hit_rate':
                raise RuntimeError("Attribute error")
            return super().__getattribute__(name)
    
    failing_stats = FailingStats()
    # 设置其他属性
    for attr in ['pool_size', 'max_size', 'created_count', 'hit_count', 'logger_count', 
                 'total_access_count', 'avg_access_time', 'memory_usage_mb', 'timestamp']:
        setattr(failing_stats, attr, getattr(sample_stats, attr))
    
    score = exporter._calculate_performance_score(failing_stats)
    assert score == 0.0


def test_generate_recommendations_no_stats(exporter):
    """测试生成建议（无统计信息）"""
    recommendations = exporter._generate_recommendations(None)
    assert recommendations == []


def test_generate_recommendations_with_alerts(exporter, sample_stats):
    """测试生成建议（有告警状态）"""
    alert_status = {
        'hit_rate_low': True,
        'pool_usage_high': True,
        'memory_high': True
    }
    
    recommendations = exporter._generate_recommendations(sample_stats, alert_status)
    
    assert len(recommendations) > 0
    assert any("命中率" in r or "hit rate" in r.lower() for r in recommendations)
    assert any("使用率" in r or "usage" in r.lower() for r in recommendations)
    assert any("内存" in r or "memory" in r.lower() for r in recommendations)


def test_generate_recommendations_high_hit_rate(exporter, metrics_exporter_module):
    """测试生成建议（高命中率）"""
    LoggerPoolStats = metrics_exporter_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.96,  # 高命中率
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    recommendations = exporter._generate_recommendations(stats)
    assert any("优秀" in r or "excellent" in r.lower() or "良好" in r for r in recommendations)


def test_generate_recommendations_low_hit_rate(exporter, metrics_exporter_module):
    """测试生成建议（低命中率）"""
    LoggerPoolStats = metrics_exporter_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=5,
        hit_rate=0.5,  # 低命中率
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    recommendations = exporter._generate_recommendations(stats)
    assert any("偏低" in r or "low" in r.lower() or "配置" in r for r in recommendations)


def test_generate_recommendations_high_utilization(exporter, metrics_exporter_module):
    """测试生成建议（高使用率）"""
    LoggerPoolStats = metrics_exporter_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=9,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=9,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    recommendations = exporter._generate_recommendations(stats)
    assert any("使用率" in r or "usage" in r.lower() or "扩容" in r for r in recommendations)


def test_generate_recommendations_high_access_time(exporter, metrics_exporter_module):
    """测试生成建议（高访问时间）"""
    LoggerPoolStats = metrics_exporter_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=10,
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.02,  # 20ms，高于10ms阈值
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    recommendations = exporter._generate_recommendations(stats)
    assert any("访问时间" in r or "access time" in r.lower() or "优化" in r for r in recommendations)


def test_generate_recommendations_zero_max_size(exporter, metrics_exporter_module):
    """测试生成建议（max_size为0）"""
    LoggerPoolStats = metrics_exporter_module.LoggerPoolStats
    stats = LoggerPoolStats(
        pool_size=5,
        max_size=0,
        created_count=20,
        hit_count=15,
        hit_rate=0.75,
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    recommendations = exporter._generate_recommendations(stats)
    # 应该不会因为max_size为0而崩溃
    assert isinstance(recommendations, list)


def test_fallback_logger_pool_stats_definition(metrics_exporter_module, monkeypatch):
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
    module_name = "src.infrastructure.monitoring.components.logger_pool_metrics_exporter"
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
        logger_count=5,
        total_access_count=100,
        avg_access_time=0.001,
        memory_usage_mb=10.0,
        timestamp=time.time()
    )
    
    assert stats.pool_size == 5
    assert stats.hit_rate == 0.75

