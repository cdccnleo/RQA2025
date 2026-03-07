#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试生产环境健康评估器组件
"""

import importlib
import sys
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

import pytest


@pytest.fixture
def production_health_evaluator_module():
    """确保每次测试都重新导入模块"""
    module_name = "src.infrastructure.monitoring.components.production_health_evaluator"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


@pytest.fixture
def evaluator(production_health_evaluator_module):
    """创建ProductionHealthEvaluator实例"""
    module = production_health_evaluator_module
    return module.ProductionHealthEvaluator()


def test_initialization(evaluator):
    """测试初始化"""
    assert evaluator is not None


def test_evaluate_health_status_no_data(evaluator):
    """测试评估系统健康状态（无数据）"""
    result = evaluator.evaluate_health_status([], {}, False)
    
    assert result['status'] == 'no_data'
    assert 'message' in result


def test_evaluate_health_status_success(evaluator):
    """测试评估系统健康状态（成功）"""
    metrics_history = [
        {
            'timestamp': '2025-01-01T10:00:00',
            'cpu': {'percent': 50.0},
            'memory': {'percent': 60.0},
            'disk': {'percent': 70.0}
        }
    ]
    system_info = {'hostname': 'test-host'}
    
    result = evaluator.evaluate_health_status(metrics_history, system_info, True)
    
    assert result['status'] in ['healthy', 'warning', 'critical']
    assert 'health_score' in result
    assert 'latest_metrics' in result
    assert result['system_info'] == system_info
    assert result['monitoring_active'] is True
    assert 'evaluation_time' in result


def test_evaluate_health_status_exception(evaluator, monkeypatch):
    """测试评估系统健康状态（异常）"""
    metrics_history = [{'invalid': 'data'}]
    
    # 模拟_calculate_health_score抛出异常
    def failing_score(*args, **kwargs):
        raise RuntimeError("Calculation error")
    
    monkeypatch.setattr(evaluator, '_calculate_health_score', failing_score)
    
    result = evaluator.evaluate_health_status(metrics_history, {}, False)
    
    assert result['status'] == 'error'
    assert 'message' in result
    assert 'evaluation_time' in result


def test_generate_performance_report_no_data(evaluator):
    """测试生成性能报告（无数据）"""
    result = evaluator.generate_performance_report([], [])
    
    assert 'error' in result
    assert result['error'] == 'No metrics data available'


def test_generate_performance_report_success(evaluator):
    """测试生成性能报告（成功）"""
    metrics_history = [
        {
            'timestamp': '2025-01-01T10:00:00',
            'cpu': {'percent': 50.0},
            'memory': {'percent': 60.0},
            'disk': {'percent': 70.0}
        },
        {
            'timestamp': '2025-01-01T10:01:00',
            'cpu': {'percent': 60.0},
            'memory': {'percent': 70.0},
            'disk': {'percent': 80.0}
        }
    ]
    alerts_history = [
        {'type': 'cpu_high', 'level': 'warning'}
    ]
    
    result = evaluator.generate_performance_report(metrics_history, alerts_history)
    
    assert 'time_range' in result
    assert 'cpu_stats' in result
    assert 'memory_stats' in result
    assert 'disk_stats' in result
    assert 'alerts_summary' in result
    assert 'report_generated_at' in result


def test_generate_performance_report_exception(evaluator, monkeypatch):
    """测试生成性能报告（异常）"""
    metrics_history = [{'timestamp': '2025-01-01T10:00:00'}]
    
    # 模拟_get_time_range_info抛出异常
    def failing_range(*args, **kwargs):
        raise RuntimeError("Range error")
    
    monkeypatch.setattr(evaluator, '_get_time_range_info', failing_range)
    
    result = evaluator.generate_performance_report(metrics_history, [])
    
    assert 'error' in result


def test_calculate_health_score_healthy(evaluator):
    """测试计算健康评分（健康）"""
    metrics = {
        'cpu': {'percent': 50.0},
        'memory': {'percent': 60.0},
        'disk': {'percent': 70.0}
    }
    
    score = evaluator._calculate_health_score(metrics)
    
    assert 0 <= score <= 100
    assert score >= 80  # 健康状态


def test_calculate_health_score_warning(evaluator):
    """测试计算健康评分（警告）"""
    metrics = {
        'cpu': {'percent': 70.0},  # 60-80之间，扣10分
        'memory': {'percent': 75.0},  # 70-85之间，扣10分
        'disk': {'percent': 85.0}  # 80-90之间，扣15分
    }
    
    score = evaluator._calculate_health_score(metrics)
    
    assert 0 <= score <= 100
    assert 60 <= score < 80  # 警告状态


def test_calculate_health_score_critical(evaluator):
    """测试计算健康评分（严重）"""
    metrics = {
        'cpu': {'percent': 90.0},  # >80，扣20分
        'memory': {'percent': 90.0},  # >85，扣20分
        'disk': {'percent': 95.0}  # >90，扣30分
    }
    
    score = evaluator._calculate_health_score(metrics)
    
    assert 0 <= score <= 100
    assert score < 60  # 严重状态


def test_calculate_health_score_exception(evaluator):
    """测试计算健康评分（异常）"""
    # 创建一个会导致异常的数据 - 在访问嵌套字典时抛出异常
    class FailingDict(dict):
        def get(self, key, default=None):
            if key == 'cpu':
                raise RuntimeError("Access error")
            return super().get(key, default)
    
    metrics = FailingDict({'cpu': {'percent': 50.0}})
    
    score = evaluator._calculate_health_score(metrics)
    
    assert score == 0


def test_calculate_cpu_penalty_high(evaluator):
    """测试计算CPU使用率惩罚分数（高）"""
    assert evaluator._calculate_cpu_penalty(85.0) == 20


def test_calculate_cpu_penalty_medium(evaluator):
    """测试计算CPU使用率惩罚分数（中）"""
    assert evaluator._calculate_cpu_penalty(70.0) == 10


def test_calculate_cpu_penalty_low(evaluator):
    """测试计算CPU使用率惩罚分数（低）"""
    assert evaluator._calculate_cpu_penalty(50.0) == 0


def test_calculate_memory_penalty_high(evaluator):
    """测试计算内存使用率惩罚分数（高）"""
    assert evaluator._calculate_memory_penalty(90.0) == 20


def test_calculate_memory_penalty_medium(evaluator):
    """测试计算内存使用率惩罚分数（中）"""
    assert evaluator._calculate_memory_penalty(75.0) == 10


def test_calculate_memory_penalty_low(evaluator):
    """测试计算内存使用率惩罚分数（低）"""
    assert evaluator._calculate_memory_penalty(60.0) == 0


def test_calculate_disk_penalty_high(evaluator):
    """测试计算磁盘使用率惩罚分数（高）"""
    assert evaluator._calculate_disk_penalty(95.0) == 30


def test_calculate_disk_penalty_medium(evaluator):
    """测试计算磁盘使用率惩罚分数（中）"""
    assert evaluator._calculate_disk_penalty(85.0) == 15


def test_calculate_disk_penalty_low(evaluator):
    """测试计算磁盘使用率惩罚分数（低）"""
    assert evaluator._calculate_disk_penalty(70.0) == 0


def test_determine_health_status_healthy(evaluator):
    """测试根据健康评分确定健康状态（健康）"""
    assert evaluator._determine_health_status(85) == 'healthy'
    assert evaluator._determine_health_status(100) == 'healthy'


def test_determine_health_status_warning(evaluator):
    """测试根据健康评分确定健康状态（警告）"""
    assert evaluator._determine_health_status(70) == 'warning'
    assert evaluator._determine_health_status(60) == 'warning'


def test_determine_health_status_critical(evaluator):
    """测试根据健康评分确定健康状态（严重）"""
    assert evaluator._determine_health_status(50) == 'critical'
    assert evaluator._determine_health_status(0) == 'critical'


def test_get_time_range_info_success(evaluator):
    """测试获取时间范围信息（成功）"""
    metrics_history = [
        {'timestamp': '2025-01-01T10:00:00'},
        {'timestamp': '2025-01-01T10:01:00'},
        {'timestamp': '2025-01-01T10:02:00'}
    ]
    
    result = evaluator._get_time_range_info(metrics_history)
    
    assert result['start'] == '2025-01-01T10:00:00'
    assert result['end'] == '2025-01-01T10:02:00'
    assert result['data_points'] == 3


def test_get_time_range_info_empty(evaluator):
    """测试获取时间范围信息（空）"""
    result = evaluator._get_time_range_info([])
    
    assert result == {}


def test_calculate_cpu_statistics_success(evaluator):
    """测试计算CPU统计信息（成功）"""
    metrics_history = [
        {'cpu': {'percent': 50.0}},
        {'cpu': {'percent': 60.0}},
        {'cpu': {'percent': 70.0}}
    ]
    
    result = evaluator._calculate_cpu_statistics(metrics_history)
    
    assert result['avg_percent'] == 60.0
    assert result['max_percent'] == 70.0
    assert result['min_percent'] == 50.0
    assert result['data_points'] == 3


def test_calculate_cpu_statistics_empty(evaluator):
    """测试计算CPU统计信息（空）"""
    metrics_history = [
        {'cpu': {}},  # 没有percent
        {'memory': {'percent': 60.0}}  # 没有cpu
    ]
    
    result = evaluator._calculate_cpu_statistics(metrics_history)
    
    assert result == {}


def test_calculate_cpu_statistics_none_percent(evaluator):
    """测试计算CPU统计信息（percent为None）"""
    metrics_history = [
        {'cpu': {'percent': None}},  # percent为None
        {'cpu': {'percent': 50.0}}
    ]
    
    result = evaluator._calculate_cpu_statistics(metrics_history)
    
    # 应该只包含非None的值
    assert result['data_points'] == 1
    assert result['avg_percent'] == 50.0


def test_calculate_cpu_statistics_exception(evaluator):
    """测试计算CPU统计信息（异常）"""
    metrics_history = [object()]  # 不是字典
    
    result = evaluator._calculate_cpu_statistics(metrics_history)
    
    assert result == {}


def test_calculate_memory_statistics_success(evaluator):
    """测试计算内存统计信息（成功）"""
    metrics_history = [
        {'memory': {'percent': 60.0}},
        {'memory': {'percent': 70.0}},
        {'memory': {'percent': 80.0}}
    ]
    
    result = evaluator._calculate_memory_statistics(metrics_history)
    
    assert result['avg_percent'] == 70.0
    assert result['max_percent'] == 80.0
    assert result['min_percent'] == 60.0
    assert result['data_points'] == 3


def test_calculate_memory_statistics_empty(evaluator):
    """测试计算内存统计信息（空）"""
    metrics_history = [
        {'memory': {}},  # 没有percent
        {'cpu': {'percent': 60.0}}  # 没有memory
    ]
    
    result = evaluator._calculate_memory_statistics(metrics_history)
    
    assert result == {}


def test_calculate_memory_statistics_none_percent(evaluator):
    """测试计算内存统计信息（percent为None）"""
    metrics_history = [
        {'memory': {'percent': None}},  # percent为None
        {'memory': {'percent': 60.0}}
    ]
    
    result = evaluator._calculate_memory_statistics(metrics_history)
    
    # 应该只包含非None的值
    assert result['data_points'] == 1
    assert result['avg_percent'] == 60.0


def test_calculate_memory_statistics_exception(evaluator):
    """测试计算内存统计信息（异常）"""
    # 创建一个会导致sum或max抛出异常的数据
    class FailingNumber:
        def __add__(self, other):
            raise RuntimeError("Add error")
    
    metrics_history = [
        {'memory': {'percent': FailingNumber()}}
    ]
    
    result = evaluator._calculate_memory_statistics(metrics_history)
    
    assert result == {}


def test_calculate_disk_statistics_success(evaluator):
    """测试计算磁盘统计信息（成功）"""
    metrics_history = [
        {'disk': {'percent': 70.0}},
        {'disk': {'percent': 80.0}},
        {'disk': {'percent': 90.0}}
    ]
    
    result = evaluator._calculate_disk_statistics(metrics_history)
    
    assert result['avg_percent'] == 80.0
    assert result['max_percent'] == 90.0
    assert result['min_percent'] == 70.0
    assert result['data_points'] == 3


def test_calculate_disk_statistics_empty(evaluator):
    """测试计算磁盘统计信息（空）"""
    metrics_history = [
        {'disk': {}},  # 没有percent
        {'cpu': {'percent': 60.0}}  # 没有disk
    ]
    
    result = evaluator._calculate_disk_statistics(metrics_history)
    
    assert result == {}


def test_calculate_disk_statistics_none_percent(evaluator):
    """测试计算磁盘统计信息（percent为None）"""
    metrics_history = [
        {'disk': {'percent': None}},  # percent为None
        {'disk': {'percent': 70.0}}
    ]
    
    result = evaluator._calculate_disk_statistics(metrics_history)
    
    # 应该只包含非None的值
    assert result['data_points'] == 1
    assert result['avg_percent'] == 70.0


def test_calculate_disk_statistics_exception(evaluator):
    """测试计算磁盘统计信息（异常）"""
    # 创建一个会导致sum或max抛出异常的数据
    class FailingNumber:
        def __add__(self, other):
            raise RuntimeError("Add error")
    
    metrics_history = [
        {'disk': {'percent': FailingNumber()}}
    ]
    
    result = evaluator._calculate_disk_statistics(metrics_history)
    
    assert result == {}


def test_generate_alerts_summary_empty(evaluator):
    """测试生成告警摘要（空）"""
    result = evaluator._generate_alerts_summary([])
    
    assert result['total_alerts'] == 0
    assert result['alert_types'] == []
    assert result['recent_alerts'] == []


def test_generate_alerts_summary_success(evaluator):
    """测试生成告警摘要（成功）"""
    alerts_history = [
        {'type': 'cpu_high', 'level': 'warning'},
        {'type': 'memory_high', 'level': 'error'},
        {'type': 'cpu_high', 'level': 'warning'},
        {'type': 'disk_high', 'level': 'error'},
        {'type': 'cpu_high', 'level': 'warning'},
        {'type': 'memory_high', 'level': 'error'},
        {'type': 'disk_high', 'level': 'error'}
    ]
    
    result = evaluator._generate_alerts_summary(alerts_history)
    
    assert result['total_alerts'] == 7
    assert len(result['alert_types']) == 3  # cpu_high, memory_high, disk_high
    assert len(result['recent_alerts']) == 5  # 最近5个
    assert 'alert_levels' in result
    assert result['alert_levels']['warning'] == 3
    assert result['alert_levels']['error'] == 4


def test_generate_alerts_summary_exception(evaluator):
    """测试生成告警摘要（异常）"""
    alerts_history = [object()]  # 不是字典
    
    result = evaluator._generate_alerts_summary(alerts_history)
    
    assert result == {}


def test_count_alert_levels_success(evaluator):
    """测试统计告警级别（成功）"""
    alerts_history = [
        {'level': 'warning'},
        {'level': 'error'},
        {'level': 'warning'},
        {'level': 'info'},
        {'level': 'error'}
    ]
    
    result = evaluator._count_alert_levels(alerts_history)
    
    assert result['warning'] == 2
    assert result['error'] == 2
    assert result['info'] == 1


def test_count_alert_levels_exception(evaluator):
    """测试统计告警级别（异常）"""
    alerts_history = [object()]  # 不是字典
    
    result = evaluator._count_alert_levels(alerts_history)
    
    assert result == {}


def test_get_health_recommendations_critical(evaluator):
    """测试获取健康建议（严重）"""
    metrics = {
        'cpu': {'percent': 90.0},
        'memory': {'percent': 90.0},
        'disk': {'percent': 95.0}
    }
    
    recommendations = evaluator.get_health_recommendations(50, metrics)
    
    assert len(recommendations) > 0
    assert any('严重' in r for r in recommendations)
    assert any('CPU使用率过高' in r for r in recommendations)
    assert any('内存使用率过高' in r for r in recommendations)
    assert any('磁盘空间不足' in r for r in recommendations)


def test_get_health_recommendations_healthy(evaluator):
    """测试获取健康建议（健康）"""
    metrics = {
        'cpu': {'percent': 50.0},
        'memory': {'percent': 60.0},
        'disk': {'percent': 70.0}
    }
    
    recommendations = evaluator.get_health_recommendations(95, metrics)
    
    assert len(recommendations) > 0
    assert any('运行良好' in r for r in recommendations)


def test_get_health_recommendations_exception(evaluator):
    """测试获取健康建议（异常）"""
    metrics = object()  # 不是字典
    
    recommendations = evaluator.get_health_recommendations(50, metrics)
    
    # 异常应该被捕获，返回包含错误信息的建议
    assert len(recommendations) > 0
    assert any('出错' in r for r in recommendations)

