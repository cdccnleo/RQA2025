# -*- coding: utf-8 -*-
"""
性能监控器真实实现测试
测试 PerformanceMonitor 的核心功能
"""

import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
import time
from datetime import datetime, timedelta

from src.data.monitoring.performance_monitor import PerformanceMonitor, PerformanceMetric, PerformanceAlert


@pytest.fixture
def performance_monitor():
    """创建性能监控器实例"""
    return PerformanceMonitor(max_history=100)


def test_performance_monitor_initialization(performance_monitor):
    """测试性能监控器初始化"""
    assert performance_monitor.max_history == 100
    assert not performance_monitor.is_monitoring
    assert len(performance_monitor.metrics) == 0
    assert len(performance_monitor.alerts) == 0


def test_record_metric(performance_monitor):
    """测试记录性能指标"""
    performance_monitor.record_metric('test_metric', 10.5, 'ms')
    
    assert 'test_metric' in performance_monitor.metrics
    assert len(performance_monitor.metrics['test_metric']) == 1
    assert performance_monitor.metrics['test_metric'][0].value == 10.5


def test_record_cache_hit_rate(performance_monitor):
    """测试记录缓存命中率"""
    performance_monitor.record_cache_hit_rate(0.85)
    
    metric = performance_monitor.get_current_metric('cache_hit_rate')
    assert metric is not None
    assert metric.value == 0.85
    assert metric.unit == '%'


def test_record_data_load_time(performance_monitor):
    """测试记录数据加载时间"""
    performance_monitor.record_data_load_time(2.5)
    
    metric = performance_monitor.get_current_metric('data_load_time')
    assert metric is not None
    assert metric.value == 2.5
    assert metric.unit == 'seconds'


def test_record_memory_usage(performance_monitor):
    """测试记录内存使用率"""
    performance_monitor.record_memory_usage(0.75)
    
    metric = performance_monitor.get_current_metric('memory_usage')
    assert metric is not None
    assert metric.value == 0.75


def test_record_error_rate(performance_monitor):
    """测试记录错误率"""
    performance_monitor.record_error_rate(0.05)
    
    metric = performance_monitor.get_current_metric('error_rate')
    assert metric is not None
    assert metric.value == 0.05


def test_record_throughput(performance_monitor):
    """测试记录吞吐量"""
    performance_monitor.record_throughput(100.0)
    
    metric = performance_monitor.get_current_metric('throughput')
    assert metric is not None
    assert metric.value == 100.0


def test_get_metric_history(performance_monitor):
    """测试获取指标历史"""
    # 记录多个指标
    for i in range(5):
        performance_monitor.record_metric('test_metric', float(i), 'ms')
        time.sleep(0.01)
    
    history = performance_monitor.get_metric_history('test_metric', hours=1)
    
    assert len(history) == 5
    assert all(isinstance(m, PerformanceMetric) for m in history)


def test_get_current_metric(performance_monitor):
    """测试获取当前指标"""
    performance_monitor.record_metric('test_metric', 10.0, 'ms')
    
    current = performance_monitor.get_current_metric('test_metric')
    
    assert current is not None
    assert current.value == 10.0


def test_get_current_metric_nonexistent(performance_monitor):
    """测试获取不存在的指标"""
    current = performance_monitor.get_current_metric('nonexistent')
    
    assert current is None


def test_get_metric_statistics(performance_monitor):
    """测试获取指标统计信息"""
    # 记录多个指标值
    values = [10.0, 20.0, 30.0, 40.0, 50.0]
    for value in values:
        performance_monitor.record_metric('test_metric', value, 'ms')
    
    stats = performance_monitor.get_metric_statistics('test_metric', hours=1)
    
    assert 'avg' in stats
    assert 'min' in stats
    assert 'max' in stats
    assert stats['avg'] == 30.0
    assert stats['min'] == 10.0
    assert stats['max'] == 50.0


def test_alert_thresholds(performance_monitor):
    """测试告警阈值"""
    # 记录低于阈值的缓存命中率（应该触发告警）
    performance_monitor.record_cache_hit_rate(0.3)  # 低于critical阈值0.4
    
    assert len(performance_monitor.alerts) > 0
    assert performance_monitor.alerts[-1].level == 'critical'


def test_start_stop_monitoring(performance_monitor):
    """测试启动和停止监控"""
    performance_monitor.start_monitoring()
    
    assert performance_monitor.is_monitoring is True
    assert performance_monitor.monitor_thread is not None
    
    performance_monitor.stop_monitoring()
    
    assert performance_monitor.is_monitoring is False


def test_metric_history_limit(performance_monitor):
    """测试指标历史记录限制"""
    # 记录超过max_history的指标
    for i in range(150):
        performance_monitor.record_metric('test_metric', float(i), 'ms')
    
    # 应该只保留最近的100条记录
    assert len(performance_monitor.metrics['test_metric']) == 100


def test_set_alert_threshold(performance_monitor):
    """测试设置告警阈值"""
    performance_monitor.set_alert_threshold('custom_metric', 'warning', 50.0)
    
    assert 'custom_metric' in performance_monitor.alert_thresholds
    assert performance_monitor.alert_thresholds['custom_metric']['warning'] == 50.0


def test_get_recent_alerts(performance_monitor):
    """测试获取最近告警"""
    # 记录触发告警的指标
    performance_monitor.record_cache_hit_rate(0.3)  # 触发critical告警
    
    alerts = performance_monitor.get_recent_alerts(hours=1)
    
    assert len(alerts) > 0
    assert all(isinstance(alert, PerformanceAlert) for alert in alerts)


def test_get_all_metrics_summary(performance_monitor):
    """测试获取所有指标摘要"""
    performance_monitor.record_metric('metric1', 10.0, 'ms')
    performance_monitor.record_metric('metric2', 20.0, 'ms')
    
    summary = performance_monitor.get_all_metrics_summary()
    
    assert isinstance(summary, dict)
    assert 'metric1' in summary
    assert 'metric2' in summary


def test_export_metrics(performance_monitor):
    """测试导出指标"""
    performance_monitor.record_metric('test_metric', 10.0, 'ms')
    
    exported = performance_monitor.export_metrics(format='json')
    
    assert isinstance(exported, str)
    assert len(exported) > 0

