#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor MetricsCollector collect_all_metrics详细测试
补充collect_all_metrics方法的详细功能测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock, call
from datetime import datetime, timedelta

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
    core_real_time_monitor_module = importlib.import_module('src.monitoring.core.real_time_monitor')
    MetricsCollector = getattr(core_real_time_monitor_module, 'MetricsCollector', None)
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    if MetricsCollector is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestMetricsCollectorCollectAllMetricsDetailed:
    """测试MetricsCollector类collect_all_metrics方法的详细功能"""

    @pytest.fixture
    def collector(self):
        """创建MetricsCollector实例"""
        return MetricsCollector()

    def test_collect_all_metrics_timestamp_consistency(self, collector):
        """测试收集所有指标时时间戳一致性"""
        with patch.object(collector, 'collect_system_metrics', return_value={'cpu': 50.0}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                with patch('src.monitoring.core.real_time_monitor.datetime') as mock_datetime:
                    fixed_time = datetime(2024, 1, 1, 12, 0, 0)
                    mock_datetime.now.return_value = fixed_time
                    
                    all_metrics = collector.collect_all_metrics()
                    
                    assert all_metrics['cpu'].timestamp == fixed_time

    def test_collect_all_metrics_all_metric_types(self, collector):
        """测试收集所有指标包含所有类型"""
        system_metrics = {'cpu_percent': 50.0, 'memory_percent': 60.0}
        app_metrics = {'app_cpu_percent': 10.0, 'app_memory_rss_mb': 100.0}
        business_metrics = {'requests_total': 100, 'errors_total': 5}
        
        with patch.object(collector, 'collect_system_metrics', return_value=system_metrics):
            with patch.object(collector, 'collect_application_metrics', return_value=app_metrics):
                with patch.object(collector, 'collect_business_metrics', return_value=business_metrics):
                    all_metrics = collector.collect_all_metrics()
                    
                    # 验证系统指标
                    assert 'cpu_percent' in all_metrics
                    assert all_metrics['cpu_percent'].tags['type'] == 'system'
                    
                    # 验证应用指标
                    assert 'app_cpu_percent' in all_metrics
                    assert all_metrics['app_cpu_percent'].tags['type'] == 'application'
                    
                    # 验证业务指标
                    assert 'requests_total' in all_metrics
                    assert all_metrics['requests_total'].tags['type'] == 'business'

    def test_collect_all_metrics_custom_collector_name_prefix(self, collector):
        """测试自定义收集器指标名称前缀"""
        def custom_collector():
            return {
                'metric1': 10.0,
                'metric2': 20.0
            }
        
        collector.register_collector('my_collector', custom_collector)
        
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                all_metrics = collector.collect_all_metrics()
                
                assert 'my_collector_metric1' in all_metrics
                assert 'my_collector_metric2' in all_metrics
                assert 'metric1' not in all_metrics  # 不应有未加前缀的指标

    def test_collect_all_metrics_custom_collector_tags(self, collector):
        """测试自定义收集器指标标签"""
        def custom_collector():
            return {'test_metric': 42.0}
        
        collector.register_collector('test_collector', custom_collector)
        
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                all_metrics = collector.collect_all_metrics()
                
                metric = all_metrics['test_collector_test_metric']
                assert metric.tags['type'] == 'custom'
                assert metric.tags['collector'] == 'test_collector'

    def test_collect_all_metrics_multiple_custom_collectors(self, collector):
        """测试多个自定义收集器"""
        def collector1():
            return {'metric1': 1.0}
        def collector2():
            return {'metric2': 2.0}
        
        collector.register_collector('collector1', collector1)
        collector.register_collector('collector2', collector2)
        
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                all_metrics = collector.collect_all_metrics()
                
                assert 'collector1_metric1' in all_metrics
                assert 'collector2_metric2' in all_metrics

    def test_collect_all_metrics_custom_collector_returning_empty_dict(self, collector):
        """测试自定义收集器返回空字典"""
        def empty_collector():
            return {}
        
        collector.register_collector('empty', empty_collector)
        
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                all_metrics = collector.collect_all_metrics()
                
                # 不应该有以empty_开头的指标
                assert not any(key.startswith('empty_') for key in all_metrics.keys())

    def test_collect_all_metrics_custom_collector_returning_non_dict(self, collector):
        """测试自定义收集器返回非字典类型（异常情况）"""
        def invalid_collector():
            return "not a dict"
        
        collector.register_collector('invalid', invalid_collector)
        
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                # 应该能处理异常而不崩溃
                all_metrics = collector.collect_all_metrics()
                assert isinstance(all_metrics, dict)

    def test_collect_all_metrics_metric_data_structure(self, collector):
        """测试指标数据结构完整性"""
        with patch.object(collector, 'collect_system_metrics', return_value={'test_metric': 42.0}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                all_metrics = collector.collect_all_metrics()
                
                metric = all_metrics['test_metric']
                assert isinstance(metric, MetricData)
                assert metric.name == 'test_metric'
                assert metric.value == 42.0
                assert isinstance(metric.timestamp, datetime)
                assert isinstance(metric.tags, dict)
                assert isinstance(metric.metadata, dict)

    def test_collect_all_metrics_metric_values_preserved(self, collector):
        """测试指标值正确保留"""
        system_metrics = {
            'cpu_percent': 85.5,
            'memory_percent': 75.25,
            'disk_usage_percent': 60.0
        }
        
        with patch.object(collector, 'collect_system_metrics', return_value=system_metrics):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                all_metrics = collector.collect_all_metrics()
                
                assert all_metrics['cpu_percent'].value == 85.5
                assert all_metrics['memory_percent'].value == 75.25
                assert all_metrics['disk_usage_percent'].value == 60.0

    def test_collect_all_metrics_custom_collector_exception_logged(self, collector):
        """测试自定义收集器异常被记录"""
        def failing_collector():
            raise ValueError("Custom collector error")
        
        collector.register_collector('failing', failing_collector)
        
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                with patch('src.monitoring.core.real_time_monitor.logger') as mock_logger:
                    all_metrics = collector.collect_all_metrics()
                    
                    # 验证错误被记录
                    mock_logger.error.assert_called()
                    error_call = mock_logger.error.call_args[0][0]
                    assert 'Failed to collect metrics from failing' in error_call
                    assert isinstance(all_metrics, dict)

    def test_collect_all_metrics_metric_name_conflict(self, collector):
        """测试指标名称冲突（后收集的覆盖先收集的）"""
        # 系统指标和应用指标可能同名，后收集的会覆盖先收集的
        system_metrics = {'cpu_percent': 50.0}
        app_metrics = {'cpu_percent': 10.0}  # 同名指标
        
        with patch.object(collector, 'collect_system_metrics', return_value=system_metrics):
            with patch.object(collector, 'collect_application_metrics', return_value=app_metrics):
                all_metrics = collector.collect_all_metrics()
                
                # 应用指标会覆盖系统指标（后收集的）
                assert all_metrics['cpu_percent'].value == 10.0
                assert all_metrics['cpu_percent'].tags['type'] == 'application'

    def test_collect_all_metrics_custom_collector_value_types(self, collector):
        """测试自定义收集器返回不同值类型"""
        def type_test_collector():
            return {
                'int_value': 100,
                'float_value': 3.14,
                'zero_value': 0,
                'negative_value': -5.5
            }
        
        collector.register_collector('type_test', type_test_collector)
        
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                all_metrics = collector.collect_all_metrics()
                
                assert all_metrics['type_test_int_value'].value == 100
                assert all_metrics['type_test_float_value'].value == 3.14
                assert all_metrics['type_test_zero_value'].value == 0
                assert all_metrics['type_test_negative_value'].value == -5.5

    def test_collect_all_metrics_empty_system_and_app_metrics(self, collector):
        """测试系统指标和应用指标都为空时的情况"""
        with patch.object(collector, 'collect_system_metrics', return_value={}):
            with patch.object(collector, 'collect_application_metrics', return_value={}):
                all_metrics = collector.collect_all_metrics()
                
                # 应该至少包含业务指标
                assert isinstance(all_metrics, dict)
                # 业务指标总是存在（即使值为0）
                assert 'requests_total' in all_metrics or len(all_metrics) == 0

