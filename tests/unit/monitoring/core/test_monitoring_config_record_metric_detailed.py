#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig记录指标详细测试
补充record_metric方法的详细测试，包括tags、timestamp、边界情况等
"""

import pytest
from unittest.mock import patch, MagicMock
from datetime import datetime

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

class TestMonitoringSystemRecordMetricDetailed:
    """测试MonitoringSystem记录指标的详细功能"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_record_metric_with_tags(self, monitoring_system):
        """测试记录指标时包含tags"""
        tags = {'env': 'test', 'host': 'server1'}
        monitoring_system.record_metric('cpu_usage', 50.0, tags)
        
        assert 'cpu_usage' in monitoring_system.metrics
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert metric['tags'] == tags
        assert metric['tags']['env'] == 'test'
        assert metric['tags']['host'] == 'server1'

    def test_record_metric_without_tags(self, monitoring_system):
        """测试记录指标时不包含tags"""
        monitoring_system.record_metric('cpu_usage', 50.0)
        
        assert 'cpu_usage' in monitoring_system.metrics
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert metric['tags'] == {}

    def test_record_metric_with_none_tags(self, monitoring_system):
        """测试记录指标时tags为None"""
        monitoring_system.record_metric('cpu_usage', 50.0, None)
        
        assert 'cpu_usage' in monitoring_system.metrics
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert metric['tags'] == {}

    def test_record_metric_with_empty_tags(self, monitoring_system):
        """测试记录指标时tags为空字典"""
        monitoring_system.record_metric('cpu_usage', 50.0, {})
        
        assert 'cpu_usage' in monitoring_system.metrics
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert metric['tags'] == {}

    def test_record_metric_timestamp_format(self, monitoring_system):
        """测试指标timestamp格式"""
        monitoring_system.record_metric('cpu_usage', 50.0)
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert 'timestamp' in metric
        assert isinstance(metric['timestamp'], str)
        # 验证是ISO格式
        try:
            datetime.fromisoformat(metric['timestamp'])
        except ValueError:
            pytest.fail("Invalid timestamp format")

    def test_record_metric_timestamp_different(self, monitoring_system):
        """测试不同指标的timestamp不同"""
        monitoring_system.record_metric('cpu_usage', 50.0)
        
        import time
        time.sleep(0.01)  # 确保时间不同
        
        monitoring_system.record_metric('memory_usage', 60.0)
        
        cpu_metric = monitoring_system.metrics['cpu_usage'][0]
        memory_metric = monitoring_system.metrics['memory_usage'][0]
        
        assert cpu_metric['timestamp'] != memory_metric['timestamp']

    def test_record_metric_name_preserved(self, monitoring_system):
        """测试指标名称被正确保留"""
        monitoring_system.record_metric('custom_metric_name', 42.0)
        
        metric = monitoring_system.metrics['custom_metric_name'][0]
        assert metric['name'] == 'custom_metric_name'

    def test_record_metric_value_preserved(self, monitoring_system):
        """测试指标值被正确保留"""
        monitoring_system.record_metric('cpu_usage', 75.5)
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert metric['value'] == 75.5

    def test_record_metric_float_value(self, monitoring_system):
        """测试记录浮点数指标值"""
        monitoring_system.record_metric('cpu_usage', 50.5)
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert isinstance(metric['value'], float)
        assert metric['value'] == 50.5

    def test_record_metric_zero_value(self, monitoring_system):
        """测试记录零值指标"""
        monitoring_system.record_metric('cpu_usage', 0.0)
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert metric['value'] == 0.0

    def test_record_metric_negative_value(self, monitoring_system):
        """测试记录负值指标"""
        monitoring_system.record_metric('temperature', -10.0)
        
        metric = monitoring_system.metrics['temperature'][0]
        assert metric['value'] == -10.0

    def test_record_metric_very_large_value(self, monitoring_system):
        """测试记录非常大的指标值"""
        large_value = 1e10
        monitoring_system.record_metric('large_metric', large_value)
        
        metric = monitoring_system.metrics['large_metric'][0]
        assert metric['value'] == large_value

    def test_record_metric_multiple_tags(self, monitoring_system):
        """测试记录多个tags"""
        tags = {
            'env': 'production',
            'host': 'server1',
            'region': 'us-east',
            'service': 'api'
        }
        monitoring_system.record_metric('cpu_usage', 50.0, tags)
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert len(metric['tags']) == 4
        assert metric['tags']['env'] == 'production'
        assert metric['tags']['host'] == 'server1'
        assert metric['tags']['region'] == 'us-east'
        assert metric['tags']['service'] == 'api'

    def test_record_metric_tags_not_shared(self, monitoring_system):
        """测试不同指标的tags不共享"""
        tags1 = {'env': 'test1'}
        tags2 = {'env': 'test2'}
        
        monitoring_system.record_metric('metric1', 10.0, tags1)
        monitoring_system.record_metric('metric2', 20.0, tags2)
        
        metric1 = monitoring_system.metrics['metric1'][0]
        metric2 = monitoring_system.metrics['metric2'][0]
        
        assert metric1['tags']['env'] == 'test1'
        assert metric2['tags']['env'] == 'test2'
        assert metric1['tags'] != metric2['tags']

    def test_record_metric_same_name_multiple_times(self, monitoring_system):
        """测试同一指标名称记录多次"""
        for i in range(5):
            monitoring_system.record_metric('cpu_usage', float(i * 10))
        
        assert len(monitoring_system.metrics['cpu_usage']) == 5
        assert monitoring_system.metrics['cpu_usage'][0]['value'] == 0.0
        assert monitoring_system.metrics['cpu_usage'][4]['value'] == 40.0

    def test_record_metric_structure_complete(self, monitoring_system):
        """测试指标结构完整性"""
        tags = {'env': 'test'}
        monitoring_system.record_metric('cpu_usage', 50.0, tags)
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        
        # 验证所有必需字段存在
        assert 'name' in metric
        assert 'value' in metric
        assert 'timestamp' in metric
        assert 'tags' in metric
        
        # 验证字段值
        assert metric['name'] == 'cpu_usage'
        assert metric['value'] == 50.0
        assert isinstance(metric['timestamp'], str)
        assert isinstance(metric['tags'], dict)

    def test_record_metric_tags_shared_reference(self, monitoring_system):
        """测试记录后修改原始tags会影响已记录的指标（因为使用字典引用）"""
        original_tags = {'env': 'original'}
        monitoring_system.record_metric('cpu_usage', 50.0, original_tags)
        
        # 修改原始tags
        original_tags['env'] = 'modified'
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        # 由于使用字典引用，已记录的指标tags会受影响
        assert metric['tags']['env'] == 'modified'

    def test_record_metric_special_characters_in_name(self, monitoring_system):
        """测试指标名称包含特殊字符"""
        special_name = 'metric.with-special_chars@123'
        monitoring_system.record_metric(special_name, 50.0)
        
        assert special_name in monitoring_system.metrics
        metric = monitoring_system.metrics[special_name][0]
        assert metric['name'] == special_name

    def test_record_metric_empty_name(self, monitoring_system):
        """测试空字符串指标名称"""
        monitoring_system.record_metric('', 50.0)
        
        assert '' in monitoring_system.metrics
        metric = monitoring_system.metrics[''][0]
        assert metric['name'] == ''

    def test_record_metric_tags_with_nested_values(self, monitoring_system):
        """测试tags中包含复杂值"""
        tags = {
            'number': 42,
            'boolean': True,
            'string': 'value'
        }
        monitoring_system.record_metric('cpu_usage', 50.0, tags)
        
        metric = monitoring_system.metrics['cpu_usage'][0]
        assert metric['tags']['number'] == 42
        assert metric['tags']['boolean'] is True
        assert metric['tags']['string'] == 'value'

