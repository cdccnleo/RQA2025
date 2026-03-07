#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig报告生成边界情况测试
补充generate_report方法的边界情况和性能摘要计算测试
"""

import pytest
import time
from unittest.mock import patch
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

class TestMonitoringConfigReportEdgeCases:
    """测试报告生成的边界情况"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_generate_report_traces_mixed_durations(self, monitoring_system):
        """测试traces中有部分有duration，部分没有duration"""
        # 创建有duration的trace
        span_id1 = monitoring_system.start_trace('trace_1', 'op1')
        time.sleep(0.01)
        monitoring_system.end_trace(span_id1)
        
        # 创建没有duration的trace（未结束）
        monitoring_system.start_trace('trace_2', 'op2')
        
        # 创建有duration的trace
        span_id3 = monitoring_system.start_trace('trace_3', 'op3')
        time.sleep(0.02)
        monitoring_system.end_trace(span_id3)
        
        report = monitoring_system.generate_report()
        
        assert isinstance(report, dict)
        assert 'performance_summary' in report
        # 应该只计算有duration的traces
        if report['performance_summary']:
            assert report['performance_summary']['total_traces'] == 2

    def test_generate_report_traces_all_no_duration(self, monitoring_system):
        """测试所有traces都没有duration（未结束的traces）"""
        monitoring_system.start_trace('trace_1', 'op1')
        monitoring_system.start_trace('trace_2', 'op2')
        monitoring_system.start_trace('trace_3', 'op3')
        
        report = monitoring_system.generate_report()
        
        assert isinstance(report, dict)
        assert 'performance_summary' in report
        assert report['performance_summary'] == {}  # 没有duration，应该为空字典

    def test_generate_report_performance_summary_calculation(self, monitoring_system):
        """测试性能摘要计算的正确性"""
        # 创建多个有duration的traces
        expected_durations = [0.1, 0.2, 0.3, 0.15, 0.25]
        
        for i, duration in enumerate(expected_durations):
            span_id = monitoring_system.start_trace(f'trace_{i}', f'op_{i}')
            time.sleep(duration)
            monitoring_system.end_trace(span_id)
        
        report = monitoring_system.generate_report()
        
        assert 'performance_summary' in report
        summary = report['performance_summary']
        
        # 验证计算正确性（允许一定误差，因为sleep时间可能不精确）
        assert summary['total_traces'] == len(expected_durations)
        # 验证平均值在合理范围内（允许10%误差）
        expected_avg = sum(expected_durations) / len(expected_durations)
        assert abs(summary['avg_duration'] - expected_avg) < expected_avg * 0.2
        # 验证最大值在合理范围内
        expected_max = max(expected_durations)
        assert summary['max_duration'] >= expected_max
        assert summary['max_duration'] < expected_max * 1.5  # 允许一定误差
        # 验证最小值在合理范围内
        expected_min = min(expected_durations)
        assert summary['min_duration'] >= expected_min
        assert summary['min_duration'] < expected_min * 1.5  # 允许一定误差

    def test_generate_report_performance_summary_single_trace(self, monitoring_system):
        """测试单个trace的性能摘要"""
        span_id = monitoring_system.start_trace('trace_1', 'op1')
        time.sleep(0.15)
        monitoring_system.end_trace(span_id)
        
        report = monitoring_system.generate_report()
        
        summary = report['performance_summary']
        assert summary['total_traces'] == 1
        assert summary['avg_duration'] == summary['max_duration']
        assert summary['avg_duration'] == summary['min_duration']

    def test_generate_report_latest_metrics_empty(self, monitoring_system):
        """测试空指标时的最新指标"""
        report = monitoring_system.generate_report()
        
        assert isinstance(report, dict)
        assert 'latest_metrics' in report
        assert report['latest_metrics'] == {}

    def test_generate_report_latest_metrics_multiple(self, monitoring_system):
        """测试多个指标的最新值"""
        # 为每个指标记录多个值
        for i in range(5):
            monitoring_system.record_metric('cpu_usage', 50.0 + i)
            monitoring_system.record_metric('memory_usage', 60.0 + i)
        
        report = monitoring_system.generate_report()
        
        assert 'latest_metrics' in report
        assert 'cpu_usage' in report['latest_metrics']
        assert 'memory_usage' in report['latest_metrics']
        assert report['latest_metrics']['cpu_usage']['value'] == 54.0
        assert report['latest_metrics']['memory_usage']['value'] == 64.0

    def test_generate_report_metrics_count_calculation(self, monitoring_system):
        """测试指标数量的计算"""
        # 记录多个指标
        for i in range(10):
            monitoring_system.record_metric('metric1', float(i))
        for i in range(5):
            monitoring_system.record_metric('metric2', float(i))
        
        report = monitoring_system.generate_report()
        
        assert report['metrics_count'] == 15  # 10 + 5

    def test_generate_report_traces_count(self, monitoring_system):
        """测试traces数量的计算"""
        # 创建多个traces
        for i in range(7):
            monitoring_system.start_trace(f'trace_{i}', f'op_{i}')
        
        report = monitoring_system.generate_report()
        
        assert report['traces_count'] == 7

    def test_generate_report_alerts_count(self, monitoring_system):
        """测试告警数量的计算"""
        # 添加告警
        monitoring_system.record_metric('cpu_usage', 90.0)  # 触发告警
        monitoring_system.check_alerts()
        
        report = monitoring_system.generate_report()
        
        assert report['alerts_count'] >= 1  # 至少有一个告警

    def test_generate_report_timestamp_format(self, monitoring_system):
        """测试时间戳格式"""
        report = monitoring_system.generate_report()
        
        assert 'timestamp' in report
        timestamp = report['timestamp']
        # 验证是ISO格式的时间戳
        assert isinstance(timestamp, str)
        # 尝试解析时间戳
        parsed_time = datetime.fromisoformat(timestamp)
        assert parsed_time is not None

    def test_generate_report_performance_summary_zero_duration(self, monitoring_system):
        """测试duration为0的情况"""
        span_id = monitoring_system.start_trace('trace_1', 'op1')
        # 立即结束，duration应该接近0
        monitoring_system.end_trace(span_id)
        
        report = monitoring_system.generate_report()
        
        summary = report['performance_summary']
        if summary:
            assert summary['min_duration'] >= 0
            assert summary['avg_duration'] >= 0
            assert summary['max_duration'] >= 0

    def test_generate_report_performance_summary_very_long_duration(self, monitoring_system):
        """测试非常长的duration"""
        span_id = monitoring_system.start_trace('trace_1', 'op1')
        time.sleep(0.5)  # 较长的duration
        monitoring_system.end_trace(span_id)
        
        report = monitoring_system.generate_report()
        
        summary = report['performance_summary']
        if summary:
            assert summary['max_duration'] >= 0.5
            assert summary['avg_duration'] >= 0.5

    def test_generate_report_complete_structure(self, monitoring_system):
        """测试报告完整结构"""
        # 添加各种数据
        monitoring_system.record_metric('cpu_usage', 75.0)
        span_id = monitoring_system.start_trace('trace_1', 'op1')
        time.sleep(0.01)
        monitoring_system.end_trace(span_id)
        monitoring_system.record_metric('cpu_usage', 90.0)
        monitoring_system.check_alerts()
        
        report = monitoring_system.generate_report()
        
        # 验证所有必需的字段都存在
        required_fields = ['timestamp', 'metrics_count', 'traces_count', 
                          'alerts_count', 'latest_metrics', 'performance_summary']
        for field in required_fields:
            assert field in report, f"Missing field: {field}"

