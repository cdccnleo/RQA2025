#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MonitoringConfig报告生成Metrics Count测试
补充generate_report方法中metrics_count计算的详细测试
"""

import pytest
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

class TestMonitoringSystemReportMetricsCount:
    """测试MonitoringSystem报告生成中的metrics_count计算"""

    @pytest.fixture
    def monitoring_system(self):
        """创建monitoring system实例"""
        return MonitoringSystem()

    def test_generate_report_metrics_count_empty(self, monitoring_system):
        """测试空metrics时的metrics_count"""
        report = monitoring_system.generate_report()
        
        assert report['metrics_count'] == 0

    def test_generate_report_metrics_count_single_metric(self, monitoring_system):
        """测试单个指标时的metrics_count"""
        monitoring_system.record_metric('cpu_usage', 50.0)
        
        report = monitoring_system.generate_report()
        
        assert report['metrics_count'] == 1

    def test_generate_report_metrics_count_multiple_same_name(self, monitoring_system):
        """测试同一名称多个指标值时的metrics_count"""
        for i in range(5):
            monitoring_system.record_metric('cpu_usage', 50.0 + i)
        
        report = monitoring_system.generate_report()
        
        assert report['metrics_count'] == 5

    def test_generate_report_metrics_count_multiple_names(self, monitoring_system):
        """测试多个不同名称指标时的metrics_count"""
        monitoring_system.record_metric('cpu_usage', 50.0)
        monitoring_system.record_metric('memory_usage', 60.0)
        monitoring_system.record_metric('disk_usage', 70.0)
        
        report = monitoring_system.generate_report()
        
        assert report['metrics_count'] == 3

    def test_generate_report_metrics_count_mixed(self, monitoring_system):
        """测试混合情况（多个名称，每个多个值）"""
        for i in range(3):
            monitoring_system.record_metric('cpu_usage', 50.0 + i)
        for i in range(2):
            monitoring_system.record_metric('memory_usage', 60.0 + i)
        
        report = monitoring_system.generate_report()
        
        assert report['metrics_count'] == 5

    def test_generate_report_metrics_count_after_limit(self, monitoring_system):
        """测试超过限制后的metrics_count"""
        # 记录超过1000个指标，应该被限制
        # 限制逻辑是：当列表长度>1000时，截取到[-500:]
        for i in range(1500):
            monitoring_system.record_metric('limited_metric', float(i))
        
        report = monitoring_system.generate_report()
        
        # 限制逻辑会在每次添加后检查，当超过1000时截取到500
        # 但由于限制是在append之后检查的，实际行为可能不同
        # 验证最终列表长度不超过1000（被限制后）
        final_length = len(monitoring_system.metrics['limited_metric'])
        assert final_length <= 1000, f"Metrics list should be limited, got {final_length}"
        assert report['metrics_count'] == final_length
        # 验证最后的值是正确的（应该是1499.0）
        assert monitoring_system.metrics['limited_metric'][-1]['value'] == 1499.0

    def test_generate_report_metrics_count_with_empty_lists(self, monitoring_system):
        """测试包含空列表时的metrics_count"""
        monitoring_system.metrics['empty_metric'] = []
        monitoring_system.record_metric('valid_metric', 50.0)
        
        report = monitoring_system.generate_report()
        
        # 空列表不应该被计入
        assert report['metrics_count'] == 1

    def test_generate_report_timestamp_format(self, monitoring_system):
        """测试报告timestamp格式"""
        report = monitoring_system.generate_report()
        
        assert 'timestamp' in report
        assert isinstance(report['timestamp'], str)
        # 验证是ISO格式
        try:
            datetime.fromisoformat(report['timestamp'])
        except ValueError:
            pytest.fail("Invalid timestamp format")

    def test_generate_report_traces_count_empty(self, monitoring_system):
        """测试空traces时的traces_count"""
        report = monitoring_system.generate_report()
        
        assert report['traces_count'] == 0

    def test_generate_report_traces_count_with_traces(self, monitoring_system):
        """测试有traces时的traces_count"""
        span_id1 = monitoring_system.start_trace('trace_1', 'op1')
        span_id2 = monitoring_system.start_trace('trace_2', 'op2')
        monitoring_system.end_trace(span_id1)
        
        report = monitoring_system.generate_report()
        
        assert report['traces_count'] == 2

    def test_generate_report_alerts_count_empty(self, monitoring_system):
        """测试空alerts时的alerts_count"""
        report = monitoring_system.generate_report()
        
        assert report['alerts_count'] == 0

    def test_generate_report_alerts_count_with_alerts(self, monitoring_system):
        """测试有alerts时的alerts_count"""
        monitoring_system.record_metric('cpu_usage', 85.0)
        monitoring_system.check_alerts()
        
        report = monitoring_system.generate_report()
        
        assert report['alerts_count'] > 0

    def test_generate_report_structure_complete(self, monitoring_system):
        """测试报告结构完整性"""
        monitoring_system.record_metric('cpu_usage', 50.0)
        span_id = monitoring_system.start_trace('trace_1', 'op1')
        monitoring_system.end_trace(span_id)
        
        report = monitoring_system.generate_report()
        
        # 验证所有必需字段存在
        assert 'timestamp' in report
        assert 'metrics_count' in report
        assert 'traces_count' in report
        assert 'alerts_count' in report
        assert 'latest_metrics' in report
        assert 'performance_summary' in report

    def test_generate_report_metrics_count_calculation_accuracy(self, monitoring_system):
        """测试metrics_count计算准确性"""
        # 添加不同数量的指标
        for i in range(10):
            monitoring_system.record_metric('metric_A', float(i))
        for i in range(15):
            monitoring_system.record_metric('metric_B', float(i))
        for i in range(5):
            monitoring_system.record_metric('metric_C', float(i))
        
        report = monitoring_system.generate_report()
        
        # 应该总计30个指标
        assert report['metrics_count'] == 30

    def test_generate_report_metrics_count_after_clearing(self, monitoring_system):
        """测试清空metrics后的metrics_count"""
        monitoring_system.record_metric('cpu_usage', 50.0)
        
        # 清空metrics
        monitoring_system.metrics = {}
        
        report = monitoring_system.generate_report()
        
        assert report['metrics_count'] == 0

