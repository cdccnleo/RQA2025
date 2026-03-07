#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FullLinkMonitor性能报告测试
补充full_link_monitor.py中get_performance_report方法的全面测试
"""

import sys
import importlib
from pathlib import Path
import pytest
import numpy as np
from unittest.mock import Mock, patch
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
    engine_full_link_monitor_module = importlib.import_module('src.monitoring.engine.full_link_monitor')
    FullLinkMonitor = getattr(engine_full_link_monitor_module, 'FullLinkMonitor', None)
    AlertLevel = getattr(engine_full_link_monitor_module, 'AlertLevel', None)
    MonitorType = getattr(engine_full_link_monitor_module, 'MonitorType', None)
    MetricData = getattr(engine_full_link_monitor_module, 'MetricData', None)
    AlertRule = getattr(engine_full_link_monitor_module, 'AlertRule', None)
    Alert = getattr(engine_full_link_monitor_module, 'Alert', None)
    PerformanceMetrics = getattr(engine_full_link_monitor_module, 'PerformanceMetrics', None)
    
    if FullLinkMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestFullLinkMonitorPerformanceReport:
    """测试FullLinkMonitor性能报告功能"""

    @pytest.fixture
    def monitor(self):
        """创建monitor实例"""
        return FullLinkMonitor()

    @pytest.fixture
    def monitor_with_metrics(self, monitor):
        """准备有性能指标的monitor"""
        # 添加不同时间段的性能指标
        now = datetime.now()
        for i in range(10):
            metrics = PerformanceMetrics(
                cpu_usage=50.0 + i * 2,
                memory_usage=60.0 + i * 1.5,
                disk_usage=70.0 + i * 1.0,
                network_io={'sent': 1000 + i * 100, 'received': 2000 + i * 200},
                response_time=100.0 + i * 5,
                throughput=1000.0 + i * 50,
                error_rate=0.01 + i * 0.002,
                timestamp=now - timedelta(hours=i)
            )
            monitor.performance_metrics.append(metrics)
        return monitor

    def test_get_performance_report_empty_metrics(self, monitor):
        """测试没有性能指标时的报告"""
        report = monitor.get_performance_report(hours=24)
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert report['summary']['total_metrics'] == 0
        assert report['summary']['total_alerts'] == len(monitor.alert_history)
        assert 'period' in report['summary']
        assert report['summary']['period']['hours'] == 24

    def test_get_performance_report_with_metrics(self, monitor_with_metrics):
        """测试有性能指标时的报告"""
        report = monitor_with_metrics.get_performance_report(hours=24)
        
        assert isinstance(report, dict)
        assert 'summary' in report
        assert 'system_metrics' in report
        assert 'application_metrics' in report
        assert 'alerts' in report
        assert report['summary']['total_metrics'] > 0

    def test_get_performance_report_custom_hours(self, monitor_with_metrics):
        """测试自定义时间窗口"""
        report = monitor_with_metrics.get_performance_report(hours=12)
        
        assert isinstance(report, dict)
        assert report['summary']['period']['hours'] == 12
        # 应该只包含12小时内的数据
        assert report['summary']['total_metrics'] <= len(monitor_with_metrics.performance_metrics)

    def test_get_performance_report_one_hour(self, monitor_with_metrics):
        """测试1小时窗口"""
        report = monitor_with_metrics.get_performance_report(hours=1)
        
        assert isinstance(report, dict)
        assert report['summary']['period']['hours'] == 1
        # 应该只包含1小时内的数据
        cutoff_time = datetime.now() - timedelta(hours=1)
        recent_count = sum(1 for m in monitor_with_metrics.performance_metrics 
                          if m.timestamp >= cutoff_time)
        assert report['summary']['total_metrics'] == recent_count

    def test_get_performance_report_long_hours(self, monitor_with_metrics):
        """测试长时间窗口"""
        report = monitor_with_metrics.get_performance_report(hours=48)
        
        assert isinstance(report, dict)
        assert report['summary']['period']['hours'] == 48
        # 应该包含所有数据
        assert report['summary']['total_metrics'] == len(monitor_with_metrics.performance_metrics)

    def test_get_performance_report_system_metrics_structure(self, monitor_with_metrics):
        """测试系统指标结构"""
        report = monitor_with_metrics.get_performance_report()
        
        assert 'system_metrics' in report
        system_metrics = report['system_metrics']
        assert 'cpu_usage' in system_metrics
        assert 'memory_usage' in system_metrics
        assert 'disk_usage' in system_metrics
        
        # 验证每个指标都有avg, max, min
        for metric_name in ['cpu_usage', 'memory_usage', 'disk_usage']:
            assert 'avg' in system_metrics[metric_name]
            assert 'max' in system_metrics[metric_name]
            assert 'min' in system_metrics[metric_name]
            assert isinstance(system_metrics[metric_name]['avg'], (int, float))
            assert isinstance(system_metrics[metric_name]['max'], (int, float))
            assert isinstance(system_metrics[metric_name]['min'], (int, float))

    def test_get_performance_report_application_metrics_structure(self, monitor_with_metrics):
        """测试应用指标结构"""
        report = monitor_with_metrics.get_performance_report()
        
        assert 'application_metrics' in report
        app_metrics = report['application_metrics']
        assert 'response_time' in app_metrics
        assert 'throughput' in app_metrics
        assert 'error_rate' in app_metrics
        
        # 验证每个指标都有avg, max, min
        for metric_name in ['response_time', 'throughput', 'error_rate']:
            assert 'avg' in app_metrics[metric_name]
            assert 'max' in app_metrics[metric_name]
            assert 'min' in app_metrics[metric_name]
            assert isinstance(app_metrics[metric_name]['avg'], (int, float))
            assert isinstance(app_metrics[metric_name]['max'], (int, float))
            assert isinstance(app_metrics[metric_name]['min'], (int, float))

    def test_get_performance_report_alerts_structure(self, monitor_with_metrics):
        """测试告警结构"""
        report = monitor_with_metrics.get_performance_report()
        
        assert 'alerts' in report
        alerts = report['alerts']
        assert 'active_count' in alerts
        assert 'total_count' in alerts
        assert 'resolved_count' in alerts
        assert isinstance(alerts['active_count'], int)
        assert isinstance(alerts['total_count'], int)
        assert isinstance(alerts['resolved_count'], int)

    def test_get_performance_report_summary_structure(self, monitor_with_metrics):
        """测试摘要结构"""
        report = monitor_with_metrics.get_performance_report()
        
        assert 'summary' in report
        summary = report['summary']
        assert 'period' in summary
        assert 'total_metrics' in summary
        assert 'total_alerts' in summary
        
        period = summary['period']
        assert 'start' in period
        assert 'end' in period
        assert 'hours' in period
        assert isinstance(period['start'], str)
        assert isinstance(period['end'], str)
        assert isinstance(period['hours'], int)

    def test_get_performance_report_calculations(self, monitor):
        """测试性能报告计算准确性"""
        # 添加已知值的性能指标
        metrics_list = [
            PerformanceMetrics(
                cpu_usage=50.0,
                memory_usage=60.0,
                disk_usage=70.0,
                network_io={'sent': 1000, 'received': 2000},
                response_time=100.0,
                throughput=1000.0,
                error_rate=0.01,
                timestamp=datetime.now()
            ),
            PerformanceMetrics(
                cpu_usage=60.0,
                memory_usage=70.0,
                disk_usage=80.0,
                network_io={'sent': 2000, 'received': 3000},
                response_time=200.0,
                throughput=2000.0,
                error_rate=0.02,
                timestamp=datetime.now()
            )
        ]
        monitor.performance_metrics = metrics_list
        
        report = monitor.get_performance_report(hours=24)
        
        # 验证平均值计算
        assert report['system_metrics']['cpu_usage']['avg'] == 55.0
        assert report['system_metrics']['memory_usage']['avg'] == 65.0
        assert report['application_metrics']['response_time']['avg'] == 150.0
        
        # 验证最大值计算
        assert report['system_metrics']['cpu_usage']['max'] == 60.0
        assert report['application_metrics']['response_time']['max'] == 200.0
        
        # 验证最小值计算
        assert report['system_metrics']['cpu_usage']['min'] == 50.0
        assert report['application_metrics']['response_time']['min'] == 100.0

    def test_get_performance_report_with_alerts(self, monitor_with_metrics):
        """测试包含告警的报告"""
        # 添加一些告警
        alert1 = Alert(
            id='alert1',
            rule_name='test_rule1',
            metric_name='cpu_usage',
            current_value=90.0,
            threshold='> 80',
            level=AlertLevel.WARNING,
            timestamp=datetime.now(),
            message='Test alert 1'
        )
        alert2 = Alert(
            id='alert2',
            rule_name='test_rule2',
            metric_name='memory_usage',
            current_value=95.0,
            threshold='> 85',
            level=AlertLevel.ERROR,
            timestamp=datetime.now(),
            message='Test alert 2',
            resolved=True
        )
        
        monitor_with_metrics.active_alerts['test_rule1'] = alert1
        monitor_with_metrics.alert_history.append(alert1)
        monitor_with_metrics.alert_history.append(alert2)
        
        report = monitor_with_metrics.get_performance_report()
        
        assert report['alerts']['active_count'] == 1
        assert report['alerts']['total_count'] == 2
        assert report['alerts']['resolved_count'] == 1

    def test_get_performance_report_time_filtering(self, monitor):
        """测试时间过滤"""
        now = datetime.now()
        # 添加不同时间的指标
        old_metric = PerformanceMetrics(
            cpu_usage=50.0,
            memory_usage=60.0,
            disk_usage=70.0,
            network_io={'sent': 1000, 'received': 2000},
            response_time=100.0,
            throughput=1000.0,
            error_rate=0.01,
            timestamp=now - timedelta(hours=30)  # 30小时前
        )
        recent_metric = PerformanceMetrics(
            cpu_usage=60.0,
            memory_usage=70.0,
            disk_usage=80.0,
            network_io={'sent': 2000, 'received': 3000},
            response_time=200.0,
            throughput=2000.0,
            error_rate=0.02,
            timestamp=now - timedelta(hours=10)  # 10小时前
        )
        
        monitor.performance_metrics.append(old_metric)
        monitor.performance_metrics.append(recent_metric)
        
        report = monitor.get_performance_report(hours=24)
        
        # 应该只包含10小时前的指标（在24小时内）
        assert report['summary']['total_metrics'] == 1
        assert report['system_metrics']['cpu_usage']['avg'] == 60.0

    def test_get_performance_report_zero_hours(self, monitor_with_metrics):
        """测试0小时窗口（边界情况）"""
        report = monitor_with_metrics.get_performance_report(hours=0)
        
        assert isinstance(report, dict)
        assert report['summary']['period']['hours'] == 0
        # 0小时窗口应该只包含当前时刻的数据（如果有）
        assert report['summary']['total_metrics'] >= 0

    def test_get_performance_report_single_metric(self, monitor):
        """测试只有单个指标时的报告"""
        single_metric = PerformanceMetrics(
            cpu_usage=75.0,
            memory_usage=85.0,
            disk_usage=90.0,
            network_io={'sent': 5000, 'received': 10000},
            response_time=500.0,
            throughput=5000.0,
            error_rate=0.05,
            timestamp=datetime.now()
        )
        monitor.performance_metrics.append(single_metric)
        
        report = monitor.get_performance_report()
        
        assert report['summary']['total_metrics'] == 1
        # 单个指标时，avg, max, min应该相同
        assert report['system_metrics']['cpu_usage']['avg'] == 75.0
        assert report['system_metrics']['cpu_usage']['max'] == 75.0
        assert report['system_metrics']['cpu_usage']['min'] == 75.0

    def test_get_performance_report_period_timestamps(self, monitor_with_metrics):
        """测试时间段时间戳"""
        report = monitor_with_metrics.get_performance_report(hours=12)
        
        # 验证时间戳格式
        period = report['summary']['period']
        assert isinstance(period['start'], str)
        assert isinstance(period['end'], str)
        
        # 验证时间戳可以解析
        from datetime import datetime
        start_time = datetime.fromisoformat(period['start'])
        end_time = datetime.fromisoformat(period['end'])
        
        assert isinstance(start_time, datetime)
        assert isinstance(end_time, datetime)
        assert end_time >= start_time



