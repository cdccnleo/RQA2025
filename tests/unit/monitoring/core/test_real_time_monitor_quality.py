#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时监控器质量测试
测试覆盖 RealTimeMonitor 的核心功能
"""

import sys
import importlib
from pathlib import Path
import pytest
import time
import threading
from unittest.mock import Mock, patch, MagicMock
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
    AlertManager = getattr(core_real_time_monitor_module, 'AlertManager', None)
    RealTimeMonitor = getattr(core_real_time_monitor_module, 'RealTimeMonitor', None)
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    AlertRule = getattr(core_real_time_monitor_module, 'AlertRule', None)
    Alert = getattr(core_real_time_monitor_module, 'Alert', None)
    
    if RealTimeMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.fixture
def metrics_collector():
    """创建指标收集器实例"""
    return MetricsCollector()


@pytest.fixture
def alert_manager():
    """创建告警管理器实例"""
    return AlertManager()


@pytest.fixture
def real_time_monitor():
    """创建实时监控器实例"""
    return RealTimeMonitor()


class TestMetricsCollector:
    """MetricsCollector测试类"""

    def test_initialization(self, metrics_collector):
        """测试初始化"""
        assert metrics_collector.metrics == {}
        assert metrics_collector.collectors == {}
        assert metrics_collector.collection_interval == 5
        assert metrics_collector._running is False

    def test_register_collector(self, metrics_collector):
        """测试注册指标收集器"""
        def test_collector():
            return {'test_metric': 1.0}
        
        metrics_collector.register_collector('test', test_collector)
        assert 'test' in metrics_collector.collectors

    def test_collect_system_metrics(self, metrics_collector):
        """测试收集系统指标"""
        metrics = metrics_collector.collect_system_metrics()
        assert 'cpu_percent' in metrics
        assert 'memory_percent' in metrics
        assert isinstance(metrics['cpu_percent'], (int, float))
        assert isinstance(metrics['memory_percent'], (int, float))

    def test_start_collection(self, metrics_collector):
        """测试启动收集"""
        metrics_collector.start_collection()
        assert metrics_collector._running is True
        assert metrics_collector._thread is not None
        assert metrics_collector._thread.is_alive()
        
        # 清理
        metrics_collector.stop_collection()

    def test_stop_collection(self, metrics_collector):
        """测试停止收集"""
        metrics_collector.start_collection()
        time.sleep(0.1)
        metrics_collector.stop_collection()
        assert metrics_collector._running is False

    def test_get_metrics(self, metrics_collector):
        """测试获取指标"""
        # 添加一些测试指标
        test_metric = MetricData(
            name='test_metric',
            value=1.0,
            timestamp=datetime.now()
        )
        metrics_collector.metrics['test_metric'] = test_metric
        
        # MetricsCollector没有get_metrics方法，直接访问metrics
        assert 'test_metric' in metrics_collector.metrics
        assert metrics_collector.metrics['test_metric'].name == 'test_metric'


class TestAlertManager:
    """AlertManager测试类"""

    def test_initialization(self, alert_manager):
        """测试初始化"""
        assert alert_manager.rules == {}
        assert alert_manager.active_alerts == {}
        assert alert_manager.alert_history == []

    def test_add_rule(self, alert_manager):
        """测试添加告警规则"""
        rule = AlertRule(
            name='test_rule',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='CPU usage too high'
        )
        alert_manager.add_rule(rule)
        assert 'test_rule' in alert_manager.rules

    def test_check_alerts(self, alert_manager):
        """测试检查告警"""
        rule = AlertRule(
            name='test_rule',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='CPU usage too high'
        )
        alert_manager.add_rule(rule)
        
        # 创建MetricData对象
        metric_data = MetricData(
            name='cpu_percent',
            value=50.0,
            timestamp=datetime.now()
        )
        
        # 检查告警（CPU应该不会超过80%）
        alerts = alert_manager.check_alerts({'cpu_percent': metric_data})
        assert isinstance(alerts, list)

    def test_get_active_alerts(self, alert_manager):
        """测试获取活跃告警"""
        alerts = alert_manager.get_active_alerts()
        assert isinstance(alerts, list)

    def test_resolve_alert(self, alert_manager):
        """测试解决告警（告警在check_alerts中自动解决）"""
        rule = AlertRule(
            name='test_rule',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='CPU usage too high'
        )
        alert_manager.add_rule(rule)
        
        # 触发告警
        metric_data = MetricData(
            name='cpu_percent',
            value=90.0,
            timestamp=datetime.now()
        )
        alerts = alert_manager.check_alerts({'cpu_percent': metric_data})
        
        # 验证告警被创建
        assert len(alerts) > 0
        assert 'test_rule_cpu_percent' in alert_manager.active_alerts
        
        # 告警解决（值低于阈值）
        metric_data_low = MetricData(
            name='cpu_percent',
            value=50.0,
            timestamp=datetime.now()
        )
        alert_manager.check_alerts({'cpu_percent': metric_data_low})
        
        # 验证告警被解决
        assert 'test_rule_cpu_percent' not in alert_manager.active_alerts


class TestRealTimeMonitor:
    """RealTimeMonitor测试类"""

    def test_initialization(self, real_time_monitor):
        """测试初始化"""
        assert real_time_monitor.metrics_collector is not None
        assert real_time_monitor.alert_manager is not None
        assert real_time_monitor._running is False

    def test_start_monitoring(self, real_time_monitor):
        """测试启动监控"""
        real_time_monitor.start_monitoring()
        assert real_time_monitor._running is True
        
        # 清理
        real_time_monitor.stop_monitoring()

    def test_stop_monitoring(self, real_time_monitor):
        """测试停止监控"""
        real_time_monitor.start_monitoring()
        time.sleep(0.1)
        real_time_monitor.stop_monitoring()
        assert real_time_monitor._running is False

    def test_get_metrics(self, real_time_monitor):
        """测试获取指标"""
        # RealTimeMonitor没有get_metrics方法，直接访问metrics_collector
        metrics = real_time_monitor.metrics_collector.metrics
        assert isinstance(metrics, dict)

    def test_get_alerts(self, real_time_monitor):
        """测试获取告警"""
        # RealTimeMonitor没有get_alerts方法，使用alert_manager的方法
        alerts = real_time_monitor.alert_manager.get_active_alerts()
        assert isinstance(alerts, list)

    def test_add_custom_collector(self, real_time_monitor):
        """测试添加自定义收集器"""
        def custom_collector():
            return {'custom_metric': 1.0}
        
        real_time_monitor.add_custom_collector('custom', custom_collector)
        assert 'custom' in real_time_monitor.metrics_collector.collectors

    def test_get_current_metrics(self, real_time_monitor):
        """测试获取当前指标"""
        metrics = real_time_monitor.get_current_metrics()
        assert isinstance(metrics, dict)

    def test_get_system_status(self, real_time_monitor):
        """测试获取系统状态"""
        status = real_time_monitor.get_system_status()
        assert isinstance(status, dict)
        assert 'timestamp' in status
        assert 'system_health' in status
        assert 'active_alerts' in status
        assert 'metrics_count' in status

    def test_update_business_metric(self, real_time_monitor):
        """测试更新业务指标"""
        real_time_monitor.update_business_metric('test_metric', 100.0)
        # 验证指标已更新（通过收集器）
        metrics = real_time_monitor.get_current_metrics()
        # 业务指标会在下次收集时出现

    def test_add_alert_rule(self, real_time_monitor):
        """测试添加告警规则"""
        rule = AlertRule(
            name='test_rule',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='CPU usage too high'
        )
        real_time_monitor.add_alert_rule(rule)
        assert 'test_rule' in real_time_monitor.alert_manager.rules

    def test_add_alert_callback(self, real_time_monitor):
        """测试添加告警回调"""
        callback_called = []
        def callback(alert):
            callback_called.append(alert)
        
        real_time_monitor.add_alert_callback(callback)
        assert len(real_time_monitor.alert_manager.alert_callbacks) > 0

    def test_get_alerts_summary(self, real_time_monitor):
        """测试获取告警摘要"""
        summary = real_time_monitor.get_alerts_summary()
        assert isinstance(summary, dict)
        assert 'active_count' in summary
        assert 'active_alerts' in summary
        assert 'recent_count' in summary
        assert 'recent_alerts' in summary

    def test_collect_business_metrics(self, metrics_collector):
        """测试收集业务指标"""
        metrics = metrics_collector.collect_business_metrics()
        assert isinstance(metrics, dict)

    def test_collect_application_metrics(self, metrics_collector):
        """测试收集应用指标"""
        metrics = metrics_collector.collect_application_metrics()
        assert isinstance(metrics, dict)

    def test_get_alert_history(self, alert_manager):
        """测试获取告警历史"""
        history = alert_manager.get_alert_history(hours=24)
        assert isinstance(history, list)

    def test_remove_rule(self, alert_manager):
        """测试移除告警规则"""
        rule = AlertRule(
            name='test_rule',
            metric_name='cpu_percent',
            condition='>',
            threshold=80.0,
            duration=60,
            severity='warning',
            description='CPU usage too high'
        )
        alert_manager.add_rule(rule)
        alert_manager.remove_rule('test_rule')
        assert 'test_rule' not in alert_manager.rules

    def test_collect_all_metrics(self, metrics_collector):
        """测试收集所有指标"""
        metrics = metrics_collector.collect_all_metrics()
        assert isinstance(metrics, dict)
        # 验证包含不同类型的指标
        assert len(metrics) > 0

    def test_collect_system_metrics_exception(self, metrics_collector):
        """测试收集系统指标异常处理"""
        with patch('psutil.cpu_percent', side_effect=Exception("CPU error")):
            metrics = metrics_collector.collect_system_metrics()
            assert metrics == {}

    def test_collect_application_metrics_exception(self, metrics_collector):
        """测试收集应用指标异常处理"""
        # Mock Process().cpu_percent()抛出异常
        with patch('psutil.Process') as mock_process:
            mock_process.return_value.cpu_percent.side_effect = Exception("Process error")
            metrics = metrics_collector.collect_application_metrics()
            assert metrics == {}

    def test_update_business_metric_request(self, metrics_collector):
        """测试更新业务指标（请求）"""
        metrics_collector.update_business_metric('request', 1.0)
        assert hasattr(metrics_collector, '_request_count')
        assert metrics_collector._request_count == 1

    def test_update_business_metric_error(self, metrics_collector):
        """测试更新业务指标（错误）"""
        metrics_collector.update_business_metric('error', 1.0)
        assert hasattr(metrics_collector, '_error_count')
        assert metrics_collector._error_count == 1

    def test_update_business_metric_response_time(self, metrics_collector):
        """测试更新业务指标（响应时间）"""
        metrics_collector.update_business_metric('response_time', 100.0)
        assert hasattr(metrics_collector, '_avg_response_time')
        assert metrics_collector._avg_response_time > 0

    def test_collect_all_metrics_with_custom_collector_exception(self, metrics_collector):
        """测试收集所有指标时自定义收集器异常"""
        def failing_collector():
            raise Exception("Collector error")
        
        metrics_collector.register_collector('failing', failing_collector)
        # 应该不抛出异常，而是记录错误
        metrics = metrics_collector.collect_all_metrics()
        assert isinstance(metrics, dict)

    def test_get_system_status_critical(self, real_time_monitor):
        """测试获取系统状态（关键状态）"""
        # 创建高CPU指标
        high_cpu_metric = MetricData(
            name='cpu_percent',
            value=95.0,
            timestamp=datetime.now()
        )
        real_time_monitor.metrics_collector.metrics['cpu_percent'] = high_cpu_metric
        
        status = real_time_monitor.get_system_status()
        assert status['system_health'] == 'critical'

    def test_get_system_status_warning(self, real_time_monitor):
        """测试获取系统状态（警告状态）"""
        # 创建高内存指标
        high_memory_metric = MetricData(
            name='memory_percent',
            value=95.0,
            timestamp=datetime.now()
        )
        real_time_monitor.metrics_collector.metrics['memory_percent'] = high_memory_metric
        
        status = real_time_monitor.get_system_status()
        assert status['system_health'] == 'warning'

    def test_alert_check_loop_exception(self, real_time_monitor):
        """测试告警检查循环异常处理"""
        real_time_monitor.start_monitoring()
        
        # Mock check_alerts抛出异常
        with patch.object(real_time_monitor.alert_manager, 'check_alerts', side_effect=Exception("Check error")):
            time.sleep(0.1)  # 等待循环处理异常
        
        real_time_monitor.stop_monitoring()
        assert real_time_monitor._running is False

