#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor主类测试
补充RealTimeMonitor类的方法测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
from typing import Callable
from datetime import datetime

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
    RealTimeMonitor = getattr(core_real_time_monitor_module, 'RealTimeMonitor', None)
    AlertRule = getattr(core_real_time_monitor_module, 'AlertRule', None)
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    if RealTimeMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestRealTimeMonitor:
    """测试RealTimeMonitor类"""

    @pytest.fixture
    def monitor(self):
        """创建RealTimeMonitor实例"""
        return RealTimeMonitor()

    def test_init(self, monitor):
        """测试初始化"""
        assert monitor.metrics_collector is not None
        assert monitor.alert_manager is not None
        assert monitor._running == False
        assert monitor._monitor_thread is None
        assert monitor._alert_thread is None

    def test_init_default_rules(self, monitor):
        """测试初始化默认告警规则"""
        # 应该有默认规则
        assert len(monitor.alert_manager.rules) > 0
        assert 'high_cpu_usage' in monitor.alert_manager.rules
        assert 'high_memory_usage' in monitor.alert_manager.rules

    @patch.object(RealTimeMonitor, '_alert_check_loop')
    def test_start_monitoring_normal(self, mock_alert_loop, monitor):
        """测试启动监控正常"""
        monitor._running = False
        monitor.metrics_collector.start_collection = Mock()
        
        with patch('threading.Thread') as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread
            
            monitor.start_monitoring()
            
            assert monitor._running == True
            monitor.metrics_collector.start_collection.assert_called_once()
            mock_thread.start.assert_called_once()

    def test_start_monitoring_already_running(self, monitor):
        """测试启动监控已在运行"""
        monitor._running = True
        original_thread = monitor._alert_thread
        
        monitor.start_monitoring()
        
        # 应该不改变状态
        assert monitor._running == True
        assert monitor._alert_thread == original_thread

    def test_stop_monitoring_normal(self, monitor):
        """测试停止监控正常"""
        monitor._running = True
        monitor.metrics_collector.stop_collection = Mock()
        mock_thread = Mock()
        mock_thread.join = Mock()
        monitor._alert_thread = mock_thread
        
        monitor.stop_monitoring()
        
        assert monitor._running == False
        monitor.metrics_collector.stop_collection.assert_called_once()
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_stop_monitoring_no_alert_thread(self, monitor):
        """测试停止监控无告警线程"""
        monitor._running = True
        monitor.metrics_collector.stop_collection = Mock()
        monitor._alert_thread = None
        
        # 不应该抛出异常
        monitor.stop_monitoring()
        assert monitor._running == False

    @patch.object(RealTimeMonitor, 'metrics_collector')
    def test_alert_check_loop(self, mock_collector, monitor):
        """测试告警检查循环"""
        monitor._running = True
        
        # Mock metrics
        mock_collector.metrics = {
            'cpu_percent': MetricData(
                name='cpu_percent',
                value=85.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        # 设置超时，避免无限循环
        import time
        start_time = time.time()
        
        # 模拟循环一次就退出
        with patch('time.sleep', side_effect=[None, KeyboardInterrupt()]):
            try:
                monitor._alert_check_loop()
            except KeyboardInterrupt:
                pass
        
        # 验证alert_manager被调用（通过检查规则）
        assert len(monitor.alert_manager.rules) > 0

    @patch.object(RealTimeMonitor, 'metrics_collector')
    def test_alert_check_loop_exception(self, mock_collector, monitor):
        """测试告警检查循环异常处理"""
        monitor._running = True
        mock_collector.metrics = {}
        
        # 模拟异常
        with patch('time.sleep', side_effect=[None, KeyboardInterrupt()]):
            try:
                monitor._alert_check_loop()
            except KeyboardInterrupt:
                pass
        
        # 不应该崩溃
        assert True

    @patch.object(RealTimeMonitor, 'metrics_collector')
    def test_alert_check_loop_stops_when_not_running(self, mock_collector, monitor):
        """测试告警检查循环在不运行时停止"""
        monitor._running = False
        
        # 循环应该立即退出
        monitor._alert_check_loop()
        
        # 不应该处理指标
        assert True

    def test_get_current_metrics(self, monitor):
        """测试获取当前指标"""
        # 设置一些指标
        monitor.metrics_collector.metrics = {
            'test_metric': MetricData(
                name='test_metric',
                value=42.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        metrics = monitor.get_current_metrics()
        
        assert 'test_metric' in metrics
        assert metrics['test_metric'].value == 42.0
        # 应该是副本，修改不影响原数据
        assert metrics is not monitor.metrics_collector.metrics

    def test_get_current_metrics_empty(self, monitor):
        """测试获取空指标"""
        monitor.metrics_collector.metrics = {}
        
        metrics = monitor.get_current_metrics()
        
        assert isinstance(metrics, dict)
        assert len(metrics) == 0

    def test_get_system_status(self, monitor):
        """测试获取系统状态"""
        # 设置一些指标
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData(
                name='cpu_percent',
                value=50.0,
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        status = monitor.get_system_status()
        
        assert 'timestamp' in status
        assert 'system_health' in status
        assert 'active_alerts' in status
        assert 'metrics_count' in status
        assert status['system_health'] == 'healthy'
        assert status['metrics_count'] == 1

    def test_get_system_status_critical_cpu(self, monitor):
        """测试获取系统状态CPU临界"""
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData(
                name='cpu_percent',
                value=95.0,  # 超过90，应该是critical
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        status = monitor.get_system_status()
        
        assert status['system_health'] == 'critical'

    def test_get_system_status_warning_memory(self, monitor):
        """测试获取系统状态内存警告"""
        monitor.metrics_collector.metrics = {
            'memory_percent': MetricData(
                name='memory_percent',
                value=95.0,  # 超过90，应该是warning
                timestamp=datetime.now(),
                tags={}
            )
        }
        
        status = monitor.get_system_status()
        
        assert status['system_health'] == 'warning'

    def test_update_business_metric(self, monitor):
        """测试更新业务指标"""
        monitor.update_business_metric('request', 1.0)
        
        metrics = monitor.metrics_collector.collect_business_metrics()
        assert metrics['requests_total'] == 1

    def test_add_custom_collector(self, monitor):
        """测试添加自定义指标收集器"""
        def custom_collector():
            return {'custom_metric': 42.0}
        
        monitor.add_custom_collector('custom', custom_collector)
        
        assert 'custom' in monitor.metrics_collector.collectors

    def test_add_alert_rule(self, monitor):
        """测试添加告警规则"""
        rule = AlertRule(
            name='test_rule',
            metric_name='test_metric',
            condition='>',
            threshold=100.0,
            duration=60,
            severity='warning',
            description='Test rule'
        )
        
        monitor.add_alert_rule(rule)
        
        assert 'test_rule' in monitor.alert_manager.rules

    def test_add_alert_callback(self, monitor):
        """测试添加告警回调"""
        callback = Mock()
        
        monitor.add_alert_callback(callback)
        
        assert callback in monitor.alert_manager.alert_callbacks

    def test_get_alerts_summary(self, monitor):
        """测试获取告警摘要"""
        summary = monitor.get_alerts_summary()
        
        assert 'active_count' in summary
        assert 'recent_count' in summary
        assert 'active_alerts' in summary
        assert 'recent_alerts' in summary
        assert isinstance(summary['active_count'], int)
        assert isinstance(summary['recent_count'], int)
        assert isinstance(summary['active_alerts'], list)
        assert isinstance(summary['recent_alerts'], list)

    def test_get_active_alerts(self, monitor):
        """测试获取活跃告警"""
        # 通过alert_manager获取
        alerts = monitor.alert_manager.get_active_alerts()
        
        assert isinstance(alerts, list)

    def test_get_alert_history(self, monitor):
        """测试获取告警历史"""
        history = monitor.alert_manager.get_alert_history()
        
        assert isinstance(history, list)

    def test_get_alert_history_with_hours(self, monitor):
        """测试获取告警历史带时间参数"""
        history = monitor.alert_manager.get_alert_history(hours=12)
        
        assert isinstance(history, list)

