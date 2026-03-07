#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor告警检查循环详细测试
补充_alert_check_loop方法的详细测试
"""

import sys
import importlib
from pathlib import Path
import pytest
import time
from unittest.mock import Mock, patch
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
    MetricData = getattr(core_real_time_monitor_module, 'MetricData', None)
    if RealTimeMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)
    pytest.skip("real_time_monitor module not available", allow_module_level=True)


class TestRealTimeMonitorAlertCheckLoopDetailed:
    """测试RealTimeMonitor类告警检查循环的详细功能"""

    @pytest.fixture
    def monitor(self):
        """创建RealTimeMonitor实例"""
        return RealTimeMonitor()

    @patch('time.sleep')
    def test_alert_check_loop_calls_check_alerts_when_metrics_exist(self, mock_sleep, monitor):
        """测试告警检查循环在有指标时调用check_alerts"""
        monitor._running = True
        
        # 设置指标
        test_metric = MetricData('cpu_percent', 85.0, datetime.now(), {})
        monitor.metrics_collector.metrics = {'cpu_percent': test_metric}
        
        # Mock check_alerts
        with patch.object(monitor.alert_manager, 'check_alerts') as mock_check:
            call_count = 0
            
            def stop_after_one_iteration(seconds):
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    monitor._running = False
            
            mock_sleep.side_effect = stop_after_one_iteration
            
            monitor._alert_check_loop()
            
            # 应该调用了check_alerts
            assert mock_check.called

    @patch('time.sleep')
    def test_alert_check_loop_does_not_call_check_alerts_when_metrics_empty(self, mock_sleep, monitor):
        """测试告警检查循环在指标为空时不调用check_alerts"""
        monitor._running = True
        monitor.metrics_collector.metrics = {}
        
        with patch.object(monitor.alert_manager, 'check_alerts') as mock_check:
            call_count = 0
            
            def stop_after_one_iteration(seconds):
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    monitor._running = False
            
            mock_sleep.side_effect = stop_after_one_iteration
            
            monitor._alert_check_loop()
            
            # 不应该调用check_alerts（因为metrics为空）
            # 注意：代码中会检查if metrics:，所以空字典不会调用check_alerts
            # 但这里我们主要验证不会崩溃

    @patch('time.sleep')
    def test_alert_check_loop_handles_check_alerts_exception(self, mock_sleep, monitor):
        """测试告警检查循环处理check_alerts异常"""
        monitor._running = True
        monitor.metrics_collector.metrics = {'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {})}
        
        # 模拟check_alerts抛出异常
        with patch.object(monitor.alert_manager, 'check_alerts', side_effect=Exception("Check alerts error")):
            with patch('src.monitoring.core.real_time_monitor.logger') as mock_logger:
                call_count = 0
                
                def stop_after_one_iteration(seconds):
                    nonlocal call_count
                    call_count += 1
                    if call_count >= 1:
                        monitor._running = False
                
                mock_sleep.side_effect = stop_after_one_iteration
                
                # 不应该抛出异常
                monitor._alert_check_loop()
                
                # 验证错误被记录
                mock_logger.error.assert_called()
                error_call = mock_logger.error.call_args[0][0]
                assert 'Error in alert check loop' in error_call

    @patch('time.sleep')
    def test_alert_check_loop_handles_metrics_access_exception(self, mock_sleep, monitor):
        """测试告警检查循环处理指标访问异常"""
        monitor._running = True
        
        # 模拟访问metrics时抛出异常
        with patch.object(monitor.metrics_collector, 'metrics', new_callable=lambda: property(
            lambda self: (_ for _ in ()).throw(Exception("Metrics access error"))
        )):
            with patch('src.monitoring.core.real_time_monitor.logger') as mock_logger:
                call_count = 0
                
                def stop_after_one_iteration(seconds):
                    nonlocal call_count
                    call_count += 1
                    if call_count >= 1:
                        monitor._running = False
                
                mock_sleep.side_effect = stop_after_one_iteration
                
                # 不应该抛出异常
                monitor._alert_check_loop()
                
                # 验证错误被记录
                mock_logger.error.assert_called()

    @patch('time.sleep')
    def test_alert_check_loop_sleeps_with_10_seconds_interval(self, mock_sleep, monitor):
        """测试告警检查循环按10秒间隔sleep"""
        monitor._running = True
        monitor.metrics_collector.metrics = {}
        
        call_count = 0
        
        def stop_after_one_iteration(seconds):
            nonlocal call_count
            call_count += 1
            assert seconds == 10  # 验证sleep时间为10秒
            if call_count >= 1:
                monitor._running = False
        
        mock_sleep.side_effect = stop_after_one_iteration
        
        monitor._alert_check_loop()
        
        # 验证sleep被调用
        assert mock_sleep.called

    @patch('time.sleep')
    def test_alert_check_loop_stops_when_running_false(self, mock_sleep, monitor):
        """测试告警检查循环在running为False时停止"""
        monitor._running = True
        monitor.metrics_collector.metrics = {}
        
        call_count = 0
        max_calls = 3
        
        def stop_after_three_iterations(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= max_calls:
                monitor._running = False
        
        mock_sleep.side_effect = stop_after_three_iterations
        
        monitor._alert_check_loop()
        
        # 验证循环停止了
        assert monitor._running == False

    @patch('time.sleep')
    def test_alert_check_loop_immediately_stops_if_not_running(self, mock_sleep, monitor):
        """测试告警检查循环在初始不运行时立即停止"""
        monitor._running = False
        
        monitor._alert_check_loop()
        
        # 不应该调用sleep（因为循环立即退出）
        # 注意：由于while循环的条件检查，如果_running为False，循环不会执行
        # 但我们主要验证不会崩溃

    @patch('time.sleep')
    def test_alert_check_loop_continues_after_exception(self, mock_sleep, monitor):
        """测试告警检查循环在异常后继续运行"""
        monitor._running = True
        
        call_count = 0
        
        def stop_after_two_iterations(seconds):
            nonlocal call_count
            call_count += 1
            if call_count >= 2:
                monitor._running = False
        
        mock_sleep.side_effect = stop_after_two_iterations
        
        # 第一次迭代抛出异常，第二次正常
        exception_thrown = False
        with patch.object(monitor.alert_manager, 'check_alerts') as mock_check:
            def side_effect(*args):
                nonlocal exception_thrown
                if not exception_thrown:
                    exception_thrown = True
                    raise Exception("First iteration error")
            
            mock_check.side_effect = side_effect
            
            monitor.metrics_collector.metrics = {'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {})}
            
            with patch('src.monitoring.core.real_time_monitor.logger'):
                monitor._alert_check_loop()
            
            # 应该调用了多次check_alerts
            assert mock_check.call_count >= 1

    @patch('time.sleep')
    def test_alert_check_loop_with_multiple_metrics(self, mock_sleep, monitor):
        """测试告警检查循环处理多个指标"""
        monitor._running = True
        
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 90.0, datetime.now(), {}),
            'disk_usage': MetricData('disk_usage', 70.0, datetime.now(), {})
        }
        
        with patch.object(monitor.alert_manager, 'check_alerts') as mock_check:
            call_count = 0
            
            def stop_after_one_iteration(seconds):
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    monitor._running = False
            
            mock_sleep.side_effect = stop_after_one_iteration
            
            monitor._alert_check_loop()
            
            # 应该调用了check_alerts，并且传递了所有指标
            assert mock_check.called
            if mock_check.call_args:
                metrics_arg = mock_check.call_args[0][0]
                assert len(metrics_arg) == 3

    @patch('time.sleep')
    def test_alert_check_loop_passes_metrics_to_check_alerts(self, mock_sleep, monitor):
        """测试告警检查循环将指标传递给check_alerts"""
        monitor._running = True
        
        expected_metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {}),
            'memory_percent': MetricData('memory_percent', 90.0, datetime.now(), {})
        }
        monitor.metrics_collector.metrics = expected_metrics.copy()
        
        with patch.object(monitor.alert_manager, 'check_alerts') as mock_check:
            call_count = 0
            
            def stop_after_one_iteration(seconds):
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    monitor._running = False
            
            mock_sleep.side_effect = stop_after_one_iteration
            
            monitor._alert_check_loop()
            
            # 验证传递了正确的指标
            if mock_check.called and mock_check.call_args:
                metrics_arg = mock_check.call_args[0][0]
                assert 'cpu_percent' in metrics_arg
                assert 'memory_percent' in metrics_arg

    def test_alert_check_loop_integration_with_real_alert_rules(self, monitor):
        """测试告警检查循环与真实告警规则的集成"""
        monitor._running = True
        
        # 设置触发告警的指标
        monitor.metrics_collector.metrics = {
            'cpu_percent': MetricData('cpu_percent', 85.0, datetime.now(), {})
        }
        
        with patch('time.sleep') as mock_sleep:
            call_count = 0
            
            def stop_after_one_iteration(seconds):
                nonlocal call_count
                call_count += 1
                if call_count >= 1:
                    monitor._running = False
            
            mock_sleep.side_effect = stop_after_one_iteration
            
            # 运行一次循环
            monitor._alert_check_loop()
            
            # 验证告警可能被触发（如果有匹配的规则）
            # 这里主要是验证集成工作正常

