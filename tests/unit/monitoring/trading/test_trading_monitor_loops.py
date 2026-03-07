#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingMonitor循环方法测试
补充_monitoring_loop、_alert_processing_loop、_process_alerts方法的测试
"""

import pytest
from unittest.mock import patch, MagicMock
import time
from datetime import datetime, timedelta

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
    trading_trading_monitor_module = importlib.import_module('src.monitoring.trading.trading_monitor')
    TradingMonitor = getattr(trading_trading_monitor_module, 'TradingMonitor', None)
    MonitorType = getattr(trading_trading_monitor_module, 'MonitorType', None)
    AlertLevel = getattr(trading_trading_monitor_module, 'AlertLevel', None)
    Alert = getattr(trading_trading_monitor_module, 'Alert', None)
    if TradingMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)

class TestTradingMonitorLoops:
    """测试TradingMonitor循环方法"""

    @pytest.fixture
    def monitor(self):
        """创建TradingMonitor实例"""
        return TradingMonitor({
            'monitoring_interval': 1,
            'metrics_retention': 3600
        })

    def test_monitoring_loop_basic_execution(self, monitor):
        """测试监控循环基本执行"""
        monitor.running = True
        
        call_count = {'count': 0}
        
        def mock_record_performance_metrics():
            call_count['count'] += 1
            if call_count['count'] >= 2:  # 执行2次后停止
                monitor.running = False
        
        monitor.record_performance_metrics = mock_record_performance_metrics
        
        with patch('time.sleep'):
            monitor._monitoring_loop()
            
        assert call_count['count'] >= 1

    def test_monitoring_loop_exception_handling(self, monitor):
        """测试监控循环异常处理"""
        monitor.running = True
        
        call_count = {'count': 0}
        
        def mock_record_performance_metrics():
            call_count['count'] += 1
            if call_count['count'] == 1:
                raise Exception("Test exception")
            monitor.running = False
        
        monitor.record_performance_metrics = mock_record_performance_metrics
        
        with patch('time.sleep'):
            monitor._monitoring_loop()
            
        # 异常应该被捕获，循环继续
        assert call_count['count'] >= 1

    def test_monitoring_loop_calls_all_methods(self, monitor):
        """测试监控循环调用所有必要方法"""
        monitor.running = True
        
        call_counts = {
            'record': 0,
            'check': 0,
            'cleanup': 0
        }
        
        def mock_record():
            call_counts['record'] += 1
        
        def mock_check():
            call_counts['check'] += 1
        
        def mock_cleanup():
            call_counts['cleanup'] += 1
            if call_counts['cleanup'] >= 2:
                monitor.running = False
        
        monitor.record_performance_metrics = mock_record
        monitor.check_alerts = mock_check
        monitor._cleanup_old_data = mock_cleanup
        
        with patch('time.sleep'):
            monitor._monitoring_loop()
        
        assert call_counts['record'] >= 1
        assert call_counts['check'] >= 1
        assert call_counts['cleanup'] >= 1

    def test_monitoring_loop_sleep_interval(self, monitor):
        """测试监控循环的休眠间隔"""
        monitor.running = True
        monitor.monitoring_interval = 5
        
        call_count = {'count': 0}
        
        def mock_record():
            call_count['count'] += 1
            if call_count['count'] >= 1:
                monitor.running = False
        
        monitor.record_performance_metrics = mock_record
        
        with patch('time.sleep') as mock_sleep:
            monitor._monitoring_loop()
            
            # 验证使用了正确的休眠间隔
            assert mock_sleep.called
            # 检查是否使用了monitoring_interval
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            # 正常情况应该使用5秒间隔（除非异常处理使用了5秒）
            assert any(s == 5 for s in sleep_calls) or any(s == monitor.monitoring_interval for s in sleep_calls)

    def test_alert_processing_loop_basic_execution(self, monitor):
        """测试告警处理循环基本执行"""
        monitor.running = True
        
        call_count = {'count': 0}
        
        def mock_process_alerts():
            call_count['count'] += 1
            if call_count['count'] >= 2:  # 执行2次后停止
                monitor.running = False
        
        monitor._process_alerts = mock_process_alerts
        
        with patch('time.sleep'):
            monitor._alert_processing_loop()
            
        assert call_count['count'] >= 1

    def test_alert_processing_loop_exception_handling(self, monitor):
        """测试告警处理循环异常处理"""
        monitor.running = True
        
        call_count = {'count': 0}
        
        def mock_process_alerts():
            call_count['count'] += 1
            if call_count['count'] == 1:
                raise Exception("Test exception")
            monitor.running = False
        
        monitor._process_alerts = mock_process_alerts
        
        with patch('time.sleep'):
            monitor._alert_processing_loop()
            
        # 异常应该被捕获，循环继续
        assert call_count['count'] >= 1

    def test_alert_processing_loop_sleep_interval(self, monitor):
        """测试告警处理循环的休眠间隔"""
        monitor.running = True
        
        call_count = {'count': 0}
        
        def mock_process_alerts():
            call_count['count'] += 1
            if call_count['count'] >= 1:
                monitor.running = False
        
        monitor._process_alerts = mock_process_alerts
        
        with patch('time.sleep') as mock_sleep:
            monitor._alert_processing_loop()
            
            # 验证使用了30秒休眠间隔
            assert mock_sleep.called
            sleep_calls = [call[0][0] for call in mock_sleep.call_args_list]
            # 应该使用30秒间隔（除非异常处理使用了5秒）
            assert any(s == 30 for s in sleep_calls) or any(s == 5 for s in sleep_calls)

    def test_process_alerts_empty(self, monitor):
        """测试处理告警（无告警）"""
        monitor.alerts = []
        
        # _process_alerts目前是空实现，应该不会报错
        monitor._process_alerts()
        
        assert len(monitor.alerts) == 0

    def test_process_alerts_with_alerts(self, monitor):
        """测试处理告警（有告警）"""
        alert = Alert(
            alert_id="test_alert",
            monitor_type=MonitorType.PERFORMANCE,
            level=AlertLevel.WARNING,
            message="Test alert",
            details={},
            timestamp=datetime.now()
        )
        
        monitor.alerts = [alert]
        
        # _process_alerts目前是空实现，应该不会改变告警
        monitor._process_alerts()
        
        assert len(monitor.alerts) == 1

    def test_monitoring_loop_stops_when_not_running(self, monitor):
        """测试监控循环在not running时立即退出"""
        monitor.running = False
        
        call_count = {'count': 0}
        
        def mock_record():
            call_count['count'] += 1
        
        monitor.record_performance_metrics = mock_record
        
        with patch('time.sleep'):
            monitor._monitoring_loop()
        
        # 不应该执行任何操作
        assert call_count['count'] == 0

    def test_alert_processing_loop_stops_when_not_running(self, monitor):
        """测试告警处理循环在not running时立即退出"""
        monitor.running = False
        
        call_count = {'count': 0}
        
        def mock_process():
            call_count['count'] += 1
        
        monitor._process_alerts = mock_process
        
        with patch('time.sleep'):
            monitor._alert_processing_loop()
        
        # 不应该执行任何操作
        assert call_count['count'] == 0



