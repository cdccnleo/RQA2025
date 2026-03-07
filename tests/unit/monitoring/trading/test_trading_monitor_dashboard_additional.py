#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingMonitorDashboard附加方法测试
补充start_server、run_in_background、add_status_callback、_trigger_status_callbacks、get_dashboard_summary等方法的测试
"""

import pytest
import threading
import time
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

try:
    from flask import Flask
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

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

# 直接导入模块
try:
    from src.monitoring.trading.trading_monitor_dashboard import TradingMonitorDashboard, TradingStatus
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


@pytest.mark.skipif(not FLASK_AVAILABLE, reason="Flask not available")
class TestTradingMonitorDashboardAdditional:
    """测试TradingMonitorDashboard附加方法"""

    @pytest.fixture
    def dashboard(self):
        """创建dashboard实例"""
        return TradingMonitorDashboard({'update_interval': 0.1})

    def test_start_server_no_app(self, dashboard):
        """测试在没有app时启动服务器"""
        dashboard.app = None
        with patch('src.monitoring.trading.trading_monitor_dashboard.logger') as mock_logger:
            dashboard.start_server()
            mock_logger.error.assert_called_once()

    def test_start_server_with_app(self, dashboard):
        """测试启动服务器"""
        if not dashboard.app:
            pytest.skip("Flask app not available")
        
        with patch.object(dashboard.app, 'run') as mock_run:
            with patch('src.monitoring.trading.trading_monitor_dashboard.logger') as mock_logger:
                dashboard.start_server(host='localhost', port=5002, debug=False)
                mock_run.assert_called_once_with(
                    host='localhost',
                    port=5002,
                    debug=False,
                    threaded=True
                )
                assert mock_logger.info.called

    def test_start_server_with_exception(self, dashboard):
        """测试启动服务器时发生异常"""
        if not dashboard.app:
            pytest.skip("Flask app not available")
        
        with patch.object(dashboard.app, 'run', side_effect=Exception("Server error")):
            with patch('src.monitoring.trading.trading_monitor_dashboard.logger') as mock_logger:
                dashboard.start_server()
                mock_logger.error.assert_called_once()

    def test_run_in_background(self, dashboard):
        """测试在后台运行服务器"""
        with patch.object(dashboard, 'start_server') as mock_start_server:
            with patch('threading.Thread') as mock_thread_class:
                mock_thread = MagicMock()
                mock_thread_class.return_value = mock_thread
                
                with patch('src.monitoring.trading.trading_monitor_dashboard.logger'):
                    dashboard.run_in_background(host='localhost', port=5002, debug=False)
                    
                    mock_thread_class.assert_called_once()
                    mock_thread.start.assert_called_once()

    def test_run_in_background_thread_args(self, dashboard):
        """测试后台运行服务器的线程参数"""
        with patch.object(dashboard, 'start_server') as mock_start_server:
            with patch('threading.Thread') as mock_thread_class:
                mock_thread = MagicMock()
                mock_thread_class.return_value = mock_thread
                
                dashboard.run_in_background(host='testhost', port=9999, debug=True)
                
                call_args = mock_thread_class.call_args
                assert call_args[0][0] == dashboard.start_server
                assert call_args[1]['args'] == ('testhost', 9999, True)
                assert call_args[1]['daemon'] is True

    def test_add_status_callback(self, dashboard):
        """测试添加状态回调"""
        callback1 = Mock()
        callback2 = Mock()
        
        dashboard.add_status_callback(callback1)
        dashboard.add_status_callback(callback2)
        
        assert len(dashboard.status_callbacks) == 2
        assert callback1 in dashboard.status_callbacks
        assert callback2 in dashboard.status_callbacks

    def test_trigger_status_callbacks_success(self, dashboard):
        """测试触发状态回调成功"""
        callback1 = Mock()
        callback2 = Mock()
        
        dashboard.status_callbacks = [callback1, callback2]
        dashboard._collect_trading_status()
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.logger'):
            dashboard._trigger_status_callbacks()
            
            callback1.assert_called_once_with(dashboard.current_status)
            callback2.assert_called_once_with(dashboard.current_status)

    def test_trigger_status_callbacks_with_exception(self, dashboard):
        """测试状态回调执行失败"""
        callback1 = Mock(side_effect=Exception("Callback error"))
        callback2 = Mock()
        
        dashboard.status_callbacks = [callback1, callback2]
        dashboard._collect_trading_status()
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.logger') as mock_logger:
            dashboard._trigger_status_callbacks()
            
            # 即使callback1失败，callback2仍应该被调用
            callback1.assert_called_once()
            callback2.assert_called_once()
            mock_logger.error.assert_called_once()

    def test_trigger_status_callbacks_empty(self, dashboard):
        """测试空回调列表"""
        dashboard.status_callbacks = []
        dashboard._collect_trading_status()
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.logger'):
            # 应该不抛出异常
            dashboard._trigger_status_callbacks()

    def test_get_dashboard_summary(self, dashboard):
        """测试获取仪表板汇总信息"""
        dashboard._collect_trading_status()
        
        summary = dashboard.get_dashboard_summary()
        
        assert isinstance(summary, dict)
        assert 'current_status' in summary
        assert 'health_score' in summary
        assert 'active_alerts' in summary
        assert 'total_positions' in summary
        assert 'connected_exchanges' in summary
        assert 'last_update' in summary

    def test_get_dashboard_summary_with_alerts(self, dashboard):
        """测试带告警的仪表板汇总"""
        dashboard._collect_trading_status()
        dashboard.current_status.alerts = [
            {'level': 'warning', 'message': 'Test alert 1'},
            {'level': 'error', 'message': 'Test alert 2'}
        ]
        
        summary = dashboard.get_dashboard_summary()
        
        assert summary['active_alerts'] == 2

    def test_get_dashboard_summary_no_positions(self, dashboard):
        """测试无持仓时的仪表板汇总"""
        dashboard._collect_trading_status()
        dashboard.current_status.positions = {}
        
        summary = dashboard.get_dashboard_summary()
        
        assert summary['total_positions'] == 0

    def test_get_dashboard_summary_connected_exchanges(self, dashboard):
        """测试连接交易所计数"""
        dashboard._collect_trading_status()
        dashboard.current_status.connections = {
            'NYSE': {'status': 'connected'},
            'NASDAQ': {'status': 'disconnected'},
            'CME': {'status': 'connected'}
        }
        
        summary = dashboard.get_dashboard_summary()
        
        assert summary['connected_exchanges'] == 2

    def test_get_dashboard_summary_health_score(self, dashboard):
        """测试健康分数计算"""
        dashboard._collect_trading_status()
        
        summary = dashboard.get_dashboard_summary()
        
        assert 'health_score' in summary
        assert isinstance(summary['health_score'], (int, float))
        assert 0 <= summary['health_score'] <= 100

    def test_add_status_callback_multiple(self, dashboard):
        """测试添加多个状态回调"""
        callbacks = [Mock() for _ in range(5)]
        
        for callback in callbacks:
            dashboard.add_status_callback(callback)
        
        assert len(dashboard.status_callbacks) == 5

    def test_trigger_status_callbacks_in_monitoring_loop(self, dashboard):
        """测试监控循环中的状态回调触发"""
        callback = Mock()
        dashboard.add_status_callback(callback)
        dashboard._collect_trading_status()
        
        # 模拟监控循环中的一次迭代
        with patch('time.sleep'):
            dashboard.start_monitoring()
            time.sleep(0.15)  # 等待一次迭代
            dashboard.stop_monitoring()
        
        # 回调应该至少被调用一次
        assert callback.call_count >= 1

    def test_start_monitoring_already_running(self, dashboard):
        """测试重复启动监控"""
        dashboard.is_monitoring = True
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.logger') as mock_logger:
            dashboard.start_monitoring()
            mock_logger.warning.assert_called_once()

    def test_start_monitoring_creates_thread(self, dashboard):
        """测试启动监控创建线程"""
        dashboard.is_monitoring = False
        
        with patch('threading.Thread') as mock_thread_class:
            mock_thread = MagicMock()
            mock_thread_class.return_value = mock_thread
            
            dashboard.start_monitoring()
            
            mock_thread_class.assert_called_once()
            mock_thread.start.assert_called_once()
            assert dashboard.is_monitoring is True

    def test_stop_monitoring(self, dashboard):
        """测试停止监控"""
        dashboard.is_monitoring = True
        mock_thread = MagicMock()
        dashboard.monitor_thread = mock_thread
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.logger'):
            dashboard.stop_monitoring()
            
            assert dashboard.is_monitoring is False
            mock_thread.join.assert_called_once_with(timeout=2.0)

    def test_stop_monitoring_no_thread(self, dashboard):
        """测试停止监控时没有线程"""
        dashboard.is_monitoring = True
        dashboard.monitor_thread = None
        
        with patch('src.monitoring.trading.trading_monitor_dashboard.logger'):
            # 应该不抛出异常
            dashboard.stop_monitoring()
            assert dashboard.is_monitoring is False



