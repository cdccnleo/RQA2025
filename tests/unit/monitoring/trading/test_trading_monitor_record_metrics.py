#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TradingMonitor指标记录功能测试
覆盖record_performance_metrics、record_strategy_metrics、record_risk_metrics等方法
的各种场景和边界情况
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from collections import deque

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
    Alert = getattr(trading_trading_monitor_module, 'Alert', None)
    AlertLevel = getattr(trading_trading_monitor_module, 'AlertLevel', None)
    MonitorType = getattr(trading_trading_monitor_module, 'MonitorType', None)
    if Alert is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestTradingMonitorStopMonitoring:
    """测试停止监控功能"""

    @pytest.fixture
    def monitor(self):
        """创建monitor实例"""
        return TradingMonitor()

    def test_stop_monitoring_not_running(self, monitor):
        """测试停止监控（未运行）"""
        monitor.running = False
        monitor.stop_monitoring()
        
        assert monitor.running == False

    def test_stop_monitoring_running(self, monitor):
        """测试停止监控（正在运行）"""
        monitor.running = True
        
        # Mock线程
        mock_monitor_thread = Mock()
        mock_monitor_thread.is_alive.return_value = True
        mock_alert_thread = Mock()
        mock_alert_thread.is_alive.return_value = True
        
        monitor.monitor_thread = mock_monitor_thread
        monitor.alert_thread = mock_alert_thread
        
        monitor.stop_monitoring()
        
        assert monitor.running == False
        mock_monitor_thread.join.assert_called_once_with(timeout=5)
        mock_alert_thread.join.assert_called_once_with(timeout=5)

    def test_stop_monitoring_thread_not_alive(self, monitor):
        """测试停止监控（线程未运行）"""
        monitor.running = True
        
        # Mock线程（未运行）
        mock_monitor_thread = Mock()
        mock_monitor_thread.is_alive.return_value = False
        mock_alert_thread = Mock()
        mock_alert_thread.is_alive.return_value = False
        
        monitor.monitor_thread = mock_monitor_thread
        monitor.alert_thread = mock_alert_thread
        
        monitor.stop_monitoring()
        
        assert monitor.running == False
        # 未运行的线程不应调用join
        mock_monitor_thread.join.assert_not_called()
        mock_alert_thread.join.assert_not_called()

    def test_stop_monitoring_no_threads(self, monitor):
        """测试停止监控（无线程）"""
        monitor.running = True
        monitor.monitor_thread = None
        monitor.alert_thread = None
        
        monitor.stop_monitoring()
        
        assert monitor.running == False



