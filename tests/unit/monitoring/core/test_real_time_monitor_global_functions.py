#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor全局函数测试
补充real_time_monitor.py中全局函数的测试
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, patch, MagicMock
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
    get_monitor = getattr(core_real_time_monitor_module, 'get_monitor', None)
    start_monitoring = getattr(core_real_time_monitor_module, 'start_monitoring', None)
    stop_monitoring = getattr(core_real_time_monitor_module, 'stop_monitoring', None)
    update_business_metric = getattr(core_real_time_monitor_module, 'update_business_metric', None)
    RealTimeMonitor = getattr(core_real_time_monitor_module, 'RealTimeMonitor', None)
    _monitor_instance = getattr(core_real_time_monitor_module, '_monitor_instance', None)
    
    if get_monitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestGlobalFunctions:
    """测试全局函数"""

    def test_get_monitor_creates_instance(self):
        """测试get_monitor创建实例"""
        # 重置全局实例
        import src.monitoring.core.real_time_monitor as rtm_module
        rtm_module._monitor_instance = None
        
        monitor = get_monitor()
        
        assert isinstance(monitor, RealTimeMonitor)
        assert rtm_module._monitor_instance is not None
        assert rtm_module._monitor_instance is monitor

    def test_get_monitor_returns_same_instance(self):
        """测试get_monitor返回相同实例"""
        import src.monitoring.core.real_time_monitor as rtm_module
        
        # 确保有实例
        if rtm_module._monitor_instance is None:
            rtm_module._monitor_instance = RealTimeMonitor()
        
        monitor1 = get_monitor()
        monitor2 = get_monitor()
        
        assert monitor1 is monitor2
        assert monitor1 is rtm_module._monitor_instance

    def test_get_monitor_singleton(self):
        """测试get_monitor单例模式"""
        import src.monitoring.core.real_time_monitor as rtm_module
        
        # 重置全局实例
        rtm_module._monitor_instance = None
        
        monitor1 = get_monitor()
        monitor2 = get_monitor()
        
        assert monitor1 is monitor2

    @patch('src.monitoring.core.real_time_monitor.get_monitor')
    def test_start_monitoring_calls_get_monitor(self, mock_get_monitor):
        """测试start_monitoring调用get_monitor"""
        mock_monitor = Mock(spec=RealTimeMonitor)
        mock_get_monitor.return_value = mock_monitor
        
        start_monitoring()
        
        mock_get_monitor.assert_called_once()
        mock_monitor.start_monitoring.assert_called_once()

    @patch('src.monitoring.core.real_time_monitor.get_monitor')
    def test_start_monitoring_initializes_monitor(self, mock_get_monitor):
        """测试start_monitoring初始化监控器"""
        mock_monitor = Mock(spec=RealTimeMonitor)
        mock_get_monitor.return_value = mock_monitor
        
        start_monitoring()
        
        mock_monitor.start_monitoring.assert_called_once()

    @patch('src.monitoring.core.real_time_monitor.get_monitor')
    def test_stop_monitoring_calls_get_monitor(self, mock_get_monitor):
        """测试stop_monitoring调用get_monitor"""
        mock_monitor = Mock(spec=RealTimeMonitor)
        mock_get_monitor.return_value = mock_monitor
        
        stop_monitoring()
        
        mock_get_monitor.assert_called_once()
        mock_monitor.stop_monitoring.assert_called_once()

    @patch('src.monitoring.core.real_time_monitor.get_monitor')
    def test_stop_monitoring_stops_monitor(self, mock_get_monitor):
        """测试stop_monitoring停止监控器"""
        mock_monitor = Mock(spec=RealTimeMonitor)
        mock_get_monitor.return_value = mock_monitor
        
        stop_monitoring()
        
        mock_monitor.stop_monitoring.assert_called_once()

    @patch('src.monitoring.core.real_time_monitor.get_monitor')
    def test_update_business_metric_calls_get_monitor(self, mock_get_monitor):
        """测试update_business_metric调用get_monitor"""
        mock_monitor = Mock(spec=RealTimeMonitor)
        mock_get_monitor.return_value = mock_monitor
        
        update_business_metric('test_metric', 42.0)
        
        mock_get_monitor.assert_called_once()
        mock_monitor.update_business_metric.assert_called_once_with('test_metric', 42.0)

    @patch('src.monitoring.core.real_time_monitor.get_monitor')
    def test_update_business_metric_with_different_values(self, mock_get_monitor):
        """测试update_business_metric使用不同值"""
        mock_monitor = Mock(spec=RealTimeMonitor)
        mock_get_monitor.return_value = mock_monitor
        
        update_business_metric('request', 1.0)
        update_business_metric('error', 2.0)
        update_business_metric('response_time', 100.0)
        
        assert mock_monitor.update_business_metric.call_count == 3
        mock_monitor.update_business_metric.assert_any_call('request', 1.0)
        mock_monitor.update_business_metric.assert_any_call('error', 2.0)
        mock_monitor.update_business_metric.assert_any_call('response_time', 100.0)

    def test_update_business_metric_integration(self):
        """测试update_business_metric集成"""
        import src.monitoring.core.real_time_monitor as rtm_module
        
        # 重置全局实例
        rtm_module._monitor_instance = None
        
        # 使用真实实例
        update_business_metric('request', 1.0)
        monitor = get_monitor()
        metrics = monitor.metrics_collector.collect_business_metrics()
        
        assert metrics['requests_total'] == 1


class TestGlobalFunctionsIntegration:
    """测试全局函数集成"""

    def test_global_functions_work_together(self):
        """测试全局函数协同工作"""
        import src.monitoring.core.real_time_monitor as rtm_module
        
        # 重置全局实例
        rtm_module._monitor_instance = None
        
        # 获取监控器
        monitor1 = get_monitor()
        assert isinstance(monitor1, RealTimeMonitor)
        
        # 启动监控
        with patch.object(monitor1, 'start_monitoring') as mock_start:
            start_monitoring()
            mock_start.assert_called_once()
        
        # 更新业务指标
        update_business_metric('test', 123.0)
        
        # 停止监控
        with patch.object(monitor1, 'stop_monitoring') as mock_stop:
            stop_monitoring()
            mock_stop.assert_called_once()
        
        # 再次获取应该是同一个实例
        monitor2 = get_monitor()
        assert monitor1 is monitor2

    def test_multiple_get_monitor_calls(self):
        """测试多次调用get_monitor"""
        import src.monitoring.core.real_time_monitor as rtm_module
        
        # 重置全局实例
        rtm_module._monitor_instance = None
        
        monitors = [get_monitor() for _ in range(5)]
        
        # 所有调用应该返回同一个实例
        assert all(mon is monitors[0] for mon in monitors)



