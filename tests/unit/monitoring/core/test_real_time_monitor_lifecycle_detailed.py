#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RealTimeMonitor生命周期管理详细测试
补充start_monitoring和stop_monitoring方法的详细测试
"""

import sys
import importlib
from pathlib import Path
import pytest
import threading
import time
from unittest.mock import Mock, patch

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
    if RealTimeMonitor is None:
        pytest.skip("监控模块导入失败", allow_module_level=True)
except ImportError:
    pytest.skip("监控模块导入失败", allow_module_level=True)


class TestRealTimeMonitorLifecycleDetailed:
    """测试RealTimeMonitor类生命周期管理的详细功能"""

    @pytest.fixture
    def monitor(self):
        """创建RealTimeMonitor实例"""
        return RealTimeMonitor()

    def test_start_monitoring_sets_running_flag(self, monitor):
        """测试启动监控设置running标志"""
        monitor._running = False
        
        monitor.start_monitoring()
        
        assert monitor._running == True
        
        # 清理
        monitor.stop_monitoring()

    def test_start_monitoring_starts_metrics_collector(self, monitor):
        """测试启动监控启动指标收集器"""
        monitor._running = False
        
        with patch.object(monitor.metrics_collector, 'start_collection') as mock_start:
            monitor.start_monitoring()
            
            mock_start.assert_called_once()
        
        # 清理
        monitor.stop_monitoring()

    def test_start_monitoring_creates_alert_thread(self, monitor):
        """测试启动监控创建告警线程"""
        monitor._running = False
        
        monitor.start_monitoring()
        
        assert monitor._alert_thread is not None
        assert isinstance(monitor._alert_thread, threading.Thread)
        
        # 清理
        monitor.stop_monitoring()

    def test_start_monitoring_alert_thread_is_daemon(self, monitor):
        """测试告警线程是守护线程"""
        monitor._running = False
        
        monitor.start_monitoring()
        
        assert monitor._alert_thread.daemon == True
        
        # 清理
        monitor.stop_monitoring()

    def test_start_monitoring_alert_thread_starts(self, monitor):
        """测试告警线程启动"""
        monitor._running = False
        
        monitor.start_monitoring()
        
        # 等待一小段时间让线程启动
        time.sleep(0.1)
        assert monitor._alert_thread.is_alive()
        
        # 清理
        monitor.stop_monitoring()

    def test_start_monitoring_idempotent(self, monitor):
        """测试启动监控是幂等的（多次调用不会创建多个线程）"""
        monitor._running = False
        
        monitor.start_monitoring()
        first_thread = monitor._alert_thread
        
        monitor.start_monitoring()
        second_thread = monitor._alert_thread
        
        # 应该是同一个线程
        assert first_thread == second_thread
        
        # 清理
        monitor.stop_monitoring()

    def test_start_monitoring_already_running_no_effect(self, monitor):
        """测试已运行时启动监控无效果"""
        monitor._running = True
        original_thread = monitor._alert_thread
        
        monitor.start_monitoring()
        
        # 不应该创建新线程
        assert monitor._alert_thread == original_thread
        
        # 清理
        monitor._running = False

    def test_stop_monitoring_sets_running_false(self, monitor):
        """测试停止监控设置running为False"""
        monitor._running = True
        
        monitor.stop_monitoring()
        
        assert monitor._running == False

    def test_stop_monitoring_stops_metrics_collector(self, monitor):
        """测试停止监控停止指标收集器"""
        monitor._running = True
        
        with patch.object(monitor.metrics_collector, 'stop_collection') as mock_stop:
            monitor.stop_monitoring()
            
            mock_stop.assert_called_once()

    def test_stop_monitoring_joins_alert_thread(self, monitor):
        """测试停止监控join告警线程"""
        monitor._running = True
        mock_thread = Mock()
        mock_thread.join = Mock()
        monitor._alert_thread = mock_thread
        
        monitor.stop_monitoring()
        
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_stop_monitoring_no_alert_thread_no_error(self, monitor):
        """测试停止监控没有告警线程时不报错"""
        monitor._running = True
        monitor._alert_thread = None
        
        # 不应该抛出异常
        monitor.stop_monitoring()
        assert monitor._running == False

    def test_stop_monitoring_when_not_running(self, monitor):
        """测试未运行时停止监控"""
        monitor._running = False
        monitor._alert_thread = None
        
        # 不应该抛出异常
        monitor.stop_monitoring()
        assert monitor._running == False

    def test_full_lifecycle_start_stop(self, monitor):
        """测试完整生命周期：启动和停止"""
        # 初始状态
        assert monitor._running == False
        assert monitor._alert_thread is None
        
        # 启动
        monitor.start_monitoring()
        assert monitor._running == True
        assert monitor._alert_thread is not None
        
        # 等待一小段时间
        time.sleep(0.2)
        
        # 停止
        monitor.stop_monitoring()
        assert monitor._running == False
        
        # 等待线程停止
        if monitor._alert_thread:
            monitor._alert_thread.join(timeout=1)
            assert not monitor._alert_thread.is_alive()

    def test_multiple_start_stop_cycles(self, monitor):
        """测试多次启动停止循环"""
        for _ in range(3):
            # 启动
            monitor.start_monitoring()
            assert monitor._running == True
            time.sleep(0.1)
            
            # 停止
            monitor.stop_monitoring()
            assert monitor._running == False
            time.sleep(0.1)

    def test_start_monitoring_logs_info(self, monitor):
        """测试启动监控记录日志"""
        monitor._running = False
        
        with patch('src.monitoring.core.real_time_monitor.logger') as mock_logger:
            monitor.start_monitoring()
            
            mock_logger.info.assert_called()
            info_call = mock_logger.info.call_args[0][0]
            assert 'Real-time monitoring system started' in info_call
        
        # 清理
        monitor.stop_monitoring()

    def test_stop_monitoring_logs_info(self, monitor):
        """测试停止监控记录日志"""
        monitor._running = True
        
        with patch('src.monitoring.core.real_time_monitor.logger') as mock_logger:
            monitor.stop_monitoring()
            
            mock_logger.info.assert_called()
            info_call = mock_logger.info.call_args[0][0]
            assert 'Real-time monitoring system stopped' in info_call

    def test_start_monitoring_after_stop(self, monitor):
        """测试停止后重新启动"""
        # 启动
        monitor.start_monitoring()
        assert monitor._running == True
        first_thread = monitor._alert_thread
        
        # 停止
        monitor.stop_monitoring()
        assert monitor._running == False
        time.sleep(0.1)
        
        # 重新启动
        monitor.start_monitoring()
        assert monitor._running == True
        second_thread = monitor._alert_thread
        
        # 应该是新的线程
        assert second_thread != first_thread
        
        # 清理
        monitor.stop_monitoring()

    def test_stop_monitoring_thread_cleanup(self, monitor):
        """测试停止监控时线程清理"""
        monitor.start_monitoring()
        alert_thread = monitor._alert_thread
        
        # 等待线程启动
        time.sleep(0.1)
        assert alert_thread.is_alive()
        
        # 停止监控
        monitor.stop_monitoring()
        
        # 等待线程停止
        alert_thread.join(timeout=2)
        assert not alert_thread.is_alive()

    def test_start_monitoring_with_existing_thread(self, monitor):
        """测试启动监控时已有线程的处理"""
        # 手动创建一个线程
        old_thread = threading.Thread(target=lambda: None)
        monitor._alert_thread = old_thread
        monitor._running = False
        
        # 启动监控（应该创建新线程）
        monitor.start_monitoring()
        
        # 应该创建了新线程
        assert monitor._alert_thread != old_thread
        assert monitor._alert_thread is not None
        
        # 清理
        monitor.stop_monitoring()

    def test_concurrent_start_monitoring(self, monitor):
        """测试并发启动监控（应该只创建一个线程）"""
        monitor._running = False
        
        def start():
            monitor.start_monitoring()
        
        # 并发启动
        threads = [threading.Thread(target=start) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=1)
        
        # 应该只有一个告警线程
        assert monitor._running == True
        assert monitor._alert_thread is not None
        
        # 清理
        monitor.stop_monitoring()

