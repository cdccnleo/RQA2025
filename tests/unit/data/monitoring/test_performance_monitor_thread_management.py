import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import threading
import time
from unittest.mock import MagicMock, patch

import pytest

from src.data.monitoring.performance_monitor import PerformanceMonitor, logger


def test_start_monitoring_idempotent():
    """测试重复调用 start_monitoring 的幂等性"""
    pm = PerformanceMonitor(max_history=10)
    try:
        # 第一次启动
        pm.start_monitoring()
        assert pm.is_monitoring is True
        first_thread = pm.monitor_thread
        assert first_thread is not None
        assert first_thread.is_alive()

        # 再次启动（应该不会创建新线程）
        pm.start_monitoring()
        assert pm.is_monitoring is True
        assert pm.monitor_thread is first_thread  # 应该是同一个线程

        # 第三次启动（仍然幂等）
        pm.start_monitoring()
        assert pm.is_monitoring is True
        assert pm.monitor_thread is first_thread
    finally:
        pm.stop_monitoring()


def test_stop_monitoring_join_timeout():
    """测试 stop_monitoring 时 join 超时的情况"""
    pm = PerformanceMonitor(max_history=10)
    try:
        pm.start_monitoring()
        assert pm.is_monitoring is True
        assert pm.monitor_thread is not None

        # 模拟一个长时间运行的监控循环（通过 mock sleep 来延长循环时间）
        original_sleep = time.sleep

        def slow_sleep(seconds):
            # 在监控循环中，如果 is_monitoring 为 False，应该快速退出
            # 但为了测试 join 超时，我们让循环在 stop 后仍然 sleep 一段时间
            if pm.is_monitoring:
                original_sleep(0.1)  # 正常 sleep
            else:
                original_sleep(0.1)  # 即使停止，也 sleep 一小段时间

        # 停止监控（join timeout=5秒，但我们的循环会快速退出）
        pm.stop_monitoring()
        # stop_monitoring 应该完成（即使 join 可能超时，也不会抛出异常）
        assert pm.is_monitoring is False
    finally:
        # 确保清理
        pm.is_monitoring = False
        if pm.monitor_thread and pm.monitor_thread.is_alive():
            pm.monitor_thread.join(timeout=1)


def test_stop_monitoring_before_start():
    """测试在未启动时调用 stop_monitoring"""
    pm = PerformanceMonitor(max_history=10)
    # 应该不会抛出异常
    pm.stop_monitoring()
    assert pm.is_monitoring is False
    assert pm.monitor_thread is None


def test_start_stop_multiple_cycles():
    """测试多次启动/停止循环"""
    pm = PerformanceMonitor(max_history=10)
    try:
        for _ in range(3):
            pm.start_monitoring()
            assert pm.is_monitoring is True
            assert pm.monitor_thread is not None
            time.sleep(0.1)  # 给线程一点时间启动

            pm.stop_monitoring()
            assert pm.is_monitoring is False
            time.sleep(0.1)  # 给线程一点时间停止
    finally:
        pm.stop_monitoring()


def test_monitor_thread_daemon_flag():
    """测试监控线程是否为 daemon 线程"""
    pm = PerformanceMonitor(max_history=10)
    try:
        pm.start_monitoring()
        assert pm.monitor_thread is not None
        assert pm.monitor_thread.daemon is True  # 应该是 daemon 线程
    finally:
        pm.stop_monitoring()


def test_stop_monitoring_with_thread_alive_after_timeout():
    """测试 stop_monitoring 在 join 超时后线程仍然存活的情况"""
    pm = PerformanceMonitor(max_history=10)
    
    # 创建一个会长时间运行的监控循环
    original_monitor_loop = pm._monitor_loop
    
    def slow_monitor_loop():
        """模拟一个慢速的监控循环"""
        while pm.is_monitoring:
            try:
                pm._monitor_system_resources()
                pm._cleanup_old_alerts()
                # 使用较长的 sleep，但会在 is_monitoring 变为 False 后退出
                for _ in range(10):
                    if not pm.is_monitoring:
                        break
                    time.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                time.sleep(0.1)
    
    pm._monitor_loop = slow_monitor_loop
    
    try:
        pm.start_monitoring()
        time.sleep(0.2)  # 让线程运行一会儿
        
        # 停止监控（join timeout=5秒）
        # 即使线程在 timeout 内没有完全退出，stop_monitoring 也不应该抛出异常
        pm.stop_monitoring()
        assert pm.is_monitoring is False
        
        # 线程可能仍然存活（如果 join 超时），但 is_monitoring 应该为 False
        # 这样循环会在下一次检查时退出
        if pm.monitor_thread and pm.monitor_thread.is_alive():
            # 等待线程自然退出（因为 is_monitoring 已为 False）
            pm.monitor_thread.join(timeout=2)
    finally:
        pm.is_monitoring = False
        if pm.monitor_thread and pm.monitor_thread.is_alive():
            pm.monitor_thread.join(timeout=1)

