#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层连接池监控器组件测试

测试目标：提升utils/components/connection_pool_monitor.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.connection_pool_monitor模块
"""

import pytest
import time
from unittest.mock import MagicMock


class TestConnectionPoolMonitor:
    """测试连接池监控器"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        assert monitor.leak_detection_threshold == 60.0
        assert monitor.leak_detection_enabled is True
        assert monitor.total_created == 0
        assert monitor.total_destroyed == 0
        assert monitor.total_acquired == 0
        assert monitor.total_released == 0
    
    def test_init_with_threshold(self):
        """测试使用阈值初始化"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor(leak_detection_threshold=120.0)
        assert monitor.leak_detection_threshold == 120.0
    
    def test_record_connection_created(self):
        """测试记录连接创建"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        monitor.record_connection_created()
        
        assert monitor.total_created == 1
    
    def test_record_connection_destroyed(self):
        """测试记录连接销毁"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        monitor.record_connection_destroyed()
        
        assert monitor.total_destroyed == 1
    
    def test_record_connection_acquired(self):
        """测试记录连接获取"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        monitor.record_connection_acquired(0.5)
        
        assert monitor.total_acquired == 1
        assert monitor.avg_acquire_time == 0.5
        assert monitor.max_acquire_time == 0.5
    
    def test_record_connection_released(self):
        """测试记录连接释放"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        monitor.record_connection_released()
        
        assert monitor.total_released == 1
    
    def test_record_timeout(self):
        """测试记录超时"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        monitor.record_timeout()
        
        assert monitor.total_timeouts == 1
    
    def test_record_error(self):
        """测试记录错误"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        monitor.record_error()
        
        assert monitor.total_errors == 1
    
    def test_detect_connection_leaks(self):
        """测试检测连接泄漏"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor(leak_detection_threshold=1.0)
        
        current_time = time.time()
        active_connections = {
            "conn1": (MagicMock(), current_time - 2.0),  # 超过阈值
            "conn2": (MagicMock(), current_time - 0.5),  # 未超过阈值
        }
        
        leaked = monitor.detect_connection_leaks(active_connections)
        assert len(leaked) == 1
        assert "conn1" in leaked
    
    def test_detect_connection_leaks_disabled(self):
        """测试泄漏检测禁用"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        monitor.leak_detection_enabled = False
        
        active_connections = {"conn1": (MagicMock(), time.time())}
        leaked = monitor.detect_connection_leaks(active_connections)
        
        assert len(leaked) == 0
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        monitor.record_connection_created()
        monitor.record_connection_acquired(0.5)
        monitor.record_connection_released()
        
        stats = monitor.get_statistics([], 5, 3)
        
        assert stats["total_created"] == 1
        assert stats["total_acquired"] == 1
        assert stats["total_released"] == 1
        assert stats["available_connections"] == 5
        assert stats["active_connections"] == 3
    
    def test_reset_statistics(self):
        """测试重置统计信息"""
        from src.infrastructure.utils.components.connection_pool_monitor import ConnectionPoolMonitor
        
        monitor = ConnectionPoolMonitor()
        monitor.record_connection_created()
        monitor.record_connection_acquired(0.5)
        
        monitor.reset_statistics()
        
        assert monitor.total_created == 0
        assert monitor.total_acquired == 0
        assert monitor.avg_acquire_time == 0.0
        assert len(monitor.acquire_times) == 0

