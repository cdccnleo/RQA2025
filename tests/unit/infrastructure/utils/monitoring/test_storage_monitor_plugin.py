#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层存储监控插件组件测试

测试目标：提升utils/monitoring/storage_monitor_plugin.py的真实覆盖率
实际导入和使用src.infrastructure.utils.monitoring.storage_monitor_plugin模块
"""

import pytest
import time
from unittest.mock import MagicMock


class TestStorageMonitorPlugin:
    """测试存储监控插件类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        assert monitor._write_count == 0
        assert monitor._error_count == 0
        assert monitor._total_size == 0
        assert monitor._start_time > 0
    
    def test_record_write(self):
        """测试记录写入操作"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        monitor.record_write(symbol="600519", size=1024, status=True)
        
        assert monitor._write_count == 1
        assert monitor._total_size == 1024
    
    def test_record_write_failed(self):
        """测试记录失败的写入操作"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        monitor.record_write(symbol="600519", size=1024, status=False)
        
        assert monitor._write_count == 1
        assert monitor._total_size == 0
    
    def test_record_write_multiple(self):
        """测试记录多次写入操作"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        monitor.record_write(symbol="600519", size=1024, status=True)
        monitor.record_write(symbol="000001", size=2048, status=True)
        
        assert monitor._write_count == 2
        assert monitor._total_size == 3072
    
    def test_record_error(self):
        """测试记录错误"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        monitor.record_error(symbol="600519")
        
        assert monitor._error_count == 1
    
    def test_record_error_multiple(self):
        """测试记录多个错误"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        monitor.record_error(symbol="600519")
        monitor.record_error(symbol="000001")
        
        assert monitor._error_count == 2
    
    def test_get_stats(self):
        """测试获取监控统计"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        monitor.record_write(symbol="600519", size=1024, status=True)
        monitor.record_error(symbol="000001")
        
        stats = monitor.get_stats()
        
        assert stats["write_count"] == 1
        assert stats["error_count"] == 1
        assert stats["total_size"] == 1024
        assert stats["uptime"] >= 0
        assert "write_rate" in stats
        assert "error_rate" in stats
    
    def test_get_stats_empty(self):
        """测试获取空统计"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        stats = monitor.get_stats()
        
        assert stats["write_count"] == 0
        assert stats["error_count"] == 0
        assert stats["total_size"] == 0
    
    def test_reset(self):
        """测试重置统计"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        monitor.record_write(symbol="600519", size=1024, status=True)
        monitor.record_error(symbol="000001")
        
        assert monitor._write_count == 1
        assert monitor._error_count == 1
        
        monitor.reset()
        
        assert monitor._write_count == 0
        assert monitor._error_count == 0
        assert monitor._total_size == 0
    
    def test_get_stats_with_uptime(self):
        """测试获取统计（包含运行时间）"""
        from src.infrastructure.utils.monitoring.storage_monitor_plugin import StorageMonitorPlugin
        
        monitor = StorageMonitorPlugin()
        time.sleep(0.1)  # 等待一小段时间
        
        stats = monitor.get_stats()
        
        assert stats["uptime"] > 0
        assert isinstance(stats["write_rate"], (int, float))
        assert isinstance(stats["error_rate"], (int, float))

