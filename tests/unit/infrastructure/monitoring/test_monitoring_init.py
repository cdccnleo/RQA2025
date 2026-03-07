#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层monitoring/__init__.py模块测试

测试目标：提升monitoring/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.monitoring模块
"""

import pytest


class TestMonitoringInit:
    """测试monitoring模块初始化"""
    
    def test_performance_monitor_import(self):
        """测试PerformanceMonitor导入"""
        from src.infrastructure.monitoring import PerformanceMonitor
        
        assert PerformanceMonitor is not None
    
    def test_metrics_collector_import(self):
        """测试MetricsCollector导入"""
        from src.infrastructure.monitoring import MetricsCollector
        
        assert MetricsCollector is not None
    
    def test_unified_monitoring_import(self):
        """测试UnifiedMonitoring导入"""
        from src.infrastructure.monitoring import UnifiedMonitoring
        
        assert UnifiedMonitoring is not None
    
    def test_system_monitor_import(self):
        """测试SystemMonitor导入"""
        from src.infrastructure.monitoring import SystemMonitor
        
        assert SystemMonitor is not None
    
    def test_storage_monitor_import(self):
        """测试StorageMonitor导入"""
        from src.infrastructure.monitoring import StorageMonitor
        
        assert StorageMonitor is not None
    
    def test_application_monitor_import(self):
        """测试ApplicationMonitor导入"""
        from src.infrastructure.monitoring import ApplicationMonitor
        
        assert ApplicationMonitor is not None
    
    def test_monitoring_service_alias(self):
        """测试MonitoringService别名"""
        from src.infrastructure.monitoring import MonitoringService
        
        assert MonitoringService is not None
        # MonitoringService应该是UnifiedMonitoring的别名
        from src.infrastructure.monitoring import UnifiedMonitoring
        assert MonitoringService == UnifiedMonitoring
    
    def test_unified_monitoring_service_alias(self):
        """测试UnifiedMonitoringService别名"""
        from src.infrastructure.monitoring import UnifiedMonitoringService
        
        assert UnifiedMonitoringService is not None
        from src.infrastructure.monitoring import UnifiedMonitoring
        assert UnifiedMonitoringService == UnifiedMonitoring

