#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试应用层生产环境监控器
"""

import pytest
from datetime import datetime


def test_production_monitor_initialization():
    """测试生产环境监控器初始化"""
    from src.infrastructure.monitoring.application.production_monitor import ProductionMonitor
    
    monitor = ProductionMonitor()
    assert monitor.status == "idle"


def test_production_monitor_start():
    """测试启动监控"""
    from src.infrastructure.monitoring.application.production_monitor import ProductionMonitor
    
    monitor = ProductionMonitor()
    monitor.start()
    assert monitor.status == "running"


def test_production_monitor_stop():
    """测试停止监控"""
    from src.infrastructure.monitoring.application.production_monitor import ProductionMonitor
    
    monitor = ProductionMonitor()
    monitor.start()
    monitor.stop()
    assert monitor.status == "stopped"


def test_production_monitor_system_monitoring():
    """测试系统监控"""
    from src.infrastructure.monitoring.application.production_monitor import ProductionMonitor
    
    monitor = ProductionMonitor()
    monitor.start()
    
    result = monitor.monitor_system()
    assert isinstance(result, dict)
    assert "status" in result
    assert "checked_at" in result
    assert result["status"] == "running"
    assert isinstance(result["checked_at"], str)


def test_production_monitor_health_check():
    """测试健康检查"""
    from src.infrastructure.monitoring.application.production_monitor import ProductionMonitor
    
    monitor = ProductionMonitor()
    
    health = monitor.health_check()
    assert isinstance(health, dict)
    assert "status" in health
    assert "timestamp" in health
    assert health["status"] == "healthy"
    assert isinstance(health["timestamp"], str)


def test_production_monitor_lifecycle():
    """测试监控器生命周期"""
    from src.infrastructure.monitoring.application.production_monitor import ProductionMonitor
    
    monitor = ProductionMonitor()
    assert monitor.status == "idle"
    
    monitor.start()
    assert monitor.status == "running"
    
    monitor.stop()
    assert monitor.status == "stopped"
    
    # 可以重复停止
    monitor.stop()
    assert monitor.status == "stopped"
    
    # 可以重新启动
    monitor.start()
    assert monitor.status == "running"

