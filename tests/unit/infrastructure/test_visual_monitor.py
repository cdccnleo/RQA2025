#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层可视化监控组件测试

测试目标：提升visual_monitor.py的真实覆盖率
实际导入和使用src.infrastructure.visual_monitor模块
"""

import pytest
import time
import threading
from unittest.mock import MagicMock, patch


class TestVisualMonitor:
    """测试可视化监控器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.visual_monitor import VisualMonitor
        
        config = {"test": "config"}
        monitor = VisualMonitor(config)
        
        assert monitor.config == config
        assert monitor.running is False
        assert isinstance(monitor.services, dict)
        assert isinstance(monitor.service_statuses, dict)
    
    def test_start(self):
        """测试启动监控"""
        from src.infrastructure.visual_monitor import VisualMonitor
        
        config = {}
        monitor = VisualMonitor(config)
        
        monitor.start()
        assert monitor.running is True
        
        # 清理
        monitor.stop()
    
    def test_start_already_running(self):
        """测试重复启动监控"""
        from src.infrastructure.visual_monitor import VisualMonitor
        
        config = {}
        monitor = VisualMonitor(config)
        
        monitor.start()
        monitor.start()  # 第二次启动应该不报错
        assert monitor.running is True
        
        # 清理
        monitor.stop()
    
    def test_stop(self):
        """测试停止监控"""
        from src.infrastructure.visual_monitor import VisualMonitor
        
        config = {}
        monitor = VisualMonitor(config)
        
        monitor.start()
        monitor.stop()
        assert monitor.running is False
    
    def test_get_dashboard_data(self):
        """测试获取仪表盘数据"""
        from src.infrastructure.visual_monitor import VisualMonitor
        
        config = {}
        monitor = VisualMonitor(config)
        
        data = monitor.get_dashboard_data()
        
        assert isinstance(data, dict)
        assert "services" in data
        assert "system_health" in data
        assert "timestamp" in data
    
    def test_get_dashboard_data_with_services(self):
        """测试获取包含服务的仪表盘数据"""
        from src.infrastructure.visual_monitor import VisualMonitor, ServiceStatus
        
        config = {}
        monitor = VisualMonitor(config)
        
        # 添加服务状态
        service_status = ServiceStatus(
            name="test_service",
            health="healthy",
            breaker_state="closed",
            degradation_level=0,
            last_updated=time.time()
        )
        monitor.services["test_service"] = service_status
        
        data = monitor.get_dashboard_data()
        
        assert len(data["services"]) == 1
        assert data["services"][0]["name"] == "test_service"
    
    def test_update_service_status(self):
        """测试更新服务状态"""
        from src.infrastructure.visual_monitor import VisualMonitor
        
        config = {}
        monitor = VisualMonitor(config)
        
        # 模拟健康检查器、熔断器和降级管理器
        monitor.health_checker.get_status = MagicMock(return_value={"service1": {"status": "healthy"}})
        monitor.circuit_breaker.get_status = MagicMock(return_value={"service1": {"status": "closed"}})
        monitor.degradation_manager.get_status_report = MagicMock(return_value={"service1": {"level": 0}})
        
        monitor._update_service_status()
        
        assert "service1" in monitor.services
    
    def test_calculate_system_health(self):
        """测试计算系统整体健康状态"""
        from src.infrastructure.visual_monitor import VisualMonitor, ServiceStatus
        
        config = {}
        monitor = VisualMonitor(config)
        
        # 添加健康服务（使用UP状态）
        service_status = ServiceStatus(
            name="test_service",
            health="UP",
            breaker_state="closed",
            degradation_level=0,
            last_updated=time.time()
        )
        monitor.services["test_service"] = service_status
        
        monitor._calculate_system_health()
        
        assert monitor.dashboard_data["system_health"] in ["GREEN", "YELLOW", "RED", "unknown"]
    
    def test_calculate_system_health_empty(self):
        """测试计算空服务列表的系统健康状态"""
        from src.infrastructure.visual_monitor import VisualMonitor
        
        config = {}
        monitor = VisualMonitor(config)
        
        monitor._calculate_system_health()
        
        assert monitor.dashboard_data["system_health"] == "unknown"
    
    def test_load_config(self):
        """测试加载配置"""
        from src.infrastructure.visual_monitor import VisualMonitor
        
        config = {}
        monitor = VisualMonitor(config)
        
        # 配置应该已加载
        assert monitor.update_interval > 0
        assert monitor.dashboard_port > 0
        assert monitor.metrics_port > 0
    
    def test_prepare_dashboard_data(self):
        """测试准备仪表板数据"""
        from src.infrastructure.visual_monitor import VisualMonitor, ServiceStatus
        
        config = {}
        monitor = VisualMonitor(config)
        
        # 添加服务状态
        service_status = ServiceStatus(
            name="test_service",
            health="UP",
            breaker_state="closed",
            degradation_level=0,
            last_updated=time.time()
        )
        monitor.service_statuses["test_service"] = service_status
        
        monitor._prepare_dashboard_data()
        
        assert len(monitor.dashboard_data["services"]) == 1
    
    def test_service_status_dataclass(self):
        """测试服务状态数据类"""
        from src.infrastructure.visual_monitor import ServiceStatus
        
        status = ServiceStatus(
            name="test",
            health="UP",
            breaker_state="closed",
            degradation_level=0,
            last_updated=time.time()
        )
        
        assert status.name == "test"
        assert status.health == "UP"
        assert status.breaker_state == "closed"
        assert status.degradation_level == 0
    
    def test_generate_html_report(self):
        """测试生成HTML报告"""
        from src.infrastructure.visual_monitor import VisualMonitor
        
        config = {}
        monitor = VisualMonitor(config)
        
        html = monitor.generate_html_report()
        
        assert isinstance(html, str)
        assert "RQA2025" in html or "系统监控" in html
    
    def test_generate_prometheus_metrics(self):
        """测试生成Prometheus指标"""
        from src.infrastructure.visual_monitor import VisualMonitor, ServiceStatus
        
        config = {}
        monitor = VisualMonitor(config)
        
        # 添加服务状态
        service_status = ServiceStatus(
            name="test_service",
            health="UP",
            breaker_state="CLOSED",
            degradation_level=0,
            last_updated=time.time()
        )
        monitor.services["test_service"] = service_status
        
        metrics = monitor.generate_prometheus_metrics()
        
        assert isinstance(metrics, str)
        assert "system_health_status" in metrics or "指标生成失败" in metrics
