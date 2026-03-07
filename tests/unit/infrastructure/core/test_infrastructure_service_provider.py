#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施核心层服务提供者测试

测试目标：提升core/infrastructure_service_provider.py的真实覆盖率
实际导入和使用src.infrastructure.core.infrastructure_service_provider模块
"""

import pytest
from unittest.mock import Mock, MagicMock


class TestInfrastructureServiceProvider:
    """测试基础设施服务提供者"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        assert provider._services == {}
        assert provider._initialized is False
        assert provider._shutdown is False
        assert provider._start_time is not None
    
    def test_config_manager_property(self):
        """测试配置管理器属性"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        config_manager = provider.config_manager
        
        assert config_manager is not None
        assert hasattr(config_manager, 'get')
        assert hasattr(config_manager, 'set')
    
    def test_cache_service_property(self):
        """测试缓存服务属性"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        cache_service = provider.cache_service
        
        assert cache_service is not None
        assert hasattr(cache_service, 'get')
        assert hasattr(cache_service, 'set')
    
    def test_logger_property(self):
        """测试日志器属性"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        logger = provider.logger
        
        assert logger is not None
        assert hasattr(logger, 'info')
        assert hasattr(logger, 'error')
    
    def test_monitor_property(self):
        """测试监控器属性"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        monitor = provider.monitor
        
        assert monitor is not None
        assert hasattr(monitor, 'record_metric')
    
    def test_security_manager_property(self):
        """测试安全管理器属性"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        security_manager = provider.security_manager
        
        assert security_manager is not None
        assert hasattr(security_manager, 'authenticate')
    
    def test_health_checker_property(self):
        """测试健康检查器属性"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        health_checker = provider.health_checker
        
        assert health_checker is not None
    
    def test_resource_manager_property(self):
        """测试资源管理器属性"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        resource_manager = provider.resource_manager
        
        assert resource_manager is not None
        assert hasattr(resource_manager, 'get_resource_usage')
    
    def test_event_bus_property(self):
        """测试事件总线属性"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        event_bus = provider.event_bus
        
        assert event_bus is not None
        assert hasattr(event_bus, 'publish')
    
    def test_service_container_property(self):
        """测试服务容器属性"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        service_container = provider.service_container
        
        assert service_container is not None
    
    def test_initialize_all_services(self):
        """测试初始化所有服务"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        result = provider.initialize_all_services()
        
        assert result is True
        assert provider._initialized is True
        assert len(provider._services) > 0
    
    def test_shutdown_all_services(self):
        """测试关闭所有服务"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        provider.initialize_all_services()
        
        result = provider.shutdown_all_services()
        
        assert result is True
        assert provider._shutdown is True
        assert len(provider._services) == 0
    
    def test_get_service_status_stopped(self):
        """测试获取服务状态（已停止）"""
        from src.infrastructure.core.infrastructure_service_provider import (
            InfrastructureServiceProvider,
            InfrastructureServiceStatus
        )
        
        provider = InfrastructureServiceProvider()
        provider._shutdown = True
        
        status = provider.get_service_status()
        assert status == InfrastructureServiceStatus.STOPPED
    
    def test_get_service_status_initializing(self):
        """测试获取服务状态（初始化中）"""
        from src.infrastructure.core.infrastructure_service_provider import (
            InfrastructureServiceProvider,
            InfrastructureServiceStatus
        )
        
        provider = InfrastructureServiceProvider()
        
        status = provider.get_service_status()
        assert status == InfrastructureServiceStatus.INITIALIZING
    
    def test_get_service_status_running(self):
        """测试获取服务状态（运行中）"""
        from src.infrastructure.core.infrastructure_service_provider import (
            InfrastructureServiceProvider,
            InfrastructureServiceStatus
        )
        
        provider = InfrastructureServiceProvider()
        provider.initialize_all_services()
        
        status = provider.get_service_status()
        assert status in [
            InfrastructureServiceStatus.RUNNING,
            InfrastructureServiceStatus.DEGRADED,
            InfrastructureServiceStatus.ERROR
        ]
    
    def test_get_service_health_report(self):
        """测试获取服务健康报告"""
        from src.infrastructure.core.infrastructure_service_provider import InfrastructureServiceProvider
        
        provider = InfrastructureServiceProvider()
        provider.initialize_all_services()
        
        report = provider.get_service_health_report()
        
        assert isinstance(report, dict)
        assert len(report) > 0
        for service_name, result in report.items():
            assert hasattr(result, 'service_name') or 'service_name' in result.__dict__


class TestMockServices:
    """测试Mock服务类"""
    
    def test_mock_config_manager(self):
        """测试Mock配置管理器"""
        from src.infrastructure.core.infrastructure_service_provider import MockConfigManager
        
        manager = MockConfigManager()
        
        assert manager.get("test_key") is None
        assert manager.set("test_key", "test_value") is True
        assert manager.get("test_key") == "test_value"
        # get_section只返回以"section."开头的键
        assert manager.get_section("test") == {}
        manager.set("test.sub_key", "sub_value")
        assert manager.get_section("test") == {"test.sub_key": "sub_value"}
        assert manager.reload() is True
        assert manager.validate_config() == []
    
    def test_mock_cache_service(self):
        """测试Mock缓存服务"""
        from src.infrastructure.core.infrastructure_service_provider import MockCacheService
        
        cache = MockCacheService()
        
        assert cache.get("test_key") is None
        assert cache.set("test_key", "test_value") is True
        assert cache.get("test_key") == "test_value"
        assert cache.exists("test_key") is True
        assert cache.delete("test_key") is True
        assert cache.exists("test_key") is False
        assert cache.clear() is True
        assert cache.get_stats()["total_keys"] == 0
    
    def test_mock_logger(self):
        """测试Mock日志器"""
        from src.infrastructure.core.infrastructure_service_provider import MockLogger
        
        logger = MockLogger()
        
        logger.debug("debug message")
        logger.info("info message")
        logger.warning("warning message")
        logger.error("error message")
        logger.critical("critical message")
        logger.log("INFO", "log message")
        assert logger.is_enabled_for("INFO") is True
    
    def test_mock_monitor(self):
        """测试Mock监控器"""
        from src.infrastructure.core.infrastructure_service_provider import MockMonitor
        
        monitor = MockMonitor()
        
        monitor.record_metric("test_metric", 100)
        monitor.increment_counter("test_counter")
        monitor.record_histogram("test_histogram", 50.0)
        timer_id = monitor.start_timer("test_timer")
        assert timer_id.startswith("timer_")
        elapsed = monitor.stop_timer(timer_id)
        assert isinstance(elapsed, float)
        assert monitor.get_metrics() == []
    
    def test_mock_security_manager(self):
        """测试Mock安全管理器"""
        from src.infrastructure.core.infrastructure_service_provider import MockSecurityManager
        
        manager = MockSecurityManager()
        
        assert manager.authenticate("user", "pass") is None
        assert manager.validate_token("token") is None
        assert manager.authorize("token", "resource", "action") is True
        assert manager.create_user({}) is True
        assert manager.update_user("user", {}) is True
        assert manager.delete_user("user") is True
        assert manager.get_user_permissions("user") == []
    
    def test_mock_health_checker(self):
        """测试Mock健康检查器"""
        from src.infrastructure.core.infrastructure_service_provider import MockHealthChecker
        
        checker = MockHealthChecker()
        
        assert checker.get_health_history() == []
        assert checker.get_detailed_status() == {"status": "healthy"}
    
    def test_mock_resource_manager(self):
        """测试Mock资源管理器"""
        from src.infrastructure.core.infrastructure_service_provider import MockResourceManager
        
        manager = MockResourceManager()
        
        assert manager.get_resource_usage("cpu") is None
        assert manager.set_resource_limit("cpu", 100, "percent") is True
        assert manager.check_resource_available("cpu", 50) is True
        assert manager.allocate_resource("cpu", 30) is True
        assert manager.release_resource("cpu", 20) is True
        assert manager.get_all_resource_quotas() == {}
    
    def test_mock_event_bus(self):
        """测试Mock事件总线"""
        from src.infrastructure.core.infrastructure_service_provider import MockEventBus
        
        bus = MockEventBus()
        
        assert bus.publish("event") is True
        sub_id = bus.subscribe("event_type", lambda x: None)
        assert sub_id == "sub_123"
        assert bus.unsubscribe(sub_id) is True
        task_id = bus.publish_async("event")
        assert task_id == "task_123"
        assert bus.get_event_history() == []
    
    def test_mock_service_container(self):
        """测试Mock服务容器"""
        from src.infrastructure.core.infrastructure_service_provider import MockServiceContainer
        
        container = MockServiceContainer()
        
        container.register(str, str)
        container.register_instance(str, "instance")
        assert container.resolve(str) is None
        assert container.has_service(str) is False
        assert container.unregister(str) is True
        assert container.get_registered_services() == []
