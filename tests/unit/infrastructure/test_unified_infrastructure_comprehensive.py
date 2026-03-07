#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层统一基础设施管理器组件综合测试

测试目标：提升unified_infrastructure.py的真实覆盖率
实际导入和使用src.infrastructure.unified_infrastructure模块
"""

import pytest


class TestCacheType:
    """测试缓存类型枚举"""
    
    def test_cache_type_values(self):
        """测试缓存类型枚举值"""
        from src.infrastructure.unified_infrastructure import CacheType
        
        assert CacheType.SMART.value == "smart"
        assert CacheType.MEMORY.value == "memory"
        assert CacheType.REDIS.value == "redis"
        assert CacheType.DISK.value == "disk"
        assert CacheType.HYBRID.value == "hybrid"


class TestServiceLifecycle:
    """测试服务生命周期枚举"""
    
    def test_service_lifecycle_values(self):
        """测试服务生命周期枚举值"""
        from src.infrastructure.unified_infrastructure import ServiceLifecycle
        
        assert ServiceLifecycle.SINGLETON.value == "singleton"
        assert ServiceLifecycle.TRANSIENT.value == "transient"
        assert ServiceLifecycle.SCOPED.value == "scoped"


class TestInfrastructureManager:
    """测试基础设施管理器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        
        assert isinstance(manager._services, dict)
        assert isinstance(manager._service_lifecycle, dict)
    
    def test_get_config_manager(self):
        """测试获取配置管理器"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        config_mgr = manager.get_config_manager()
        
        assert isinstance(config_mgr, dict)
        assert config_mgr["type"] == "unified"
    
    def test_get_config_manager_custom_type(self):
        """测试获取自定义类型配置管理器"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        config_mgr = manager.get_config_manager("basic", key="value")
        
        assert config_mgr["type"] == "basic"
        assert config_mgr["config"]["key"] == "value"
    
    def test_get_monitor(self):
        """测试获取监控器"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        monitor = manager.get_monitor()
        
        assert isinstance(monitor, dict)
        assert monitor["type"] == "unified"
    
    def test_get_monitor_custom_type(self):
        """测试获取自定义类型监控器"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        monitor = manager.get_monitor("performance", interval=60)
        
        assert monitor["type"] == "performance"
        assert monitor["config"]["interval"] == 60
    
    def test_get_cache_manager(self):
        """测试获取缓存管理器"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        cache_mgr = manager.get_cache_manager()
        
        assert isinstance(cache_mgr, dict)
        assert cache_mgr["type"] == "smart"
    
    def test_get_cache(self):
        """测试获取缓存管理器（别名方法）"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        cache = manager.get_cache("memory")
        
        assert cache["type"] == "memory"
    
    def test_register_service(self):
        """测试注册服务"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager, ServiceLifecycle
        
        manager = InfrastructureManager()
        service = MockService()
        
        manager.register_service("test_service", service, ServiceLifecycle.SINGLETON)
        
        assert "test_service" in manager._services
        assert manager._services["test_service"] == service
        assert manager._service_lifecycle["test_service"] == ServiceLifecycle.SINGLETON
    
    def test_get_service(self):
        """测试获取服务"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        service = MockService()
        
        manager.register_service("test_service", service)
        retrieved = manager.get_service("test_service")
        
        assert retrieved == service
    
    def test_get_service_nonexistent(self):
        """测试获取不存在的服务"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        result = manager.get_service("nonexistent_service")
        
        assert result is None
    
    def test_get_available_config_managers(self):
        """测试获取可用的配置管理器类型"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        types = manager.get_available_config_managers()
        
        assert isinstance(types, list)
        assert "unified" in types
        assert "basic" in types
        assert "advanced" in types
    
    def test_get_available_monitors(self):
        """测试获取可用的监控器类型"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        types = manager.get_available_monitors()
        
        assert isinstance(types, list)
        assert "unified" in types
        assert "performance" in types
        assert "health" in types
    
    def test_get_available_cache_managers(self):
        """测试获取可用的缓存管理器类型"""
        from src.infrastructure.unified_infrastructure import InfrastructureManager
        
        manager = InfrastructureManager()
        types = manager.get_available_cache_managers()
        
        assert isinstance(types, list)
        assert "smart" in types
        assert "memory" in types
        assert "redis" in types
        assert "disk" in types
        assert "hybrid" in types


class MockService:
    """模拟服务类"""
    def __init__(self):
        self.name = "MockService"


class TestGlobalFunctions:
    """测试全局便捷函数"""
    
    def test_get_infrastructure_manager(self):
        """测试获取全局基础设施管理器"""
        from src.infrastructure.unified_infrastructure import get_infrastructure_manager
        
        manager1 = get_infrastructure_manager()
        manager2 = get_infrastructure_manager()
        
        # 应该是同一个实例
        assert manager1 is manager2
    
    def test_get_config_manager_function(self):
        """测试获取配置管理器函数"""
        from src.infrastructure.unified_infrastructure import get_config_manager
        
        config_mgr = get_config_manager("basic")
        
        assert isinstance(config_mgr, dict)
        assert config_mgr["type"] == "basic"
    
    def test_get_monitor_function(self):
        """测试获取监控器函数"""
        from src.infrastructure.unified_infrastructure import get_monitor
        
        monitor = get_monitor("performance")
        
        assert isinstance(monitor, dict)
        assert monitor["type"] == "performance"
    
    def test_get_cache_manager_function(self):
        """测试获取缓存管理器函数"""
        from src.infrastructure.unified_infrastructure import get_cache_manager
        
        cache_mgr = get_cache_manager("memory")
        
        assert isinstance(cache_mgr, dict)
        assert cache_mgr["type"] == "memory"

