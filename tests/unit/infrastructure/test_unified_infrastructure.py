#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试unified_infrastructure模块

测试目标：提升unified_infrastructure.py的覆盖率到100%
"""

import pytest
from unittest.mock import Mock

from src.infrastructure.unified_infrastructure import (
    CacheType,
    ServiceLifecycle,
    InfrastructureManager,
    get_infrastructure_manager,
    get_config_manager,
    get_monitor,
    get_cache_manager,
    register_infrastructure_service,
    get_infrastructure_service,
    __all__
)


class TestCacheType:
    """测试CacheType枚举"""

    def test_cache_type_values(self):
        """测试缓存类型枚举值"""
        assert CacheType.SMART.value == "smart"
        assert CacheType.MEMORY.value == "memory"
        assert CacheType.REDIS.value == "redis"
        assert CacheType.DISK.value == "disk"
        assert CacheType.HYBRID.value == "hybrid"

    def test_cache_type_members(self):
        """测试缓存类型枚举成员"""
        assert len(CacheType) == 5
        assert CacheType.SMART in CacheType
        assert CacheType.MEMORY in CacheType


class TestServiceLifecycle:
    """测试ServiceLifecycle枚举"""

    def test_service_lifecycle_values(self):
        """测试服务生命周期枚举值"""
        assert ServiceLifecycle.SINGLETON.value == "singleton"
        assert ServiceLifecycle.TRANSIENT.value == "transient"
        assert ServiceLifecycle.SCOPED.value == "scoped"

    def test_service_lifecycle_members(self):
        """测试服务生命周期枚举成员"""
        assert len(ServiceLifecycle) == 3
        assert ServiceLifecycle.SINGLETON in ServiceLifecycle


class TestInfrastructureManager:
    """测试InfrastructureManager类"""

    def test_infrastructure_manager_init(self):
        """测试基础设施管理器初始化"""
        manager = InfrastructureManager()

        assert hasattr(manager, '_services')
        assert hasattr(manager, '_service_lifecycle')
        assert isinstance(manager._services, dict)
        assert isinstance(manager._service_lifecycle, dict)

    def test_get_config_manager(self):
        """测试获取配置管理器"""
        manager = InfrastructureManager()

        result = manager.get_config_manager("test_type", param1="value1")
        assert isinstance(result, dict)
        assert result["type"] == "test_type"
        assert result["config"]["param1"] == "value1"

    def test_get_config_manager_default(self):
        """测试获取配置管理器默认参数"""
        manager = InfrastructureManager()

        result = manager.get_config_manager()
        assert result["type"] == "unified"
        assert result["config"] == {}

    def test_get_monitor(self):
        """测试获取监控器"""
        manager = InfrastructureManager()

        result = manager.get_monitor("performance", interval=30)
        assert isinstance(result, dict)
        assert result["type"] == "performance"
        assert result["config"]["interval"] == 30

    def test_get_monitor_default(self):
        """测试获取监控器默认参数"""
        manager = InfrastructureManager()

        result = manager.get_monitor()
        assert result["type"] == "unified"

    def test_get_cache_manager(self):
        """测试获取缓存管理器"""
        manager = InfrastructureManager()

        result = manager.get_cache_manager("redis", host="localhost", port=6379)
        assert isinstance(result, dict)
        assert result["type"] == "redis"
        assert result["config"]["host"] == "localhost"
        assert result["config"]["port"] == 6379

    def test_get_cache_manager_default(self):
        """测试获取缓存管理器默认参数"""
        manager = InfrastructureManager()

        result = manager.get_cache_manager()
        assert result["type"] == "smart"

    def test_get_cache_alias(self):
        """测试get_cache方法（别名）"""
        manager = InfrastructureManager()

        result1 = manager.get_cache_manager("memory")
        result2 = manager.get_cache("memory")

        assert result1 == result2

    def test_register_service(self):
        """测试注册服务"""
        manager = InfrastructureManager()
        service = Mock()

        manager.register_service("test_service", service, ServiceLifecycle.SINGLETON)

        assert "test_service" in manager._services
        assert manager._services["test_service"] is service
        assert manager._service_lifecycle["test_service"] == ServiceLifecycle.SINGLETON

    def test_register_service_default_lifecycle(self):
        """测试注册服务默认生命周期"""
        manager = InfrastructureManager()
        service = Mock()

        manager.register_service("test_service", service)

        assert manager._service_lifecycle["test_service"] == ServiceLifecycle.SINGLETON

    def test_get_service(self):
        """测试获取服务"""
        manager = InfrastructureManager()
        service = Mock()

        manager.register_service("test_service", service)
        result = manager.get_service("test_service")

        assert result is service

    def test_get_service_not_found(self):
        """测试获取不存在的服务"""
        manager = InfrastructureManager()

        result = manager.get_service("nonexistent")

        assert result is None

    def test_get_available_config_managers(self):
        """测试获取可用配置管理器类型"""
        manager = InfrastructureManager()

        result = manager.get_available_config_managers()
        assert isinstance(result, list)
        assert "unified" in result
        assert "basic" in result
        assert "advanced" in result

    def test_get_available_monitors(self):
        """测试获取可用监控器类型"""
        manager = InfrastructureManager()

        result = manager.get_available_monitors()
        assert isinstance(result, list)
        assert "unified" in result
        assert "performance" in result
        assert "health" in result

    def test_get_available_cache_managers(self):
        """测试获取可用缓存管理器类型"""
        manager = InfrastructureManager()

        result = manager.get_available_cache_managers()
        assert isinstance(result, list)
        assert "smart" in result
        assert "memory" in result
        assert "redis" in result


class TestGlobalFunctions:
    """测试全局便捷函数"""

    def test_get_infrastructure_manager(self):
        """测试获取全局基础设施管理器"""
        manager = get_infrastructure_manager()

        assert isinstance(manager, InfrastructureManager)

    def test_get_config_manager_function(self):
        """测试get_config_manager便捷函数"""
        result = get_config_manager("test", key="value")

        assert isinstance(result, dict)
        assert result["type"] == "test"
        assert result["config"]["key"] == "value"

    def test_get_monitor_function(self):
        """测试get_monitor便捷函数"""
        result = get_monitor("health", check_interval=60)

        assert isinstance(result, dict)
        assert result["type"] == "health"
        assert result["config"]["check_interval"] == 60

    def test_get_cache_manager_function(self):
        """测试get_cache_manager便捷函数"""
        result = get_cache_manager("memory", size=1000)

        assert isinstance(result, dict)
        assert result["type"] == "memory"
        assert result["config"]["size"] == 1000

    def test_register_infrastructure_service(self):
        """测试注册基础设施服务便捷函数"""
        service = Mock()

        register_infrastructure_service("global_service", service, ServiceLifecycle.TRANSIENT)

        # 验证服务已注册到全局管理器
        retrieved = get_infrastructure_service("global_service")
        assert retrieved is service

    def test_get_infrastructure_service(self):
        """测试获取基础设施服务便捷函数"""
        service = Mock()

        register_infrastructure_service("test_global", service)
        result = get_infrastructure_service("test_global")

        assert result is service

    def test_get_infrastructure_service_not_found(self):
        """测试获取不存在的基础设施服务"""
        result = get_infrastructure_service("nonexistent_global")

        assert result is None


class TestModuleExports:
    """测试模块导出"""

    def test_all_exports(self):
        """测试__all__导出列表"""
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "CacheType" in __all__
        assert "ServiceLifecycle" in __all__
        assert "InfrastructureManager" in __all__

    def test_all_exports_functions(self):
        """测试__all__中包含的函数"""
        assert "get_infrastructure_manager" in __all__
        assert "get_config_manager" in __all__
        assert "get_monitor" in __all__
        assert "get_cache_manager" in __all__


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
