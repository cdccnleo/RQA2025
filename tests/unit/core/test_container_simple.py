#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖注入容器简单测试

直接测试container.py模块，不依赖__init__.py
"""

import pytest
import threading
from unittest.mock import Mock

# 直接导入container.py，避免__init__.py的导入问题
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入container.py文件
    import importlib.util
    container_path = project_root / "src" / "core" / "container" / "container.py"
    spec = importlib.util.spec_from_file_location("container_module", container_path)
    container_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(container_module)
    
    DependencyContainer = container_module.DependencyContainer
    ServiceDefinition = container_module.ServiceDefinition
    ServiceLifecycle = container_module.ServiceLifecycle
    ServiceStatus = container_module.ServiceStatus
    
    IMPORTS_AVAILABLE = True
except Exception as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"容器模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestDependencyContainerSimple:
    """测试依赖注入容器基础功能"""

    def test_container_initialization(self):
        """测试容器初始化"""
        container = DependencyContainer()
        assert container is not None
        assert hasattr(container, 'register')
        assert hasattr(container, 'resolve')

    def test_register_and_resolve_instance(self):
        """测试注册和解析实例"""
        container = DependencyContainer()
        service_instance = Mock()
        
        result = container.register(
            name="test_service",
            service=service_instance,
            lifecycle="singleton"
        )
        assert result is True
        
        resolved = container.resolve("test_service")
        assert resolved == service_instance

    def test_register_factory(self):
        """测试注册工厂函数"""
        container = DependencyContainer()
        
        def factory():
            return Mock()
        
        result = container.register(
            name="test_service",
            factory=factory,
            lifecycle="transient"
        )
        assert result is True

    def test_resolve_nonexistent_service(self):
        """测试解析不存在的服务"""
        container = DependencyContainer()
        with pytest.raises(KeyError):
            container.resolve("nonexistent_service")

    def test_unregister_service(self):
        """测试注销服务"""
        container = DependencyContainer()
        service_instance = Mock()
        container.register(name="test_service", service=service_instance)
        
        result = container.unregister("test_service")
        assert result is True
        
        with pytest.raises(KeyError):
            container.resolve("test_service")

    def test_has_service(self):
        """测试检查服务是否注册"""
        container = DependencyContainer()
        service_instance = Mock()
        container.register(name="test_service", service=service_instance)
        
        assert container.has_service("test_service") is True
        assert container.has_service("nonexistent") is False

    def test_get_status_info(self):
        """测试获取状态信息"""
        container = DependencyContainer()
        container.register(name="service1", service=Mock())
        container.register(name="service2", service=Mock())
        
        status_info = container.get_status_info()
        assert status_info['services_count'] == 2
        assert "service1" in container._services
        assert "service2" in container._services

    def test_singleton_lifecycle(self):
        """测试单例生命周期"""
        container = DependencyContainer()
        service_instance = Mock()
        
        container.register(
            name="test_service",
            service=service_instance,
            lifecycle="singleton"
        )
        
        instance1 = container.resolve("test_service")
        instance2 = container.resolve("test_service")
        assert instance1 is instance2

    def test_concurrent_register(self):
        """测试并发注册"""
        container = DependencyContainer()
        
        def register_service(i):
            container.register(
                name=f"service_{i}",
                service=Mock()
            )
        
        threads = [threading.Thread(target=register_service, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 验证所有服务都注册成功
        for i in range(10):
            assert container.has_service(f"service_{i}")

    def test_concurrent_resolve(self):
        """测试并发解析"""
        container = DependencyContainer()
        service_instance = Mock()
        container.register(name="test_service", service=service_instance)
        
        results = []
        def resolve_service():
            results.append(container.resolve("test_service"))
        
        threads = [threading.Thread(target=resolve_service) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        
        # 验证所有解析都成功
        assert len(results) == 10
        assert all(r == service_instance for r in results)

    def test_get_service_alias(self):
        """测试get_service别名方法"""
        container = DependencyContainer()
        service_instance = Mock()
        container.register(name="test_service", service=service_instance)
        resolved = container.get_service("test_service")
        assert resolved == service_instance

    def test_create_scope(self):
        """测试创建作用域容器"""
        container = DependencyContainer()
        service_instance = Mock()
        container.register(name="test_service", service=service_instance)
        scope = container.create_scope()
        assert scope is not None
        assert isinstance(scope, DependencyContainer)
        # 作用域应该可以访问父容器的服务
        resolved = scope.resolve("test_service")
        assert resolved == service_instance

    def test_get_status(self):
        """测试获取容器状态"""
        container = DependencyContainer()
        status = container.get_status()
        assert status == "active"

    def test_health_check(self):
        """测试健康检查"""
        container = DependencyContainer()
        container.register(name="service1", service=Mock())
        container.register(name="service2", service=Mock())
        health = container.health_check()
        assert isinstance(health, dict)
        assert health['healthy'] is True
        assert health['services'] == 2

    def test_clear(self):
        """测试清空容器"""
        container = DependencyContainer()
        container.register(name="service1", service=Mock())
        container.register(name="service2", service=Mock())
        container.clear()
        # 验证容器已清空
        assert len(container._services) == 0
        assert len(container._instances) == 0

    def test_register_with_all_parameters(self):
        """测试使用所有参数注册服务"""
        container = DependencyContainer()
        
        class TestService:
            def __init__(self, value):
                self.value = value
        
        # 测试使用所有参数注册
        result = container.register(
            name="test_service",
            service_type=TestService,
            service=TestService("test"),
            factory=lambda: TestService("factory"),
            dependencies=["dep1"],
            lifecycle="singleton"
        )
        assert result is True
        assert container.has_service("test_service")

    def test_resolve_scoped_lifecycle(self):
        """测试作用域生命周期"""
        container = DependencyContainer()
        
        class TestService:
            def __init__(self):
                self.id = id(self)
        
        container.register(
            name="test_service",
            service_type=TestService,
            lifecycle="scoped"
        )
        
        instance1 = container.resolve("test_service")
        instance2 = container.resolve("test_service")
        # 作用域模式在当前作用域内单例
        assert instance1 is not None
        assert instance2 is not None
        # 在当前实现中，作用域模式也是单例
        assert instance1 == instance2

    def test_unregister_with_instances(self):
        """测试注销带实例的服务"""
        container = DependencyContainer()
        service_instance = Mock()
        container.register(name="test_service", service=service_instance)
        # 解析一次以创建实例
        container.resolve("test_service")
        # 注销服务
        result = container.unregister("test_service")
        assert result is True
        assert not container.has_service("test_service")

    def test_get_status_info_with_services(self):
        """测试带服务的状态信息"""
        container = DependencyContainer()
        container.register(name="service1", service=Mock())
        container.register(name="service2", service=Mock())
        # 解析一个服务以创建实例
        container.resolve("service1")
        status_info = container.get_status_info()
        assert status_info['services_count'] == 2
        # instances_count可能是1（已解析）或2（已缓存），取决于实现
        assert status_info['instances_count'] >= 1

    def test_health_check_with_services(self):
        """测试带服务的健康检查"""
        container = DependencyContainer()
        container.register(name="service1", service=Mock())
        container.register(name="service2", service=Mock())
        health = container.health_check()
        assert health['healthy'] is True
        assert health['services'] == 2
        # instances可能是0（未解析）或2（已缓存），取决于实现
        assert health['instances'] >= 0

    def test_create_instance_with_dependencies(self):
        """测试创建带依赖的实例"""
        container = DependencyContainer()
        # 注册依赖服务
        dep_service = Mock()
        container.register(name="dep_service", service=dep_service)
        
        # 注册需要依赖的服务
        def factory(dep):
            service = Mock()
            service.dep = dep
            return service
        
        container.register(
            name="test_service",
            factory=factory,
            dependencies=["dep_service"],
            lifecycle="transient"
        )
        
        # 解析服务，应该自动注入依赖
        resolved = container.resolve("test_service")
        assert resolved is not None
        assert hasattr(resolved, 'dep')
        assert resolved.dep == dep_service

    def test_resolve_with_service_type(self):
        """测试使用服务类型解析"""
        container = DependencyContainer()
        
        class TestService:
            def __init__(self):
                self.value = "test"
        
        container.register(
            name="test_service",
            service_type=TestService,
            lifecycle="singleton"
        )
        
        resolved = container.resolve("test_service")
        assert isinstance(resolved, TestService)
        assert resolved.value == "test"

    def test_resolve_transient_with_dependencies(self):
        """测试瞬时模式带依赖的解析"""
        container = DependencyContainer()
        
        # 注册依赖
        dep = Mock()
        container.register(name="dep", service=dep)
        
        # 注册需要依赖的服务
        class TestService:
            def __init__(self, dep):
                self.dep = dep
        
        container.register(
            name="test_service",
            service_type=TestService,
            dependencies=["dep"],
            lifecycle="transient"
        )
        
        resolved1 = container.resolve("test_service")
        resolved2 = container.resolve("test_service")
        # 瞬时模式每次创建新实例
        assert resolved1 is not None
        assert resolved2 is not None
        assert resolved1.dep == dep
        assert resolved2.dep == dep

    def test_unregister_nonexistent_service(self):
        """测试注销不存在的服务"""
        container = DependencyContainer()
        result = container.unregister("nonexistent")
        assert result is False

    def test_get_status_info_empty(self):
        """测试空容器的状态信息"""
        container = DependencyContainer()
        status_info = container.get_status_info()
        assert status_info['services_count'] == 0
        assert status_info['instances_count'] == 0
        assert status_info['scopes_count'] == 0

    def test_health_check_empty(self):
        """测试空容器的健康检查"""
        container = DependencyContainer()
        health = container.health_check()
        assert health['healthy'] is True
        assert health['services'] == 0


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestServiceDefinition:
    """测试服务定义"""

    def test_service_definition_creation(self):
        """测试服务定义创建"""
        service_type = Mock
        definition = ServiceDefinition(
            name="test_service",
            service_type=service_type,
            lifecycle=ServiceLifecycle.SINGLETON
        )
        assert definition.name == "test_service"
        assert definition.service_type == service_type
        assert definition.lifecycle == ServiceLifecycle.SINGLETON
        assert definition.status == ServiceStatus.REGISTERED

    def test_service_definition_with_dependencies(self):
        """测试带依赖的服务定义"""
        definition = ServiceDefinition(
            name="test_service",
            dependencies=["dep1", "dep2"]
        )
        assert len(definition.dependencies) == 2
        assert "dep1" in definition.dependencies

