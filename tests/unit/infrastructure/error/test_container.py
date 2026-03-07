"""
基础设施层 - DependencyContainer 单元测试

测试依赖注入容器的核心功能，包括服务注册、解析、生命周期管理等。
覆盖率目标: 80%+
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import unittest
from unittest.mock import Mock, patch, MagicMock
import threading

from src.infrastructure.error.core.container import (
    DependencyContainer,
    ServiceDescriptor,
    ServiceRegistrationConfig,
    Lifecycle
)


class TestServiceDescriptor(unittest.TestCase):
    """ServiceDescriptor 单元测试"""

    def test_initialization_with_implementation(self):
        """测试带实现的初始化"""
        config = ServiceRegistrationConfig(
            service_type=str,
            implementation=int,
            lifecycle=Lifecycle.TRANSIENT
        )
        descriptor = ServiceDescriptor(config)

        self.assertEqual(descriptor.service_type, str)
        self.assertEqual(descriptor.implementation, int)
        self.assertEqual(descriptor.lifecycle, Lifecycle.TRANSIENT)
        self.assertIsNone(descriptor.factory)
        self.assertIsNone(descriptor.instance)

    def test_initialization_default_values(self):
        """测试默认值初始化"""
        config = ServiceRegistrationConfig(service_type=str)
        descriptor = ServiceDescriptor(config)

        self.assertEqual(descriptor.service_type, str)
        self.assertEqual(descriptor.implementation, str)
        self.assertEqual(descriptor.lifecycle, Lifecycle.SINGLETON)
        self.assertIsNone(descriptor.factory)
        self.assertIsNone(descriptor.instance)


class TestDependencyContainer(unittest.TestCase):
    """DependencyContainer 单元测试"""

    def setUp(self):
        """测试前准备"""
        self.container = DependencyContainer()

    def tearDown(self):
        """测试后清理"""
        pass

    def test_initialization(self):
        """测试初始化"""
        container = DependencyContainer()

        self.assertEqual(container._services, {})
        self.assertEqual(container._scoped_instances, {})

    def test_register_service_with_implementation(self):
        """测试注册带实现的服務"""
        self.container.register(str, implementation=int, lifecycle=Lifecycle.TRANSIENT)

        self.assertIn(str, self.container._services)
        descriptor = self.container._services[str]
        self.assertEqual(descriptor.service_type, str)
        self.assertEqual(descriptor.implementation, int)
        self.assertEqual(descriptor.lifecycle, Lifecycle.TRANSIENT)

    def test_register_service_singleton_default(self):
        """测试注册单例服务（默认）"""
        self.container.register(str)

        descriptor = self.container._services[str]
        self.assertEqual(descriptor.lifecycle, Lifecycle.SINGLETON)

    def test_register_singleton(self):
        """测试注册单例服务"""
        self.container.register_singleton(str, implementation=int)

        descriptor = self.container._services[str]
        self.assertEqual(descriptor.implementation, int)
        self.assertEqual(descriptor.lifecycle, Lifecycle.SINGLETON)

    def test_register_transient(self):
        """测试注册瞬时服务"""
        self.container.register_transient(str, implementation=int)

        descriptor = self.container._services[str]
        self.assertEqual(descriptor.implementation, int)
        self.assertEqual(descriptor.lifecycle, Lifecycle.TRANSIENT)

    def test_register_scoped(self):
        """测试注册作用域服务"""
        self.container.register_scoped(str, implementation=int)

        descriptor = self.container._services[str]
        self.assertEqual(descriptor.implementation, int)
        self.assertEqual(descriptor.lifecycle, Lifecycle.SCOPED)

    def test_resolve_singleton(self):
        """测试解析单例服务"""
        class TestService:
            def __init__(self):
                self.value = 42

        self.container.register(TestService)

        # 第一次解析
        instance1 = self.container.resolve(TestService)
        self.assertIsInstance(instance1, TestService)
        self.assertEqual(instance1.value, 42)

        # 第二次解析应该返回同一个实例
        instance2 = self.container.resolve(TestService)
        self.assertIs(instance1, instance2)

    def test_resolve_transient(self):
        """测试解析瞬时服务"""
        class TestService:
            def __init__(self):
                self.value = 42

        self.container.register_transient(TestService)

        # 每次解析都应该返回新实例
        instance1 = self.container.resolve(TestService)
        instance2 = self.container.resolve(TestService)

        self.assertIsInstance(instance1, TestService)
        self.assertIsInstance(instance2, TestService)
        self.assertIsNot(instance1, instance2)

    def test_resolve_scoped(self):
        """测试解析作用域服务"""
        class TestService:
            def __init__(self):
                self.value = 42

        self.container.register_scoped(TestService)

        # 在作用域内解析
        with self.container.scope():
            instance1 = self.container.resolve(TestService)
            instance2 = self.container.resolve(TestService)

            # 在同一作用域内应该返回同一个实例
            self.assertIs(instance1, instance2)

        # 在新作用域内应该返回新实例
        with self.container.scope():
            instance3 = self.container.resolve(TestService)
            self.assertIsNot(instance1, instance3)

    def test_resolve_unregistered_service(self):
        """测试解析未注册的服务"""
        with self.assertRaises(KeyError):
            self.container.resolve(str)

    def test_has_service(self):
        """测试服务存在检查"""
        self.assertFalse(self.container.has_service(str))

        self.container.register(str)
        self.assertTrue(self.container.has_service(str))

    def test_get_registered_services(self):
        """测试获取已注册的服务列表"""
        services = self.container.get_registered_services()
        self.assertEqual(services, {})

        self.container.register(str)
        self.container.register(int)

        services = self.container.get_registered_services()
        self.assertEqual(len(services), 2)
        self.assertIn(str, services)
        self.assertIn(int, services)

    def test_clear_all_services(self):
        """测试清空所有服务"""
        self.container.register(str)
        self.container.register(int)
        self.assertEqual(len(self.container.get_registered_services()), 2)

        self.container.clear()
        self.assertEqual(len(self.container.get_registered_services()), 0)
        self.assertEqual(self.container._services, {})

    def test_scope_context_manager(self):
        """测试作用域上下文管理器"""
        # 简单测试作用域上下文管理器可以正常使用
        with self.container.scope():
            pass  # 作用域应该可以正常进入和退出

    def test_thread_safety(self):
        """测试线程安全性"""
        # 简单测试多线程环境下注册服务
        import concurrent.futures

        def register_service(service_id):
            service_type = type(f'Service{service_id}', (), {})
            self.container.register(service_type)
            return service_id

        # 使用线程池执行器
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(register_service, i) for i in range(3)]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 验证所有服务都已注册
        self.assertEqual(len(results), 3)
        self.assertEqual(len(self.container._services), 3)


if __name__ == '__main__':
    unittest.main()
