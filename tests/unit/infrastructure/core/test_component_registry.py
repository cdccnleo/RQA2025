"""
测试基础设施层组件注册表

覆盖 InfrastructureComponentRegistry 和 ComponentFactory 类的功能
"""

import pytest
from unittest.mock import Mock, MagicMock
from src.infrastructure.core.component_registry import (
    InfrastructureComponentRegistry,
    ComponentFactory,
    get_global_registry
)


class TestInfrastructureComponentRegistry:
    """InfrastructureComponentRegistry 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        registry = InfrastructureComponentRegistry()
        assert registry._registry == {}
        assert registry._instances == {}
        assert registry._singletons == {}

    def test_register_component_basic(self):
        """测试基本组件注册"""
        registry = InfrastructureComponentRegistry()
        mock_factory = Mock(return_value="component_instance")

        registry.register_component("test_component", mock_factory)

        assert "test_component" in registry._registry
        assert registry._registry["test_component"] == mock_factory

    def test_register_component_singleton(self):
        """测试单例组件注册"""
        registry = InfrastructureComponentRegistry()
        mock_factory = Mock(return_value="component_instance")

        registry.register_component("test_component", mock_factory, singleton=True)

        assert "test_component" in registry._registry
        assert registry._registry["test_component"] == mock_factory

    def test_get_component_basic(self):
        """测试基本组件获取"""
        registry = InfrastructureComponentRegistry()
        mock_factory = Mock(return_value="component_instance")

        registry.register_component("test_component", mock_factory)
        instance = registry.get_component("test_component")

        assert instance == "component_instance"
        mock_factory.assert_called_once()
        assert "test_component" in registry._instances

    def test_get_component_singleton(self):
        """测试单例组件获取"""
        registry = InfrastructureComponentRegistry()

        class MockSingleton:
            def __init__(self):
                self._is_singleton = True

        mock_factory = Mock(return_value=MockSingleton())
        registry.register_component("test_component", mock_factory, singleton=True)

        instance1 = registry.get_component("test_component")
        instance2 = registry.get_component("test_component")

        assert instance1 is instance2  # 单例应该返回同一实例
        mock_factory.assert_called_once()  # 只调用一次工厂函数

    def test_get_component_unregistered(self):
        """测试获取未注册的组件"""
        registry = InfrastructureComponentRegistry()

        with pytest.raises(KeyError, match="组件 'nonexistent' 未注册"):
            registry.get_component("nonexistent")

    def test_get_component_factory_exception(self):
        """测试组件工厂抛出异常"""
        registry = InfrastructureComponentRegistry()
        mock_factory = Mock(side_effect=Exception("Factory error"))

        registry.register_component("test_component", mock_factory)

        with pytest.raises(Exception, match="Factory error"):
            registry.get_component("test_component")

    def test_has_component(self):
        """测试检查组件是否存在"""
        registry = InfrastructureComponentRegistry()
        mock_factory = Mock(return_value="component_instance")

        assert not registry.has_component("test_component")

        registry.register_component("test_component", mock_factory)

        assert registry.has_component("test_component")
        assert not registry.has_component("nonexistent")

    def test_list_components(self):
        """测试列出所有组件"""
        registry = InfrastructureComponentRegistry()

        assert registry.list_components() == []

        registry.register_component("component1", Mock(return_value="instance1"))
        registry.register_component("component2", Mock(return_value="instance2"))

        components = registry.list_components()
        assert set(components) == {"component1", "component2"}

    def test_unregister_component(self):
        """测试注销组件"""
        registry = InfrastructureComponentRegistry()
        mock_factory = Mock(return_value="component_instance")

        registry.register_component("test_component", mock_factory)
        assert registry.has_component("test_component")

        result = registry.unregister_component("test_component")
        assert result is True
        assert not registry.has_component("test_component")

    def test_unregister_component_nonexistent(self):
        """测试注销不存在的组件"""
        registry = InfrastructureComponentRegistry()

        result = registry.unregister_component("nonexistent")
        assert result is False

    def test_clear_cache_all(self):
        """测试清除所有缓存"""
        registry = InfrastructureComponentRegistry()
        mock_factory1 = Mock(return_value="instance1")
        mock_factory2 = Mock(return_value="instance2")

        registry.register_component("component1", mock_factory1)
        registry.register_component("component2", mock_factory2)

        # 创建实例
        registry.get_component("component1")
        registry.get_component("component2")

        assert len(registry._instances) == 2

        registry.clear_cache()

        assert len(registry._instances) == 0
        assert len(registry._singletons) == 0

    def test_clear_cache_specific(self):
        """测试清除特定组件缓存"""
        registry = InfrastructureComponentRegistry()
        mock_factory = Mock(return_value="component_instance")

        registry.register_component("test_component", mock_factory)
        registry.get_component("test_component")

        assert "test_component" in registry._instances

        registry.clear_cache("test_component")

        assert "test_component" not in registry._instances

    def test_clear_cache_nonexistent(self):
        """测试清除不存在的组件缓存"""
        registry = InfrastructureComponentRegistry()

        # 不应该抛出异常
        registry.clear_cache("nonexistent")


class TestComponentFactory:
    """ComponentFactory 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        registry = InfrastructureComponentRegistry()
        factory = ComponentFactory(registry)

        assert factory.registry is registry

    def test_create_component(self):
        """测试创建组件"""
        registry = InfrastructureComponentRegistry()
        factory = ComponentFactory(registry)

        class TestComponent:
            def __init__(self, value):
                self.value = value

        component = factory.create_component(TestComponent, "test_value")

        assert isinstance(component, TestComponent)
        assert component.value == "test_value"

    def test_register_factory(self):
        """测试注册工厂函数"""
        registry = InfrastructureComponentRegistry()
        factory = ComponentFactory(registry)

        mock_factory_func = Mock(return_value="test_instance")

        factory.register_factory("test_component", mock_factory_func)

        assert registry.has_component("test_component")
        assert registry.get_component("test_component") == "test_instance"


class TestGlobalRegistry:
    """全局注册表测试"""

    def test_get_global_registry(self):
        """测试获取全局注册表"""
        registry = get_global_registry()

        assert isinstance(registry, InfrastructureComponentRegistry)

        # 应该返回同一实例
        registry2 = get_global_registry()
        assert registry is registry2

    def test_global_registry_functionality(self):
        """测试全局注册表功能"""
        registry = get_global_registry()

        # 注册组件
        mock_factory = Mock(return_value="global_instance")
        registry.register_component("global_test", mock_factory)

        # 获取组件
        instance = registry.get_component("global_test")
        assert instance == "global_instance"

        # 清理测试数据
        registry.unregister_component("global_test")


class TestInfrastructureComponentRegistryIntegration:
    """组件注册表集成测试"""

    def test_component_lifecycle(self):
        """测试组件完整生命周期"""
        registry = InfrastructureComponentRegistry()

        # 1. 注册组件
        call_count = 0
        def factory():
            nonlocal call_count
            call_count += 1
            return f"instance_{call_count}"

        registry.register_component("lifecycle_test", factory)

        # 2. 获取组件（创建实例）
        instance1 = registry.get_component("lifecycle_test")
        assert instance1 == "instance_1"
        assert call_count == 1

        # 3. 再次获取（目前实现中每次都会调用工厂函数）
        instance2 = registry.get_component("lifecycle_test")
        assert instance2 == "instance_2"  # 创建新实例
        assert call_count == 2  # 工厂函数被再次调用

        # 4. 清除缓存
        registry.clear_cache("lifecycle_test")

        # 5. 再次获取（创建新实例）
        instance3 = registry.get_component("lifecycle_test")
        assert instance3 == "instance_3"
        assert call_count == 3

        # 6. 注销组件
        registry.unregister_component("lifecycle_test")
        assert not registry.has_component("lifecycle_test")

    def test_singleton_behavior(self):
        """测试单例行为"""
        registry = InfrastructureComponentRegistry()

        class SingletonComponent:
            def __init__(self):
                self._is_singleton = True

        call_count = 0
        def singleton_factory():
            nonlocal call_count
            call_count += 1
            return SingletonComponent()

        registry.register_component("singleton_test", singleton_factory)

        # 获取单例组件
        instance1 = registry.get_component("singleton_test")
        instance2 = registry.get_component("singleton_test")

        assert instance1 is instance2  # 应该是同一实例
        assert call_count == 1  # 只调用一次工厂函数
        assert "singleton_test" in registry._singletons

    def test_error_handling(self):
        """测试错误处理"""
        registry = InfrastructureComponentRegistry()

        # 测试注册None工厂
        registry.register_component("none_factory", None)

        # 测试获取时应该抛出TypeError
        with pytest.raises(TypeError):
            registry.get_component("none_factory")

    def test_registry_isolation(self):
        """测试注册表隔离性"""
        registry1 = InfrastructureComponentRegistry()
        registry2 = InfrastructureComponentRegistry()

        registry1.register_component("test", Mock(return_value="instance1"))
        registry2.register_component("test", Mock(return_value="instance2"))

        instance1 = registry1.get_component("test")
        instance2 = registry2.get_component("test")

        assert instance1 == "instance1"
        assert instance2 == "instance2"
        assert instance1 != instance2  # 不同注册表的实例应该不同