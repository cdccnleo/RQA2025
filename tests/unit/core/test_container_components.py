#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试容器组件

测试目标：提升container_components.py的覆盖率到100%
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

from src.core.container.container_components import (
    ComponentFactory,
    IContainerComponent,
    ContainerComponent,
    ContainerComponentFactory
)


class TestComponentFactory:
    """测试组件工厂"""

    @pytest.fixture
    def component_factory(self):
        """创建组件工厂实例"""
        return ComponentFactory()

    def test_component_factory_initialization(self, component_factory):
        """测试组件工厂初始化"""
        assert hasattr(component_factory, '_components')
        assert isinstance(component_factory._components, dict)
        assert len(component_factory._components) == 0

    def test_create_component_success(self, component_factory):
        """测试成功创建组件"""
        config = {"key": "value"}

        # Mock组件类
        mock_component_class = Mock()
        mock_component = Mock()
        mock_component.initialize.return_value = True
        mock_component_class.return_value = mock_component

        with patch.object(component_factory, '_create_component_instance', return_value=mock_component):
            result = component_factory.create_component("test_type", config)

            assert result == mock_component
            mock_component.initialize.assert_called_once_with(config)

    def test_create_component_initialization_failure(self, component_factory):
        """测试组件初始化失败"""
        config = {"key": "value"}

        mock_component = Mock()
        mock_component.initialize.return_value = False

        with patch.object(component_factory, '_create_component_instance', return_value=mock_component):
            result = component_factory.create_component("test_type", config)

            assert result is None
            mock_component.initialize.assert_called_once_with(config)

    def test_create_component_creation_failure(self, component_factory):
        """测试组件创建失败"""
        config = {"key": "value"}

        with patch.object(component_factory, '_create_component_instance', side_effect=Exception("Creation failed")):
            result = component_factory.create_component("test_type", config)

            assert result is None

    def test_create_component_none_instance(self, component_factory):
        """测试组件实例为None"""
        config = {"key": "value"}

        with patch.object(component_factory, '_create_component_instance', return_value=None):
            result = component_factory.create_component("test_type", config)

            assert result is None

    def test_create_component_instance_default(self, component_factory):
        """测试默认组件实例创建"""
        config = {"key": "value"}

        result = component_factory._create_component_instance("unknown_type", config)

        assert result is None


class TestIContainerComponent:
    """测试容器组件接口"""

    def test_interface_is_abstract(self):
        """测试接口是抽象的"""
        # IContainerComponent是抽象基类，不能直接实例化
        with pytest.raises(TypeError):
            IContainerComponent()


class TestContainerComponent:
    """测试容器组件"""

    @pytest.fixture
    def container_component(self):
        """创建容器组件实例"""
        return ContainerComponent()

    def test_container_component_initialization(self, container_component):
        """测试容器组件初始化"""
        assert hasattr(container_component, '_component_id')
        assert hasattr(container_component, '_created_at')
        assert hasattr(container_component, '_config')
        assert isinstance(container_component._created_at, datetime)

    def test_get_info(self, container_component):
        """测试获取组件信息"""
        info = container_component.get_info()

        assert isinstance(info, dict)
        assert "component_id" in info
        assert "component_type" in info
        assert "created_at" in info
        assert "status" in info
        assert "config" in info

    def test_get_info_with_custom_data(self, container_component):
        """测试获取包含自定义数据的组件信息"""
        # 设置一些自定义属性
        container_component._component_id = "test_id_123"
        container_component._config = {"key": "value"}

        info = container_component.get_info()

        assert info["component_id"] == "test_id_123"
        assert info["config"] == {"key": "value"}
        assert info["component_type"] == "ContainerComponent"

    def test_component_id_generation(self, container_component):
        """测试组件ID生成"""
        # 组件ID应该是一个字符串
        assert isinstance(container_component._component_id, str)
        assert len(container_component._component_id) > 0

    def test_created_at_timestamp(self, container_component):
        """测试创建时间戳"""
        assert isinstance(container_component._created_at, datetime)

        # 创建时间应该在当前时间附近
        now = datetime.now()
        time_diff = abs((now - container_component._created_at).total_seconds())
        assert time_diff < 1  # 应该在1秒内


class TestContainerComponentFactory:
    """测试容器组件工厂"""

    @pytest.fixture
    def component_factory(self):
        """创建容器组件工厂实例"""
        return ContainerComponentFactory()

    def test_factory_initialization(self, component_factory):
        """测试工厂初始化"""
        assert hasattr(component_factory, '_registered_components')
        assert isinstance(component_factory._registered_components, dict)

    def test_register_component(self, component_factory):
        """测试注册组件"""
        component_class = Mock()

        component_factory.register_component("test_component", component_class)

        assert "test_component" in component_factory._registered_components
        assert component_factory._registered_components["test_component"] == component_class

    def test_create_registered_component(self, component_factory):
        """测试创建已注册的组件"""
        component_class = Mock()
        mock_instance = Mock()
        component_class.return_value = mock_instance

        component_factory.register_component("test_component", component_class)

        config = {"param": "value"}
        result = component_factory.create_component("test_component", config)

        assert result == mock_instance
        component_class.assert_called_once_with(config)

    def test_create_unregistered_component(self, component_factory):
        """测试创建未注册的组件"""
        config = {"param": "value"}
        result = component_factory.create_component("unknown_component", config)

        assert result is None

    def test_get_registered_components(self, component_factory):
        """测试获取已注册的组件"""
        component_class1 = Mock()
        component_class2 = Mock()

        component_factory.register_component("comp1", component_class1)
        component_factory.register_component("comp2", component_class2)

        registered = component_factory.get_registered_components()

        assert isinstance(registered, list)
        assert "comp1" in registered
        assert "comp2" in registered

    def test_unregister_component(self, component_factory):
        """测试注销组件"""
        component_class = Mock()
        component_factory.register_component("test_component", component_class)

        assert "test_component" in component_factory._registered_components

        component_factory.unregister_component("test_component")

        assert "test_component" not in component_factory._registered_components

    def test_unregister_nonexistent_component(self, component_factory):
        """测试注销不存在的组件"""
        # 不应该抛出异常
        component_factory.unregister_component("nonexistent")

    def test_is_component_registered(self, component_factory):
        """测试检查组件是否已注册"""
        component_class = Mock()

        assert not component_factory.is_component_registered("test_component")

        component_factory.register_component("test_component", component_class)

        assert component_factory.is_component_registered("test_component")

    def test_get_component_class(self, component_factory):
        """测试获取组件类"""
        component_class = Mock()
        component_factory.register_component("test_component", component_class)

        result = component_factory.get_component_class("test_component")

        assert result == component_class

    def test_get_component_class_not_registered(self, component_factory):
        """测试获取未注册的组件类"""
        result = component_factory.get_component_class("unknown_component")

        assert result is None

    def test_clear_all_components(self, component_factory):
        """测试清空所有组件"""
        component_factory.register_component("comp1", Mock())
        component_factory.register_component("comp2", Mock())

        assert len(component_factory._registered_components) == 2

        component_factory.clear_all_components()

        assert len(component_factory._registered_components) == 0


class TestContainerComponentIntegration:
    """测试容器组件集成场景"""

    def test_complete_component_lifecycle(self):
        """测试完整的组件生命周期"""
        # 1. 创建工厂
        factory = ContainerComponentFactory()

        # 2. 注册组件类
        component_class = Mock()
        mock_instance = Mock()
        mock_instance.get_info.return_value = {"component_id": "test_123", "status": "active"}
        component_class.return_value = mock_instance

        factory.register_component("test_component", component_class)

        # 3. 创建组件实例
        config = {"initial_value": 42}
        component = factory.create_component("test_component", config)

        assert component == mock_instance
        component_class.assert_called_once_with(config)

        # 4. 获取组件信息
        info = component.get_info()

        assert info["component_id"] == "test_123"
        assert info["status"] == "active"

        # 5. 验证注册状态
        assert factory.is_component_registered("test_component")

        # 6. 清理
        factory.clear_all_components()
        assert not factory.is_component_registered("test_component")

    def test_multiple_component_types(self):
        """测试多种组件类型"""
        factory = ContainerComponentFactory()

        # 注册不同类型的组件
        cache_class = Mock()
        db_class = Mock()
        api_class = Mock()

        factory.register_component("cache", cache_class)
        factory.register_component("database", db_class)
        factory.register_component("api", api_class)

        # 验证所有组件都已注册
        registered = factory.get_registered_components()
        assert len(registered) == 3
        assert "cache" in registered
        assert "database" in registered
        assert "api" in registered

        # 测试分别创建不同类型的组件
        cache_config = {"ttl": 300}
        db_config = {"connection_string": "sqlite:///test.db"}
        api_config = {"base_url": "http://api.example.com"}

        factory.create_component("cache", cache_config)
        factory.create_component("database", db_config)
        factory.create_component("api", api_config)

        cache_class.assert_called_once_with(cache_config)
        db_class.assert_called_once_with(db_config)
        api_class.assert_called_once_with(api_config)

    def test_component_error_handling(self):
        """测试组件错误处理"""
        factory = ContainerComponentFactory()

        # 注册一个会抛出异常的组件类
        failing_class = Mock(side_effect=Exception("Component creation failed"))
        factory.register_component("failing_component", failing_class)

        # 尝试创建组件，应该返回None而不是抛出异常
        config = {"param": "value"}
        result = factory.create_component("failing_component", config)

        assert result is None
        failing_class.assert_called_once_with(config)

    def test_component_factory_thread_safety(self):
        """测试组件工厂线程安全"""
        import threading
        import time

        factory = ContainerComponentFactory()
        results = []
        errors = []

        def register_components(thread_id):
            try:
                for i in range(10):
                    component_name = f"thread_{thread_id}_comp_{i}"
                    component_class = Mock()
                    factory.register_component(component_name, component_class)
                    results.append(f"registered_{component_name}")
                    time.sleep(0.001)  # 模拟一些处理时间
            except Exception as e:
                errors.append(str(e))

        # 创建多个线程并发注册组件
        threads = []
        for i in range(5):
            thread = threading.Thread(target=register_components, args=(i,))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        # 验证没有错误发生
        assert len(errors) == 0

        # 验证所有组件都已注册
        registered = factory.get_registered_components()
        assert len(registered) == 50  # 5线程 x 10组件

        # 验证可以创建所有注册的组件
        for component_name in registered:
            result = factory.create_component(component_name, {})
            assert result is not None

    def test_component_configuration_validation(self):
        """测试组件配置验证"""
        factory = ContainerComponentFactory()

        # 创建一个需要验证配置的组件类
        component_class = Mock()
        mock_instance = Mock()
        component_class.return_value = mock_instance

        factory.register_component("validated_component", component_class)

        # 测试有效配置
        valid_config = {
            "required_param": "value",
            "optional_param": 42,
            "nested": {"key": "value"}
        }

        result = factory.create_component("validated_component", valid_config)
        assert result == mock_instance
        component_class.assert_called_once_with(valid_config)

        # 测试空配置
        empty_config = {}
        result = factory.create_component("validated_component", empty_config)
        assert result == mock_instance

        # 测试None配置
        result = factory.create_component("validated_component", None)
        # 应该仍然工作，因为工厂不验证配置内容

    def test_component_performance_monitoring(self):
        """测试组件性能监控"""
        import time

        factory = ContainerComponentFactory()

        # 创建一个模拟耗时组件
        slow_component_class = Mock()
        mock_instance = Mock()
        slow_component_class.return_value = mock_instance

        def slow_init(config):
            time.sleep(0.01)  # 模拟初始化耗时
            return mock_instance

        slow_component_class.side_effect = slow_init

        factory.register_component("slow_component", slow_component_class)

        start_time = time.time()
        result = factory.create_component("slow_component", {"param": "value"})
        end_time = time.time()

        assert result == mock_instance
        assert (end_time - start_time) >= 0.01  # 应该至少耗时10ms


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
