"""
测试基础设施层基础类

覆盖 BaseInfrastructureComponent, BaseServiceComponent, BaseManagerComponent
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch
from src.infrastructure.base import (
    BaseInfrastructureComponent,
    BaseServiceComponent,
    BaseManagerComponent
)


class TestBaseInfrastructureComponent:
    """BaseInfrastructureComponent 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        # Create a concrete implementation for testing
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize(self):
                pass

            def _shutdown(self):
                pass

        component = ConcreteComponent("test_component")
        assert component.component_name == "test_component"
        assert isinstance(component.start_time, datetime)
        assert component._initialized is False
        assert hasattr(component, '_lock')

    def test_get_status_initialized_false(self):
        """测试获取状态 - 未初始化"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize(self):
                pass

            def _shutdown(self):
                pass

        component = ConcreteComponent("test_component")
        status = component.get_status()

        assert status["component"] == "test_component"
        assert status["status"] == "stopped"
        assert "uptime" in status
        assert "timestamp" in status

    def test_get_status_initialized_true(self):
        """测试获取状态 - 已初始化"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize(self):
                pass

            def _shutdown(self):
                pass

        component = ConcreteComponent("test_component")
        component._initialized = True
        status = component.get_status()

        assert status["component"] == "test_component"
        assert status["status"] == "running"
        assert "uptime" in status
        assert "timestamp" in status

    def test_health_check_successful(self):
        """测试健康检查 - 成功"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize(self):
                pass

            def _shutdown(self):
                pass

        component = ConcreteComponent("test_component")
        result = component.health_check()

        assert result["component"] == "test_component"
        assert result["status"] == "healthy"
        assert "timestamp" in result

    def test_health_check_failed(self):
        """测试健康检查 - 失败"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return False

            def _initialize(self):
                pass

            def _shutdown(self):
                pass

        component = ConcreteComponent("test_component")
        result = component.health_check()

        assert result["component"] == "test_component"
        assert result["status"] == "unhealthy"
        assert "timestamp" in result

    def test_health_check_exception(self):
        """测试健康检查 - 异常"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                raise Exception("Test error")

            def _initialize(self):
                pass

            def _shutdown(self):
                pass

        component = ConcreteComponent("test_component")
        result = component.health_check()

        assert result["component"] == "test_component"
        assert result["status"] == "error"
        assert result["error"] == "Test error"
        assert "timestamp" in result

    def test_initialize(self):
        """测试初始化方法"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize(self):
                pass

            def _shutdown(self):
                pass

        component = ConcreteComponent("test_component")
        component.initialize()
        assert component._initialized is True

    def test_shutdown(self):
        """测试关闭方法"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize(self):
                pass

            def _shutdown(self):
                pass

        component = ConcreteComponent("test_component")
        component._initialized = True

        component.shutdown()
        assert component._initialized is False

    def test_initialize_method(self):
        """测试initialize方法"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize_component(self):
                pass

            def _shutdown_component(self):
                pass

        component = ConcreteComponent("test_component")
        result = component.initialize()
        assert result is True
        assert component._initialized is True

    def test_shutdown_method(self):
        """测试shutdown方法"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize_component(self):
                pass

            def _shutdown_component(self):
                pass

        component = ConcreteComponent("test_component")
        component._initialized = True

        result = component.shutdown()
        assert result is True
        assert component._initialized is False

    def test_abstract_methods(self):
        """测试抽象方法"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize(self):
                pass

            def _shutdown(self):
                pass

        component = ConcreteComponent("test_component")

        # These methods should exist
        assert hasattr(component, '_perform_health_check')
        assert hasattr(component, '_initialize')
        assert hasattr(component, '_shutdown')


class TestBaseServiceComponent:
    """BaseServiceComponent 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        class ConcreteService(BaseServiceComponent):
            def _start_service(self):
                pass

            def _stop_service(self):
                pass

        service = ConcreteService("test_service")
        assert service.component_name == "test_service"
        assert hasattr(service, 'host')
        assert hasattr(service, 'port')
        assert hasattr(service, 'is_running')

    def test_initialize_service(self):
        """测试初始化服务"""
        class ConcreteService(BaseServiceComponent):
            def _start_service(self):
                self.is_running = True

            def _stop_service(self):
                self.is_running = False

        service = ConcreteService("test_service")
        result = service.initialize()
        assert result is True
        assert service._initialized is True
        assert service.is_running is True

    def test_shutdown_service(self):
        """测试关闭服务"""
        class ConcreteService(BaseServiceComponent):
            def _start_service(self):
                self.is_running = True

            def _stop_service(self):
                self.is_running = False

        service = ConcreteService("test_service")
        service._initialized = True
        service.is_running = True

        result = service.shutdown()
        assert result is True
        assert service._initialized is False
        assert service.is_running is False

    def test_health_check_service(self):
        """测试服务健康检查"""
        class ConcreteService(BaseServiceComponent):
            def _start_service(self):
                self.is_running = True

            def _stop_service(self):
                self.is_running = False

        service = ConcreteService("test_service")
        service.is_running = True

        result = service.health_check()
        assert result["component"] == "test_service"
        assert result["status"] == "healthy"


class TestBaseManagerComponent:
    """BaseManagerComponent 单元测试"""

    def test_initialization(self):
        """测试初始化"""
        manager = BaseManagerComponent("test_manager")
        assert manager.component_name == "test_manager"
        assert hasattr(manager, 'max_items')
        assert hasattr(manager, '_items')
        assert isinstance(manager._items, dict)

    def test_add_item(self):
        """测试添加项目"""
        manager = BaseManagerComponent("test_manager")
        item = Mock()

        result = manager.add_item("item1", item)
        assert result is True
        assert "item1" in manager._items
        assert manager._items["item1"] is item

    def test_add_item_at_limit(self):
        """测试添加项目到限制"""
        manager = BaseManagerComponent("test_manager", max_items=1)

        # Add first item
        result1 = manager.add_item("item1", Mock())
        assert result1 is True

        # Try to add second item (should fail)
        result2 = manager.add_item("item2", Mock())
        assert result2 is False

    def test_get_item(self):
        """测试获取项目"""
        manager = BaseManagerComponent("test_manager")
        item = Mock()

        manager.add_item("item1", item)
        result = manager.get_item("item1")
        assert result is item

    def test_get_item_nonexistent(self):
        """测试获取不存在的项目"""
        manager = BaseManagerComponent("test_manager")

        result = manager.get_item("nonexistent")
        assert result is None

    def test_remove_item(self):
        """测试移除项目"""
        manager = BaseManagerComponent("test_manager")
        item = Mock()

        manager.add_item("item1", item)
        assert "item1" in manager._items

        result = manager.remove_item("item1")
        assert result is True
        assert "item1" not in manager._items

    def test_remove_item_nonexistent(self):
        """测试移除不存在的项目"""
        manager = BaseManagerComponent("test_manager")

        result = manager.remove_item("nonexistent")
        assert result is False

    def test_list_items(self):
        """测试列出所有项目"""
        manager = BaseManagerComponent("test_manager")
        item1 = Mock()
        item2 = Mock()

        manager.add_item("item1", item1)
        manager.add_item("item2", item2)

        items = manager.list_items()
        assert len(items) == 2
        assert "item1" in items
        assert "item2" in items

    def test_clear_items(self):
        """测试清空所有项目"""
        manager = BaseManagerComponent("test_manager")

        manager.add_item("item1", Mock())
        manager.add_item("item2", Mock())
        assert len(manager._items) == 2

        manager.clear_items()
        assert len(manager._items) == 0


class TestIntegration:
    """集成测试"""

    def test_component_lifecycle(self):
        """测试组件生命周期"""
        class ConcreteComponent(BaseInfrastructureComponent):
            def _perform_health_check(self):
                return True

            def _initialize_component(self):
                pass

            def _shutdown_component(self):
                pass

        component = ConcreteComponent("test_component")

        # Initially not initialized
        assert not component._initialized

        # Initialize
        result = component.initialize()
        assert result is True
        assert component._initialized

        # Shutdown
        result = component.shutdown()
        assert result is True
        assert not component._initialized

    def test_service_lifecycle(self):
        """测试服务生命周期"""
        class ConcreteService(BaseServiceComponent):
            def _start_service(self):
                self.is_running = True

            def _stop_service(self):
                self.is_running = False

        service = ConcreteService("test_service")

        # Initially not running
        assert not service.is_running

        # Initialize (starts service)
        result = service.initialize()
        assert result is True
        assert service.is_running

        # Shutdown (stops service)
        result = service.shutdown()
        assert result is True
        assert not service.is_running

    def test_manager_operations(self):
        """测试管理器操作"""
        manager = BaseManagerComponent("test_manager")

        # Create mock components (just use any objects)
        comp1 = object()
        comp2 = object()

        # Add items
        manager.add_item("comp1", comp1)
        manager.add_item("comp2", comp2)

        assert len(manager._items) == 2
        assert manager.get_item("comp1") is comp1
        assert manager.get_item("comp2") is comp2

        # Remove one
        manager.remove_item("comp1")
        assert len(manager._items) == 1
        assert manager.get_item("comp1") is None