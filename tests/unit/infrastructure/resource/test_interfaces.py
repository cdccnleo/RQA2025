#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
接口定义测试
测试interfaces.py中的接口定义
"""

from tests.fixtures.infrastructure_mocks import StandardMockBuilder, create_standard_mock
import pytest
from abc import ABC, abstractmethod
from typing import Any, Dict

from src.infrastructure.resource.core.interfaces import IResourceComponent


class TestIResourceComponent:
    """测试IResourceComponent接口"""

    def test_iresource_component_is_abstract(self):
        """测试IResourceComponent是抽象类"""
        # 抽象类不能直接实例化
        with pytest.raises(TypeError):
            IResourceComponent()

    def test_iresource_component_inherits_from_abc(self):
        """测试IResourceComponent继承自ABC"""
        assert issubclass(IResourceComponent, ABC)

    def test_iresource_component_has_abstract_methods(self):
        """测试IResourceComponent具有抽象方法"""
        # 检查是否有抽象方法
        abstract_methods = IResourceComponent.__abstractmethods__

        expected_methods = {'initialize', 'get_status', 'shutdown'}
        assert abstract_methods == expected_methods

    def test_iresource_component_abstract_method_signatures(self):
        """测试IResourceComponent抽象方法的签名"""
        import inspect

        # 检查initialize方法
        init_method = getattr(IResourceComponent, 'initialize', None)
        assert init_method is not None
        assert callable(init_method)

        # 检查get_status方法
        status_method = getattr(IResourceComponent, 'get_status', None)
        assert status_method is not None
        assert callable(status_method)

        # 检查shutdown方法
        shutdown_method = getattr(IResourceComponent, 'shutdown', None)
        assert shutdown_method is not None
        assert callable(shutdown_method)


class ConcreteResourceComponent(IResourceComponent):
    """IResourceComponent的具体实现，用于测试"""

    def __init__(self):
        self.initialized = False
        self.status_data = {"status": "ok"}

    def initialize(self, config: Dict[str, Any]) -> bool:
        """实现initialize方法"""
        self.initialized = True
        return True

    def get_status(self) -> Dict[str, Any]:
        """实现get_status方法"""
        return self.status_data

    def shutdown(self) -> None:
        """实现shutdown方法"""
        self.initialized = False


class TestConcreteResourceComponent:
    """测试具体实现类"""

    def setup_method(self):
        """测试前准备"""
        self.component = ConcreteResourceComponent()

    def test_concrete_implementation_can_be_instantiated(self):
        """测试具体实现可以被实例化"""
        assert isinstance(self.component, IResourceComponent)
        assert isinstance(self.component, ConcreteResourceComponent)

    def test_initialize_method(self):
        """测试initialize方法的实现"""
        config = {"key": "value"}

        result = self.component.initialize(config)

        assert result is True
        assert self.component.initialized is True

    def test_get_status_method(self):
        """测试get_status方法的实现"""
        status = self.component.get_status()

        assert isinstance(status, dict)
        assert "status" in status
        assert status["status"] == "ok"

    def test_shutdown_method(self):
        """测试shutdown方法的实现"""
        # 先初始化
        self.component.initialize({})

        # 然后关闭
        self.component.shutdown()

        assert self.component.initialized is False

    def test_interface_compliance(self):
        """测试接口合规性"""
        # 验证所有抽象方法都被实现
        assert hasattr(self.component, 'initialize')
        assert hasattr(self.component, 'get_status')
        assert hasattr(self.component, 'shutdown')

        # 验证方法是可调用的
        assert callable(self.component.initialize)
        assert callable(self.component.get_status)
        assert callable(self.component.shutdown)

        # 验证返回类型正确
        result = self.component.initialize({})
        assert isinstance(result, bool)

        status = self.component.get_status()
        assert isinstance(status, dict)

        # shutdown 不返回值，但不应该抛出异常
        self.component.shutdown()


class TestInterfaceDocumentation:
    """测试接口文档"""

    def test_interface_has_docstring(self):
        """测试接口有文档字符串"""
        assert IResourceComponent.__doc__ is not None
        assert len(IResourceComponent.__doc__.strip()) > 0

    def test_abstract_methods_have_docstrings(self):
        """测试抽象方法有文档字符串"""
        methods = ['initialize', 'get_status', 'shutdown']

        for method_name in methods:
            method = getattr(IResourceComponent, method_name)
            assert method.__doc__ is not None
            assert len(method.__doc__.strip()) > 0


class TestInterfaceInheritance:
    """测试接口继承"""

    def test_interface_inheritance_chain(self):
        """测试接口继承链"""
        # IResourceComponent应该继承自ABC
        assert issubclass(IResourceComponent, ABC)

        # 具体实现应该继承自IResourceComponent
        assert issubclass(ConcreteResourceComponent, IResourceComponent)

        # 实例应该是IResourceComponent的实例
        component = ConcreteResourceComponent()
        assert isinstance(component, IResourceComponent)

    def test_multiple_inheritance_compatibility(self):
        """测试多重继承兼容性"""

        class AnotherInterface(ABC):
            @abstractmethod
            def another_method(self) -> str:
                pass

        class MultiInheritanceComponent(IResourceComponent, AnotherInterface):
            def initialize(self, config: Dict[str, Any]) -> bool:
                return True

            def get_status(self) -> Dict[str, Any]:
                return {"status": "ok"}

            def shutdown(self) -> None:
                pass

            def another_method(self) -> str:
                return "another"

        # 测试多重继承
        component = MultiInheritanceComponent()
        assert isinstance(component, IResourceComponent)
        assert isinstance(component, AnotherInterface)

        # 测试所有方法
        assert component.initialize({}) is True
        assert component.get_status() == {"status": "ok"}
        assert component.another_method() == "another"
        component.shutdown()  # 不应该抛出异常
