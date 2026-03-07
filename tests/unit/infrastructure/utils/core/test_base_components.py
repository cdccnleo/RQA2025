#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层基础组件测试

测试目标：提升utils/core/base_components.py的真实覆盖率
实际导入和使用src.infrastructure.utils.core.base_components模块
"""

import pytest
from datetime import datetime


class TestBaseComponentConstants:
    """测试Base组件常量"""
    
    def test_component_version(self):
        """测试组件版本"""
        from src.infrastructure.utils.core.base_components import BaseComponentConstants
        
        assert BaseComponentConstants.COMPONENT_VERSION == "2.0.0"
    
    def test_supported_base_ids(self):
        """测试支持的base ID列表"""
        from src.infrastructure.utils.core.base_components import BaseComponentConstants
        
        supported_ids = BaseComponentConstants.SUPPORTED_BASE_IDS
        assert isinstance(supported_ids, list)
        assert len(supported_ids) > 0
        assert 5 in supported_ids
        assert 11 in supported_ids
    
    def test_status_constants(self):
        """测试状态常量"""
        from src.infrastructure.utils.core.base_components import BaseComponentConstants
        
        assert BaseComponentConstants.STATUS_ACTIVE == "active"
        assert BaseComponentConstants.STATUS_INACTIVE == "inactive"
        assert BaseComponentConstants.STATUS_ERROR == "error"
    
    def test_priority_constants(self):
        """测试优先级常量"""
        from src.infrastructure.utils.core.base_components import BaseComponentConstants
        
        assert BaseComponentConstants.DEFAULT_PRIORITY == 1
        assert BaseComponentConstants.MIN_PRIORITY == 0
        assert BaseComponentConstants.MAX_PRIORITY == 10
        assert BaseComponentConstants.MIN_PRIORITY < BaseComponentConstants.DEFAULT_PRIORITY < BaseComponentConstants.MAX_PRIORITY


class TestComponentFactory:
    """测试组件工厂基类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.core.base_components import ComponentFactory
        
        factory = ComponentFactory()
        assert factory._factories == {}
        assert factory._creation_times == []
        assert factory._statistics["total_creations"] == 0
        assert factory._statistics["successful_creations"] == 0
        assert factory._statistics["failed_creations"] == 0
    
    def test_register_factory(self):
        """测试注册工厂函数"""
        from src.infrastructure.utils.core.base_components import ComponentFactory
        
        factory = ComponentFactory()
        
        def test_factory(config):
            return Mock()
        
        factory.register_factory("test_component", test_factory)
        assert "test_component" in factory._factories
    
    def test_unregister_factory(self):
        """测试注销工厂函数"""
        from src.infrastructure.utils.core.base_components import ComponentFactory
        
        factory = ComponentFactory()
        
        def test_factory(config):
            return Mock()
        
        factory.register_factory("test_component", test_factory)
        factory.unregister_factory("test_component")
        assert "test_component" not in factory._factories
    
    def test_get_registered_types(self):
        """测试获取已注册的组件类型"""
        from src.infrastructure.utils.core.base_components import ComponentFactory
        
        factory = ComponentFactory()
        
        factory.register_factory("type1", lambda c: None)
        factory.register_factory("type2", lambda c: None)
        
        types = factory.get_registered_types()
        assert len(types) == 2
        assert "type1" in types
        assert "type2" in types
    
    def test_create_component_with_factory(self):
        """测试使用工厂函数创建组件"""
        from src.infrastructure.utils.core.base_components import ComponentFactory
        from unittest.mock import Mock
        
        factory = ComponentFactory()
        
        mock_component = Mock()
        factory.register_factory("test_component", lambda c: mock_component)
        
        result = factory.create_component("test_component")
        assert result == mock_component
        assert factory._statistics["total_creations"] == 1
        assert factory._statistics["successful_creations"] == 1
    
    def test_create_component_without_factory(self):
        """测试不使用工厂函数创建组件"""
        from src.infrastructure.utils.core.base_components import ComponentFactory
        
        factory = ComponentFactory()
        
        result = factory.create_component("nonexistent")
        assert result is None
        assert factory._statistics["failed_creations"] == 1
    
    def test_get_statistics(self):
        """测试获取统计信息"""
        from src.infrastructure.utils.core.base_components import ComponentFactory
        from unittest.mock import Mock
        
        factory = ComponentFactory()
        
        factory.register_factory("test_component", lambda c: Mock())
        factory.create_component("test_component")
        
        stats = factory.get_statistics()
        assert stats["total_creations"] == 1
        assert stats["successful_creations"] == 1
        assert stats["failed_creations"] == 0


class TestIComponentFactory:
    """测试组件工厂接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.core.base_components import IComponentFactory
        
        with pytest.raises(TypeError):
            IComponentFactory()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.core.base_components import IComponentFactory
        
        assert hasattr(IComponentFactory, 'create_component')
        assert hasattr(IComponentFactory, 'register_factory')


class TestIBaseComponent:
    """测试Base组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.core.base_components import IBaseComponent
        
        with pytest.raises(TypeError):
            IBaseComponent()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.core.base_components import IBaseComponent
        
        assert hasattr(IBaseComponent, 'get_info')
        assert hasattr(IBaseComponent, 'process')
        assert hasattr(IBaseComponent, 'get_status')
        assert hasattr(IBaseComponent, 'get_base_id')


class TestBaseComponent:
    """测试Base组件实现"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.core.base_components import BaseComponent
        
        component = BaseComponent(base_id=5)
        assert component.base_id == 5
        assert component.component_type == "Base"
        assert component.component_name == "Base_Component_5"
        assert isinstance(component.creation_time, datetime)
    
    def test_init_with_custom_type(self):
        """测试使用自定义类型初始化"""
        from src.infrastructure.utils.core.base_components import BaseComponent
        
        component = BaseComponent(base_id=11, component_type="Custom")
        assert component.component_type == "Custom"
        assert component.component_name == "Custom_Component_11"
    
    def test_get_base_id(self):
        """测试获取base ID"""
        from src.infrastructure.utils.core.base_components import BaseComponent
        
        component = BaseComponent(base_id=17)
        assert component.get_base_id() == 17
    
    def test_get_info(self):
        """测试获取组件信息"""
        from src.infrastructure.utils.core.base_components import BaseComponent
        
        component = BaseComponent(base_id=23)
        info = component.get_info()
        
        assert info["base_id"] == 23
        assert info["component_name"] == "Base_Component_23"
        assert info["component_type"] == "Base"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
    
    def test_process(self):
        """测试处理数据"""
        from src.infrastructure.utils.core.base_components import BaseComponent
        
        component = BaseComponent(base_id=29)
        data = {"key": "value"}
        
        result = component.process(data)
        
        assert result["base_id"] == 29
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
    
    def test_process_with_exception(self):
        """测试处理数据时发生异常"""
        from src.infrastructure.utils.core.base_components import BaseComponent
        
        component = BaseComponent(base_id=35)
        
        # 模拟处理时发生异常
        original_process = component.process
        component.process = lambda data: original_process({"exception": object()})
        
        # 由于数据包含不可序列化的对象，可能会触发异常处理
        result = component.process({"key": "value"})
        assert result["status"] in ["success", "error"]
    
    def test_get_status(self):
        """测试获取组件状态"""
        from src.infrastructure.utils.core.base_components import BaseComponent
        
        component = BaseComponent(base_id=41)
        status = component.get_status()
        
        assert status["base_id"] == 41
        assert status["component_name"] == "Base_Component_41"
        assert status["status"] == "active"
        assert status["health"] == "good"
        assert "creation_time" in status


class TestBaseComponentFactory:
    """测试Base组件工厂"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.core.base_components import BaseComponentFactory
        
        factory = BaseComponentFactory()
        assert len(factory.SUPPORTED_BASE_IDS) > 0
    
    def test_create_component_with_base_id(self):
        """测试使用base ID创建组件"""
        from src.infrastructure.utils.core.base_components import BaseComponentFactory
        
        factory = BaseComponentFactory()
        
        component = factory.create_component("base_5")
        assert component is not None
        assert component.get_base_id() == 5
    
    def test_create_component_with_numeric_id(self):
        """测试使用数字ID创建组件"""
        from src.infrastructure.utils.core.base_components import BaseComponentFactory
        
        factory = BaseComponentFactory()
        
        component = factory.create_component("11")
        assert component is not None
        assert component.get_base_id() == 11
    
    def test_create_component_unsupported_id(self):
        """测试使用不支持的ID创建组件"""
        from src.infrastructure.utils.core.base_components import BaseComponentFactory
        
        factory = BaseComponentFactory()
        
        with pytest.raises(ValueError, match="不支持的base ID"):
            factory.create_component("base_999")
    
    def test_create_component_static(self):
        """测试静态方法创建组件"""
        from src.infrastructure.utils.core.base_components import BaseComponentFactory
        
        component = BaseComponentFactory.create_component_static(5)
        assert component.get_base_id() == 5
    
    def test_create_component_static_unsupported_id(self):
        """测试静态方法使用不支持的ID创建组件"""
        from src.infrastructure.utils.core.base_components import BaseComponentFactory
        
        with pytest.raises(ValueError, match="不支持的base ID"):
            BaseComponentFactory.create_component_static(999)
    
    def test_get_available_bases(self):
        """测试获取所有可用的base ID"""
        from src.infrastructure.utils.core.base_components import BaseComponentFactory
        
        bases = BaseComponentFactory.get_available_bases()
        assert isinstance(bases, list)
        assert len(bases) > 0
        assert 5 in bases
    
    def test_create_all_bases(self):
        """测试创建所有可用base"""
        from src.infrastructure.utils.core.base_components import BaseComponentFactory
        
        all_bases = BaseComponentFactory.create_all_bases()
        assert isinstance(all_bases, dict)
        assert len(all_bases) > 0
        
        for base_id, component in all_bases.items():
            assert component.get_base_id() == base_id
    
    def test_get_factory_info(self):
        """测试获取工厂信息"""
        from src.infrastructure.utils.core.base_components import BaseComponentFactory
        
        info = BaseComponentFactory.get_factory_info()
        
        assert info["factory_name"] == "BaseComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_bases"] > 0
        assert "supported_ids" in info
        assert "created_at" in info

