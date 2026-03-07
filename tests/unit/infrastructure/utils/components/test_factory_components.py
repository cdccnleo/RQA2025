#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层工厂组件测试

测试目标：提升utils/components/factory_components.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.factory_components模块
"""

import pytest
from datetime import datetime


class TestFactoryComponentConstants:
    """测试工厂组件常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.components.factory_components import FactoryComponentConstants
        
        assert FactoryComponentConstants.COMPONENT_VERSION == "2.0.0"
        assert isinstance(FactoryComponentConstants.SUPPORTED_FACTORY_IDS, list)
        assert len(FactoryComponentConstants.SUPPORTED_FACTORY_IDS) > 0
        assert FactoryComponentConstants.DEFAULT_COMPONENT_TYPE == "Factory"
        assert FactoryComponentConstants.STATUS_ACTIVE == "active"
        assert FactoryComponentConstants.STATUS_INACTIVE == "inactive"
        assert FactoryComponentConstants.STATUS_ERROR == "error"
        assert FactoryComponentConstants.DEFAULT_PRIORITY == 1
        assert FactoryComponentConstants.MIN_PRIORITY == 0
        assert FactoryComponentConstants.MAX_PRIORITY == 10


class TestIFactoryComponent:
    """测试工厂组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.components.factory_components import IFactoryComponent
        
        with pytest.raises(TypeError):
            IFactoryComponent()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.components.factory_components import IFactoryComponent
        
        assert hasattr(IFactoryComponent, 'get_info')
        assert hasattr(IFactoryComponent, 'process')
        assert hasattr(IFactoryComponent, 'get_status')
        assert hasattr(IFactoryComponent, 'get_factory_id')


class TestFactoryComponent:
    """测试工厂组件实现"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.factory_components import FactoryComponent
        
        component = FactoryComponent(factory_id=6)
        assert component.factory_id == 6
        assert component.component_type == "Factory"
        assert component.component_name == "Factory_Component_6"
        assert isinstance(component.creation_time, datetime)
    
    def test_init_with_custom_type(self):
        """测试使用自定义类型初始化"""
        from src.infrastructure.utils.components.factory_components import FactoryComponent
        
        component = FactoryComponent(factory_id=12, component_type="Custom")
        assert component.component_type == "Custom"
        assert component.component_name == "Custom_Component_12"
    
    def test_get_factory_id(self):
        """测试获取factory ID"""
        from src.infrastructure.utils.components.factory_components import FactoryComponent
        
        component = FactoryComponent(factory_id=18)
        assert component.get_factory_id() == 18
    
    def test_get_info(self):
        """测试获取组件信息"""
        from src.infrastructure.utils.components.factory_components import FactoryComponent
        
        component = FactoryComponent(factory_id=24)
        info = component.get_info()
        
        assert info["factory_id"] == 24
        assert info["component_name"] == "Factory_Component_24"
        assert info["component_type"] == "Factory"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
    
    def test_process(self):
        """测试处理数据"""
        from src.infrastructure.utils.components.factory_components import FactoryComponent
        
        component = FactoryComponent(factory_id=30)
        data = {"key": "value"}
        
        result = component.process(data)
        
        assert result["factory_id"] == 30
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
    
    def test_get_status(self):
        """测试获取组件状态"""
        from src.infrastructure.utils.components.factory_components import FactoryComponent
        
        component = FactoryComponent(factory_id=36)
        status = component.get_status()
        
        assert status["factory_id"] == 36
        assert status["component_name"] == "Factory_Component_36"
        assert status["status"] == "active"
        assert "creation_time" in status


class TestFactoryComponentFactory:
    """测试工厂组件工厂"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.factory_components import FactoryComponentFactory
        
        factory = FactoryComponentFactory()
        assert len(factory.SUPPORTED_FACTORY_IDS) > 0
    
    def test_create_component_with_factory_id(self):
        """测试使用factory ID创建组件"""
        from src.infrastructure.utils.components.factory_components import FactoryComponentFactory
        
        factory = FactoryComponentFactory()
        
        component = factory.create_component(6)
        assert component is not None
        assert component.get_factory_id() == 6
    
    def test_create_component_with_numeric_id(self):
        """测试使用数字ID创建组件"""
        from src.infrastructure.utils.components.factory_components import FactoryComponentFactory
        
        factory = FactoryComponentFactory()
        
        component = factory.create_component(12)
        assert component is not None
        assert component.get_factory_id() == 12
    
    def test_create_component_unsupported_id(self):
        """测试使用不支持的ID创建组件"""
        from src.infrastructure.utils.components.factory_components import FactoryComponentFactory
        
        factory = FactoryComponentFactory()
        
        with pytest.raises(ValueError, match="不支持的factory ID"):
            factory.create_component(999)
    
    def test_get_available_factories(self):
        """测试获取所有可用的factory ID"""
        from src.infrastructure.utils.components.factory_components import FactoryComponentFactory
        
        # 注意：方法名是get_available_factorys（不是factories）
        factories = FactoryComponentFactory.get_available_factorys()
        assert isinstance(factories, list)
        assert len(factories) > 0
        assert 6 in factories
    
    def test_create_all_factories(self):
        """测试创建所有可用factory"""
        from src.infrastructure.utils.components.factory_components import FactoryComponentFactory
        
        # 注意：方法名是create_all_factorys（不是factories）
        all_factories = FactoryComponentFactory.create_all_factorys()
        assert isinstance(all_factories, dict)
        assert len(all_factories) > 0
        
        for factory_id, component in all_factories.items():
            assert component.get_factory_id() == factory_id
    
    def test_get_factory_info(self):
        """测试获取工厂信息"""
        from src.infrastructure.utils.components.factory_components import FactoryComponentFactory
        
        info = FactoryComponentFactory.get_factory_info()
        
        assert info["factory_name"] == "FactoryComponentFactory"
        assert info["version"] == "2.0.0"
        # 注意：实际字段名是total_factorys（不是factories）
        assert info["total_factorys"] > 0
        assert "supported_ids" in info
        assert "created_at" in info
        assert "description" in info

