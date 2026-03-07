#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层Helper组件测试

测试目标：提升utils/components/helper_components.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.helper_components模块
"""

import pytest
from datetime import datetime


class TestHelperComponentConstants:
    """测试Helper组件常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.components.helper_components import HelperComponentConstants
        
        assert HelperComponentConstants.COMPONENT_VERSION == "2.0.0"
        assert isinstance(HelperComponentConstants.SUPPORTED_HELPER_IDS, list)
        assert len(HelperComponentConstants.SUPPORTED_HELPER_IDS) > 0
        assert HelperComponentConstants.DEFAULT_COMPONENT_TYPE == "Helper"
        assert HelperComponentConstants.STATUS_ACTIVE == "active"
        assert HelperComponentConstants.STATUS_INACTIVE == "inactive"
        assert HelperComponentConstants.STATUS_ERROR == "error"
        assert HelperComponentConstants.DEFAULT_PRIORITY == 1
        assert HelperComponentConstants.MIN_PRIORITY == 0
        assert HelperComponentConstants.MAX_PRIORITY == 10


class TestIHelperComponent:
    """测试Helper组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.components.helper_components import IHelperComponent
        
        with pytest.raises(TypeError):
            IHelperComponent()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.components.helper_components import IHelperComponent
        
        assert hasattr(IHelperComponent, 'get_info')
        assert hasattr(IHelperComponent, 'process')
        assert hasattr(IHelperComponent, 'get_status')
        assert hasattr(IHelperComponent, 'get_helper_id')


class TestHelperComponent:
    """测试Helper组件实现"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.helper_components import HelperComponent
        
        component = HelperComponent(helper_id=2)
        assert component.helper_id == 2
        assert component.component_type == "Helper"
        assert component.component_name == "Helper_Component_2"
        assert isinstance(component.creation_time, datetime)
    
    def test_init_with_custom_type(self):
        """测试使用自定义类型初始化"""
        from src.infrastructure.utils.components.helper_components import HelperComponent
        
        component = HelperComponent(helper_id=8, component_type="Custom")
        assert component.component_type == "Custom"
        assert component.component_name == "Custom_Component_8"
    
    def test_get_helper_id(self):
        """测试获取helper ID"""
        from src.infrastructure.utils.components.helper_components import HelperComponent
        
        component = HelperComponent(helper_id=14)
        assert component.get_helper_id() == 14
    
    def test_get_info(self):
        """测试获取组件信息"""
        from src.infrastructure.utils.components.helper_components import HelperComponent
        
        component = HelperComponent(helper_id=20)
        info = component.get_info()
        
        assert info["helper_id"] == 20
        assert info["component_name"] == "Helper_Component_20"
        assert info["component_type"] == "Helper"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
    
    def test_process(self):
        """测试处理数据"""
        from src.infrastructure.utils.components.helper_components import HelperComponent
        
        component = HelperComponent(helper_id=26)
        data = {"key": "value"}
        
        result = component.process(data)
        
        assert result["helper_id"] == 26
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
    
    def test_get_status(self):
        """测试获取组件状态"""
        from src.infrastructure.utils.components.helper_components import HelperComponent
        
        component = HelperComponent(helper_id=32)
        status = component.get_status()
        
        assert status["helper_id"] == 32
        assert status["component_name"] == "Helper_Component_32"
        assert status["status"] == "active"
        assert "creation_time" in status


class TestHelperComponentFactory:
    """测试Helper组件工厂"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.helper_components import HelperComponentFactory
        
        factory = HelperComponentFactory()
        assert len(factory.SUPPORTED_HELPER_IDS) > 0
    
    def test_create_component_with_helper_id(self):
        """测试使用helper ID创建组件"""
        from src.infrastructure.utils.components.helper_components import HelperComponentFactory
        
        factory = HelperComponentFactory()
        
        component = factory.create_component("helper_2")
        assert component is not None
        assert component.get_helper_id() == 2
    
    def test_create_component_with_numeric_id(self):
        """测试使用数字ID创建组件"""
        from src.infrastructure.utils.components.helper_components import HelperComponentFactory
        
        factory = HelperComponentFactory()
        
        component = factory.create_component("8")
        assert component is not None
        assert component.get_helper_id() == 8
    
    def test_create_component_unsupported_id(self):
        """测试使用不支持的ID创建组件"""
        from src.infrastructure.utils.components.helper_components import HelperComponentFactory
        
        factory = HelperComponentFactory()
        
        with pytest.raises(ValueError, match="不支持的helper ID"):
            factory.create_component("helper_999")
    
    def test_create_component_static(self):
        """测试静态方法创建组件"""
        from src.infrastructure.utils.components.helper_components import HelperComponentFactory
        
        # HelperComponentFactory使用静态create_component方法（不是create_component_static）
        component = HelperComponentFactory.create_component(2)
        assert component.get_helper_id() == 2
    
    def test_get_available_helpers(self):
        """测试获取所有可用的helper ID"""
        from src.infrastructure.utils.components.helper_components import HelperComponentFactory
        
        helpers = HelperComponentFactory.get_available_helpers()
        assert isinstance(helpers, list)
        assert len(helpers) > 0
        assert 2 in helpers
    
    def test_create_all_helpers(self):
        """测试创建所有可用helper"""
        from src.infrastructure.utils.components.helper_components import HelperComponentFactory
        
        all_helpers = HelperComponentFactory.create_all_helpers()
        assert isinstance(all_helpers, dict)
        assert len(all_helpers) > 0
        
        for helper_id, component in all_helpers.items():
            assert component.get_helper_id() == helper_id
    
    def test_get_factory_info(self):
        """测试获取工厂信息"""
        from src.infrastructure.utils.components.helper_components import HelperComponentFactory
        
        info = HelperComponentFactory.get_factory_info()
        
        assert info["factory_name"] == "HelperComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_helpers"] > 0
        assert "supported_ids" in info
        assert "created_at" in info

