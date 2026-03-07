#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层Util组件测试

测试目标：提升utils/components/util_components.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.util_components模块
"""

import pytest
from datetime import datetime


class TestUtilComponentConstants:
    """测试Util组件常量"""
    
    def test_constants(self):
        """测试常量值"""
        from src.infrastructure.utils.components.util_components import UtilComponentConstants
        
        assert UtilComponentConstants.COMPONENT_VERSION == "2.0.0"
        assert isinstance(UtilComponentConstants.SUPPORTED_UTIL_IDS, list)
        assert len(UtilComponentConstants.SUPPORTED_UTIL_IDS) > 0
        assert UtilComponentConstants.DEFAULT_COMPONENT_TYPE == "Util"
        assert UtilComponentConstants.STATUS_ACTIVE == "active"
        assert UtilComponentConstants.STATUS_INACTIVE == "inactive"
        assert UtilComponentConstants.STATUS_ERROR == "error"
        assert UtilComponentConstants.DEFAULT_PRIORITY == 1
        assert UtilComponentConstants.MIN_PRIORITY == 0
        assert UtilComponentConstants.MAX_PRIORITY == 10


class TestIUtilComponent:
    """测试Util组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        from src.infrastructure.utils.components.util_components import IUtilComponent
        
        with pytest.raises(TypeError):
            IUtilComponent()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需的方法"""
        from src.infrastructure.utils.components.util_components import IUtilComponent
        
        assert hasattr(IUtilComponent, 'get_info')
        assert hasattr(IUtilComponent, 'process')
        assert hasattr(IUtilComponent, 'get_status')
        assert hasattr(IUtilComponent, 'get_util_id')


class TestUtilComponent:
    """测试Util组件实现"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.util_components import UtilComponent
        
        component = UtilComponent(util_id=1)
        assert component.util_id == 1
        assert component.component_type == "Util"
        assert component.component_name == "Util_Component_1"
        assert isinstance(component.creation_time, datetime)
    
    def test_init_with_custom_type(self):
        """测试使用自定义类型初始化"""
        from src.infrastructure.utils.components.util_components import UtilComponent
        
        component = UtilComponent(util_id=7, component_type="Custom")
        assert component.component_type == "Custom"
        assert component.component_name == "Custom_Component_7"
    
    def test_get_util_id(self):
        """测试获取util ID"""
        from src.infrastructure.utils.components.util_components import UtilComponent
        
        component = UtilComponent(util_id=13)
        assert component.get_util_id() == 13
    
    def test_get_info(self):
        """测试获取组件信息"""
        from src.infrastructure.utils.components.util_components import UtilComponent
        
        component = UtilComponent(util_id=19)
        info = component.get_info()
        
        assert info["util_id"] == 19
        assert info["component_name"] == "Util_Component_19"
        assert info["component_type"] == "Util"
        assert "creation_time" in info
        assert info["version"] == "2.0.0"
    
    def test_process(self):
        """测试处理数据"""
        from src.infrastructure.utils.components.util_components import UtilComponent
        
        component = UtilComponent(util_id=25)
        data = {"key": "value"}
        
        result = component.process(data)
        
        assert result["util_id"] == 25
        assert result["status"] == "success"
        assert result["input_data"] == data
        assert "processed_at" in result
    
    def test_get_status(self):
        """测试获取组件状态"""
        from src.infrastructure.utils.components.util_components import UtilComponent
        
        component = UtilComponent(util_id=31)
        status = component.get_status()
        
        assert status["util_id"] == 31
        assert status["component_name"] == "Util_Component_31"
        assert status["status"] == "active"
        assert "creation_time" in status


class TestUtilComponentFactory:
    """测试Util组件工厂"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.components.util_components import UtilComponentFactory
        
        factory = UtilComponentFactory()
        assert len(factory.SUPPORTED_UTIL_IDS) > 0
    
    def test_create_component_with_util_id(self):
        """测试使用util ID创建组件"""
        from src.infrastructure.utils.components.util_components import UtilComponentFactory
        
        # UtilComponentFactory.create_component是静态方法，接受util_id（int）
        component = UtilComponentFactory.create_component(1)
        assert component is not None
        assert component.get_util_id() == 1
    
    def test_create_component_with_numeric_id(self):
        """测试使用数字ID创建组件"""
        from src.infrastructure.utils.components.util_components import UtilComponentFactory
        
        # UtilComponentFactory.create_component是静态方法，接受util_id（int）
        component = UtilComponentFactory.create_component(7)
        assert component is not None
        assert component.get_util_id() == 7
    
    def test_create_component_unsupported_id(self):
        """测试使用不支持的ID创建组件"""
        from src.infrastructure.utils.components.util_components import UtilComponentFactory
        
        with pytest.raises(ValueError, match="不支持的util ID"):
            UtilComponentFactory.create_component(999)
    
    def test_create_component_static(self):
        """测试静态方法创建组件"""
        from src.infrastructure.utils.components.util_components import UtilComponentFactory
        
        # UtilComponentFactory.create_component是静态方法
        component = UtilComponentFactory.create_component(1)
        assert component.get_util_id() == 1
    
    def test_get_available_utils(self):
        """测试获取所有可用的util ID"""
        from src.infrastructure.utils.components.util_components import UtilComponentFactory
        
        utils = UtilComponentFactory.get_available_utils()
        assert isinstance(utils, list)
        assert len(utils) > 0
        assert 1 in utils
    
    def test_create_all_utils(self):
        """测试创建所有可用util"""
        from src.infrastructure.utils.components.util_components import UtilComponentFactory
        
        all_utils = UtilComponentFactory.create_all_utils()
        assert isinstance(all_utils, dict)
        assert len(all_utils) > 0
        
        for util_id, component in all_utils.items():
            assert component.get_util_id() == util_id
    
    def test_get_factory_info(self):
        """测试获取工厂信息"""
        from src.infrastructure.utils.components.util_components import UtilComponentFactory
        
        info = UtilComponentFactory.get_factory_info()
        
        assert info["factory_name"] == "UtilComponentFactory"
        assert info["version"] == "2.0.0"
        assert info["total_utils"] > 0
        assert "supported_ids" in info
        assert "created_at" in info

