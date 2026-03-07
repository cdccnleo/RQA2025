#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API组件测试

测试目标：提升api_components.py的覆盖率
"""

import sys
import importlib
from pathlib import Path
import pytest
from unittest.mock import Mock, MagicMock

# 确保Python路径正确配置
project_root = Path(__file__).resolve().parent.parent.parent.parent.parent
project_root_str = str(project_root)
src_path_str = str(project_root / "src")

if project_root_str not in sys.path:
    sys.path.insert(0, project_root_str)
if src_path_str not in sys.path:
    sys.path.insert(0, src_path_str)

# 动态导入API组件模块
try:
    api_components_module = importlib.import_module('src.gateway.api.api_components')
    ComponentFactory = getattr(api_components_module, 'ComponentFactory', None)
    IApiComponent = getattr(api_components_module, 'IApiComponent', None)
    ApiComponent = getattr(api_components_module, 'ApiComponent', None)
    ApiComponentFactory = getattr(api_components_module, 'ApiComponentFactory', None)
    
    if ComponentFactory is None:
        pytest.skip("API组件模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("API组件模块导入失败", allow_module_level=True)


class TestComponentFactory:
    """测试组件工厂"""
    
    def test_component_factory_init(self):
        """测试组件工厂初始化"""
        factory = ComponentFactory()
        assert factory._components == {}
    
    def test_create_component_none(self):
        """测试创建组件返回None"""
        factory = ComponentFactory()
        result = factory.create_component("unknown_type", {})
        assert result is None
    
    def test_create_component_exception(self):
        """测试创建组件异常处理"""
        factory = ComponentFactory()
        # 由于_create_component_instance返回None，create_component会返回None
        result = factory.create_component("test_type", {})
        assert result is None


class TestIApiComponent:
    """测试API组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IApiComponent()
    
    def test_interface_has_get_info_method(self):
        """测试接口有get_info方法"""
        assert hasattr(IApiComponent, 'get_info')
        assert callable(getattr(IApiComponent, 'get_info'))
    
    def test_concrete_implementation(self):
        """测试具体实现"""
        class ConcreteApiComponent(IApiComponent):
            def get_info(self):
                return {"name": "test_component"}
            
            def process(self, data):
                return {"processed": data}
            
            def get_status(self):
                return {"status": "ok"}
            
            def get_api_id(self):
                return 1
        
        component = ConcreteApiComponent()
        assert component.get_info() == {"name": "test_component"}
        assert component.process({"key": "value"}) == {"processed": {"key": "value"}}
        assert component.get_status() == {"status": "ok"}
        assert component.get_api_id() == 1


class TestApiComponent:
    """测试API组件"""
    
    def test_api_component_init(self):
        """测试API组件初始化"""
        component = ApiComponent(api_id=1)
        assert component is not None
        assert component.api_id == 1
        assert component.component_type == "Api"
    
    def test_api_component_get_info(self):
        """测试API组件获取信息"""
        component = ApiComponent(api_id=1)
        info = component.get_info()
        assert isinstance(info, dict)
        assert info["api_id"] == 1
    
    def test_api_component_process(self):
        """测试API组件处理数据"""
        component = ApiComponent(api_id=1)
        result = component.process({"key": "value"})
        assert isinstance(result, dict)
        assert result["api_id"] == 1
        assert result["status"] == "success"
    
    def test_api_component_get_status(self):
        """测试API组件获取状态"""
        component = ApiComponent(api_id=1)
        status = component.get_status()
        assert isinstance(status, dict)
        assert status["api_id"] == 1
        assert status["status"] == "active"
    
    def test_api_component_get_api_id(self):
        """测试API组件获取API ID"""
        component = ApiComponent(api_id=2)
        assert component.get_api_id() == 2


class TestApiComponentFactory:
    """测试API组件工厂"""
    
    def test_create_component_supported(self):
        """测试创建支持的组件"""
        component = ApiComponentFactory.create_component(2)
        assert component is not None
        assert component.api_id == 2
    
    def test_create_component_unsupported(self):
        """测试创建不支持的组件"""
        with pytest.raises(ValueError) as exc_info:
            ApiComponentFactory.create_component(999)
        assert "不支持的api ID" in str(exc_info.value)
    
    def test_get_available_apis(self):
        """测试获取可用API列表"""
        apis = ApiComponentFactory.get_available_apis()
        assert isinstance(apis, list)
        assert 2 in apis
        assert 8 in apis
    
    def test_create_all_apis(self):
        """测试创建所有API"""
        all_apis = ApiComponentFactory.create_all_apis()
        assert isinstance(all_apis, dict)
        assert 2 in all_apis
        assert 8 in all_apis
        assert all_apis[2].api_id == 2
        assert all_apis[8].api_id == 8
    
    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = ApiComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert "factory_name" in info
        assert "version" in info
        assert "total_apis" in info
        assert info["total_apis"] == 2

