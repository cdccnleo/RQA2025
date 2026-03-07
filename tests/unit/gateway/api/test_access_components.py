#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Access组件测试

测试目标：提升access_components.py的覆盖率
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

# 动态导入访问组件模块
try:
    access_components_module = importlib.import_module('src.gateway.api.access_components')
    ComponentFactory = getattr(access_components_module, 'ComponentFactory', None)
    IAccessComponent = getattr(access_components_module, 'IAccessComponent', None)
    AccessComponent = getattr(access_components_module, 'AccessComponent', None)
    AccessComponentFactory = getattr(access_components_module, 'AccessComponentFactory', None)
    
    if ComponentFactory is None:
        pytest.skip("访问组件模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("访问组件模块导入失败", allow_module_level=True)


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


class TestIAccessComponent:
    """测试Access组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IAccessComponent()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需方法"""
        assert hasattr(IAccessComponent, 'get_info')
        assert hasattr(IAccessComponent, 'process')
        assert hasattr(IAccessComponent, 'get_status')
        assert hasattr(IAccessComponent, 'get_access_id')
    
    def test_concrete_implementation(self):
        """测试具体实现"""
        class ConcreteAccessComponent(IAccessComponent):
            def get_info(self):
                return {"name": "test_component"}
            
            def process(self, data):
                return {"processed": data}
            
            def get_status(self):
                return {"status": "ok"}
            
            def get_access_id(self):
                return 1
        
        component = ConcreteAccessComponent()
        assert component.get_info() == {"name": "test_component"}
        assert component.process({"key": "value"}) == {"processed": {"key": "value"}}
        assert component.get_status() == {"status": "ok"}
        assert component.get_access_id() == 1


class TestAccessComponent:
    """测试Access组件"""
    
    def test_access_component_init(self):
        """测试Access组件初始化"""
        component = AccessComponent(access_id=6)
        assert component is not None
        assert component.access_id == 6
        assert component.component_type == "Access"
    
    def test_access_component_get_info(self):
        """测试Access组件获取信息"""
        component = AccessComponent(access_id=6)
        info = component.get_info()
        assert isinstance(info, dict)
        assert info["access_id"] == 6
    
    def test_access_component_process(self):
        """测试Access组件处理数据"""
        component = AccessComponent(access_id=6)
        result = component.process({"key": "value"})
        assert isinstance(result, dict)
        assert result["access_id"] == 6
    
    def test_access_component_get_status(self):
        """测试Access组件获取状态"""
        component = AccessComponent(access_id=6)
        status = component.get_status()
        assert isinstance(status, dict)
        assert status["access_id"] == 6
    
    def test_access_component_get_access_id(self):
        """测试Access组件获取Access ID"""
        component = AccessComponent(access_id=6)
        assert component.get_access_id() == 6
        assert component.access_id == 6


class TestAccessComponentFactory:
    """测试Access组件工厂"""
    
    def test_create_component_supported(self):
        """测试创建支持的组件"""
        component = AccessComponentFactory.create_component(6)
        assert component is not None
        assert component.access_id == 6
    
    def test_create_component_unsupported(self):
        """测试创建不支持的组件"""
        with pytest.raises(ValueError) as exc_info:
            AccessComponentFactory.create_component(999)
        assert "不支持的access ID" in str(exc_info.value)
    
    def test_get_available_accesss(self):
        """测试获取可用Access列表"""
        accesss = AccessComponentFactory.get_available_accesss()
        assert isinstance(accesss, list)
        assert 6 in accesss
    
    def test_create_all_accesss(self):
        """测试创建所有Access"""
        all_accesss = AccessComponentFactory.create_all_accesss()
        assert isinstance(all_accesss, dict)
        assert 6 in all_accesss
        assert all_accesss[6].access_id == 6
    
    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = AccessComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert "factory_name" in info
        assert "version" in info

