#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Router组件测试

测试目标：提升router_components.py的覆盖率
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

# 动态导入路由组件模块
try:
    router_components_module = importlib.import_module('src.gateway.api.router_components')
    ComponentFactory = getattr(router_components_module, 'ComponentFactory', None)
    IRouterComponent = getattr(router_components_module, 'IRouterComponent', None)
    RouterComponent = getattr(router_components_module, 'RouterComponent', None)
    RouterComponentFactory = getattr(router_components_module, 'RouterComponentFactory', None)
    
    if ComponentFactory is None:
        pytest.skip("路由组件模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("路由组件模块导入失败", allow_module_level=True)


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


class TestIRouterComponent:
    """测试Router组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IRouterComponent()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需方法"""
        assert hasattr(IRouterComponent, 'get_info')
        assert hasattr(IRouterComponent, 'process')
        assert hasattr(IRouterComponent, 'get_status')
        assert hasattr(IRouterComponent, 'get_router_id')
    
    def test_concrete_implementation(self):
        """测试具体实现"""
        class ConcreteRouterComponent(IRouterComponent):
            def get_info(self):
                return {"name": "test_component"}
            
            def process(self, data):
                return {"processed": data}
            
            def get_status(self):
                return {"status": "ok"}
            
            def get_router_id(self):
                return 1
        
        component = ConcreteRouterComponent()
        assert component.get_info() == {"name": "test_component"}
        assert component.process({"key": "value"}) == {"processed": {"key": "value"}}
        assert component.get_status() == {"status": "ok"}
        assert component.get_router_id() == 1


class TestRouterComponent:
    """测试Router组件"""
    
    def test_router_component_init(self):
        """测试Router组件初始化"""
        component = RouterComponent(router_id=4)
        assert component is not None
        assert component.router_id == 4
        assert component.component_type == "Router"
    
    def test_router_component_get_info(self):
        """测试Router组件获取信息"""
        component = RouterComponent(router_id=4)
        info = component.get_info()
        assert isinstance(info, dict)
        assert info["router_id"] == 4
    
    def test_router_component_process(self):
        """测试Router组件处理数据"""
        component = RouterComponent(router_id=4)
        result = component.process({"key": "value"})
        assert isinstance(result, dict)
        assert result["router_id"] == 4
    
    def test_router_component_get_status(self):
        """测试Router组件获取状态"""
        component = RouterComponent(router_id=4)
        status = component.get_status()
        assert isinstance(status, dict)
        assert status["router_id"] == 4
    
    def test_router_component_get_router_id(self):
        """测试Router组件获取Router ID"""
        component = RouterComponent(router_id=4)
        assert component.get_router_id() == 4


class TestRouterComponentFactory:
    """测试Router组件工厂"""
    
    def test_create_component_supported(self):
        """测试创建支持的组件"""
        component = RouterComponentFactory.create_component(4)
        assert component is not None
        assert component.router_id == 4
    
    def test_create_component_unsupported(self):
        """测试创建不支持的组件"""
        with pytest.raises(ValueError) as exc_info:
            RouterComponentFactory.create_component(999)
        assert "不支持的router ID" in str(exc_info.value)
    
    def test_get_available_routers(self):
        """测试获取可用Router列表"""
        routers = RouterComponentFactory.get_available_routers()
        assert isinstance(routers, list)
        assert 4 in routers
    
    def test_create_all_routers(self):
        """测试创建所有Router"""
        all_routers = RouterComponentFactory.create_all_routers()
        assert isinstance(all_routers, dict)
        assert 4 in all_routers
        assert all_routers[4].router_id == 4
    
    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = RouterComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert "factory_name" in info
        assert "version" in info

