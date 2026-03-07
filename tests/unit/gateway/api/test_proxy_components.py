#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Proxy组件测试

测试目标：提升proxy_components.py的覆盖率
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

# 动态导入代理组件模块
try:
    proxy_components_module = importlib.import_module('src.gateway.api.proxy_components')
    ComponentFactory = getattr(proxy_components_module, 'ComponentFactory', None)
    IProxyComponent = getattr(proxy_components_module, 'IProxyComponent', None)
    ProxyComponent = getattr(proxy_components_module, 'ProxyComponent', None)
    ProxyComponentFactory = getattr(proxy_components_module, 'ProxyComponentFactory', None)
    
    if ComponentFactory is None:
        pytest.skip("代理组件模块不可用", allow_module_level=True)
except ImportError:
    pytest.skip("代理组件模块导入失败", allow_module_level=True)


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


class TestIProxyComponent:
    """测试Proxy组件接口"""
    
    def test_interface_cannot_instantiate(self):
        """测试接口不能直接实例化"""
        with pytest.raises(TypeError):
            IProxyComponent()
    
    def test_interface_has_required_methods(self):
        """测试接口有必需方法"""
        assert hasattr(IProxyComponent, 'get_info')
        assert hasattr(IProxyComponent, 'process')
        assert hasattr(IProxyComponent, 'get_status')
        assert hasattr(IProxyComponent, 'get_proxy_id')
    
    def test_concrete_implementation(self):
        """测试具体实现"""
        class ConcreteProxyComponent(IProxyComponent):
            def get_info(self):
                return {"name": "test_component"}
            
            def process(self, data):
                return {"processed": data}
            
            def get_status(self):
                return {"status": "ok"}
            
            def get_proxy_id(self):
                return 1
        
        component = ConcreteProxyComponent()
        assert component.get_info() == {"name": "test_component"}
        assert component.process({"key": "value"}) == {"processed": {"key": "value"}}
        assert component.get_status() == {"status": "ok"}
        assert component.get_proxy_id() == 1


class TestProxyComponent:
    """测试Proxy组件"""
    
    def test_proxy_component_init(self):
        """测试Proxy组件初始化"""
        component = ProxyComponent(proxy_id=3)
        assert component is not None
        assert component.proxy_id == 3
        assert component.component_type == "Proxy"
    
    def test_proxy_component_get_info(self):
        """测试Proxy组件获取信息"""
        component = ProxyComponent(proxy_id=3)
        info = component.get_info()
        assert isinstance(info, dict)
        assert info["proxy_id"] == 3
    
    def test_proxy_component_process(self):
        """测试Proxy组件处理数据"""
        component = ProxyComponent(proxy_id=3)
        result = component.process({"key": "value"})
        assert isinstance(result, dict)
        assert result["proxy_id"] == 3
    
    def test_proxy_component_get_status(self):
        """测试Proxy组件获取状态"""
        component = ProxyComponent(proxy_id=3)
        status = component.get_status()
        assert isinstance(status, dict)
        assert status["proxy_id"] == 3
    
    def test_proxy_component_get_proxy_id(self):
        """测试Proxy组件获取Proxy ID"""
        component = ProxyComponent(proxy_id=3)
        assert component.get_proxy_id() == 3


class TestProxyComponentFactory:
    """测试Proxy组件工厂"""
    
    def test_create_component_supported(self):
        """测试创建支持的组件"""
        component = ProxyComponentFactory.create_component(3)
        assert component is not None
        assert component.proxy_id == 3
    
    def test_create_component_unsupported(self):
        """测试创建不支持的组件"""
        with pytest.raises(ValueError) as exc_info:
            ProxyComponentFactory.create_component(999)
        assert "不支持的proxy ID" in str(exc_info.value)
    
    def test_get_available_proxys(self):
        """测试获取可用Proxy列表"""
        proxys = ProxyComponentFactory.get_available_proxys()
        assert isinstance(proxys, list)
        assert 3 in proxys or 9 in proxys
    
    def test_create_all_proxys(self):
        """测试创建所有Proxy"""
        all_proxys = ProxyComponentFactory.create_all_proxys()
        assert isinstance(all_proxys, dict)
        assert len(all_proxys) > 0
    
    def test_get_factory_info(self):
        """测试获取工厂信息"""
        info = ProxyComponentFactory.get_factory_info()
        assert isinstance(info, dict)
        assert "factory_name" in info
        assert "version" in info

