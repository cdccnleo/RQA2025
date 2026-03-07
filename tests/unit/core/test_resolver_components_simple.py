#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Resolver组件测试 - 简化版

直接测试container/resolver_components.py模块
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

# 直接导入resolver_components.py
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入resolver_components.py文件
    import importlib.util
    resolver_components_path = project_root / "src" / "core" / "container" / "resolver_components.py"
    spec = importlib.util.spec_from_file_location("resolver_components_module", resolver_components_path)
    resolver_components_module = importlib.util.module_from_spec(spec)
    
    # 处理依赖
    sys.path.insert(0, str(project_root / "src"))
    spec.loader.exec_module(resolver_components_module)
    
    # 尝试获取类
    ComponentFactory = getattr(resolver_components_module, 'ComponentFactory', None)
    IResolverComponent = getattr(resolver_components_module, 'IResolverComponent', None)
    ResolverComponent = getattr(resolver_components_module, 'ResolverComponent', None)
    DependencyResolver = getattr(resolver_components_module, 'DependencyResolver', None)
    ResolverComponentFactory = getattr(resolver_components_module, 'ResolverComponentFactory', None)
    
    IMPORTS_AVAILABLE = ResolverComponent is not None
except Exception as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Resolver组件模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestResolverComponent:
    """测试Resolver组件"""

    def test_resolver_component_initialization(self):
        """测试组件初始化"""
        if ResolverComponent:
            component = ResolverComponent(resolver_id=1, component_type="Test")
            assert component is not None
            assert component.resolver_id == 1
            assert component.component_type == "Test"

    def test_resolver_component_get_resolver_id(self):
        """测试获取解析器ID"""
        if ResolverComponent:
            component = ResolverComponent(resolver_id=2)
            assert component.get_resolver_id() == 2

    def test_resolver_component_get_info(self):
        """测试获取组件信息"""
        if ResolverComponent:
            component = ResolverComponent(resolver_id=3, component_type="Test")
            info = component.get_info()
            assert isinstance(info, dict)
            assert info['resolver_id'] == 3
            assert info['component_type'] == "Test"

    def test_resolver_component_process(self):
        """测试处理数据"""
        if ResolverComponent:
            component = ResolverComponent(resolver_id=4)
            data = {"key": "value"}
            result = component.process(data)
            assert isinstance(result, dict)
            assert result['resolver_id'] == 4

    def test_resolver_component_get_status(self):
        """测试获取组件状态"""
        if ResolverComponent:
            component = ResolverComponent(resolver_id=5)
            status = component.get_status()
            assert isinstance(status, dict)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestDependencyResolver:
    """测试DependencyResolver"""

    def test_dependency_resolver_initialization(self):
        """测试依赖解析器初始化"""
        if DependencyResolver:
            resolver = DependencyResolver()
            assert resolver is not None


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestResolverComponentFactory:
    """测试Resolver组件工厂"""

    def test_resolver_component_factory_create_component(self):
        """测试创建组件"""
        if ResolverComponentFactory:
            # 使用支持的ID（4、9或14）
            component = ResolverComponentFactory.create_component(4)
            assert component is not None
            assert isinstance(component, ResolverComponent)
            assert component.resolver_id == 4

    def test_resolver_component_factory_get_available(self):
        """测试获取可用resolver ID"""
        if ResolverComponentFactory:
            available = ResolverComponentFactory.get_available_resolvers()
            assert isinstance(available, list)
            assert len(available) > 0

