#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Factory组件测试 - 简化版

直接测试container/factory_components.py模块
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

# 直接导入factory_components.py
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入factory_components.py文件
    import importlib.util
    factory_components_path = project_root / "src" / "core" / "container" / "factory_components.py"
    spec = importlib.util.spec_from_file_location("factory_components_module", factory_components_path)
    factory_components_module = importlib.util.module_from_spec(spec)
    
    # 处理依赖
    sys.path.insert(0, str(project_root / "src"))
    spec.loader.exec_module(factory_components_module)
    
    # 尝试获取类
    ComponentFactory = getattr(factory_components_module, 'ComponentFactory', None)
    IFactoryComponent = getattr(factory_components_module, 'IFactoryComponent', None)
    FactoryComponent = getattr(factory_components_module, 'FactoryComponent', None)
    FactoryComponentFactory = getattr(factory_components_module, 'FactoryComponentFactory', None)
    
    IMPORTS_AVAILABLE = FactoryComponent is not None
except Exception as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Factory组件模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestFactoryComponent:
    """测试Factory组件"""

    def test_factory_component_initialization(self):
        """测试组件初始化"""
        if FactoryComponent:
            component = FactoryComponent(factory_id=1, component_type="Test")
            assert component is not None
            assert component.factory_id == 1
            assert component.component_type == "Test"

    def test_factory_component_get_factory_id(self):
        """测试获取工厂ID"""
        if FactoryComponent:
            component = FactoryComponent(factory_id=2)
            assert component.get_factory_id() == 2

    def test_factory_component_get_info(self):
        """测试获取组件信息"""
        if FactoryComponent:
            component = FactoryComponent(factory_id=3, component_type="Test")
            info = component.get_info()
            assert isinstance(info, dict)
            assert info['factory_id'] == 3
            assert info['component_type'] == "Test"

    def test_factory_component_process(self):
        """测试处理数据"""
        if FactoryComponent:
            component = FactoryComponent(factory_id=4)
            data = {"key": "value"}
            result = component.process(data)
            assert isinstance(result, dict)
            assert result['factory_id'] == 4

    def test_factory_component_get_status(self):
        """测试获取组件状态"""
        if FactoryComponent:
            component = FactoryComponent(factory_id=5)
            status = component.get_status()
            assert isinstance(status, dict)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestFactoryComponentFactory:
    """测试Factory组件工厂"""

    def test_factory_component_factory_create_component(self):
        """测试创建组件"""
        if FactoryComponentFactory:
            # 使用支持的ID（5或10）
            component = FactoryComponentFactory.create_component(5)
            assert component is not None
            assert isinstance(component, FactoryComponent)
            assert component.factory_id == 5

    def test_factory_component_factory_get_available(self):
        """测试获取可用factory ID"""
        if FactoryComponentFactory:
            available = FactoryComponentFactory.get_available_factorys()
            assert isinstance(available, list)
            assert len(available) > 0

    def test_factory_component_factory_get_info(self):
        """测试获取工厂信息"""
        if FactoryComponentFactory:
            info = FactoryComponentFactory.get_factory_info()
            assert isinstance(info, dict)
            assert 'factory_name' in info

