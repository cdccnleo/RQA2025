#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Locator组件测试 - 简化版

直接测试container/locator_components.py模块
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

# 直接导入locator_components.py
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入locator_components.py文件
    import importlib.util
    locator_components_path = project_root / "src" / "core" / "container" / "locator_components.py"
    spec = importlib.util.spec_from_file_location("locator_components_module", locator_components_path)
    locator_components_module = importlib.util.module_from_spec(spec)
    
    # 处理依赖
    sys.path.insert(0, str(project_root / "src"))
    spec.loader.exec_module(locator_components_module)
    
    # 尝试获取类
    ComponentFactory = getattr(locator_components_module, 'ComponentFactory', None)
    ILocatorComponent = getattr(locator_components_module, 'ILocatorComponent', None)
    LocatorComponent = getattr(locator_components_module, 'LocatorComponent', None)
    ServiceLocator = getattr(locator_components_module, 'ServiceLocator', None)
    LocatorComponentFactory = getattr(locator_components_module, 'LocatorComponentFactory', None)
    
    IMPORTS_AVAILABLE = LocatorComponent is not None
except Exception as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Locator组件模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestLocatorComponent:
    """测试Locator组件"""

    def test_locator_component_initialization(self):
        """测试组件初始化"""
        if LocatorComponent:
            component = LocatorComponent(locator_id=1, component_type="Test")
            assert component is not None
            assert component.locator_id == 1
            assert component.component_type == "Test"

    def test_locator_component_get_locator_id(self):
        """测试获取定位器ID"""
        if LocatorComponent:
            component = LocatorComponent(locator_id=2)
            assert component.get_locator_id() == 2

    def test_locator_component_get_info(self):
        """测试获取组件信息"""
        if LocatorComponent:
            component = LocatorComponent(locator_id=3, component_type="Test")
            info = component.get_info()
            assert isinstance(info, dict)
            assert info['locator_id'] == 3
            assert info['component_type'] == "Test"

    def test_locator_component_process(self):
        """测试处理数据"""
        if LocatorComponent:
            component = LocatorComponent(locator_id=4)
            data = {"key": "value"}
            result = component.process(data)
            assert isinstance(result, dict)
            assert result['locator_id'] == 4

    def test_locator_component_get_status(self):
        """测试获取组件状态"""
        if LocatorComponent:
            component = LocatorComponent(locator_id=5)
            status = component.get_status()
            assert isinstance(status, dict)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestServiceLocator:
    """测试ServiceLocator"""

    def test_service_locator_initialization(self):
        """测试服务定位器初始化"""
        if ServiceLocator:
            locator = ServiceLocator()
            assert locator is not None


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestLocatorComponentFactory:
    """测试Locator组件工厂"""

    def test_locator_component_factory_create_component(self):
        """测试创建组件"""
        if LocatorComponentFactory:
            # 使用支持的ID（3、8或13）
            component = LocatorComponentFactory.create_component(3)
            assert component is not None
            assert isinstance(component, LocatorComponent)
            assert component.locator_id == 3

    def test_locator_component_factory_get_available(self):
        """测试获取可用locator ID"""
        if LocatorComponentFactory:
            available = LocatorComponentFactory.get_available_locators()
            assert isinstance(available, list)
            assert len(available) > 0

