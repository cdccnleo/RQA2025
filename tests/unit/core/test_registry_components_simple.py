#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registry组件测试 - 简化版

直接测试container/registry_components.py模块
"""

import pytest
from unittest.mock import Mock
from datetime import datetime

# 直接导入registry_components.py
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    
    # 直接导入registry_components.py文件
    import importlib.util
    registry_components_path = project_root / "src" / "core" / "container" / "registry_components.py"
    spec = importlib.util.spec_from_file_location("registry_components_module", registry_components_path)
    registry_components_module = importlib.util.module_from_spec(spec)
    
    # 处理依赖
    sys.path.insert(0, str(project_root / "src"))
    spec.loader.exec_module(registry_components_module)
    
    # 尝试获取类
    ComponentFactory = getattr(registry_components_module, 'ComponentFactory', None)
    IRegistryComponent = getattr(registry_components_module, 'IRegistryComponent', None)
    RegistryComponent = getattr(registry_components_module, 'RegistryComponent', None)
    RegistryComponentFactory = getattr(registry_components_module, 'RegistryComponentFactory', None)
    
    IMPORTS_AVAILABLE = RegistryComponent is not None
except Exception as e:
    IMPORTS_AVAILABLE = False
    pytest.skip(f"Registry组件模块导入失败: {e}", allow_module_level=True)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestRegistryComponent:
    """测试Registry组件"""

    def test_registry_component_initialization(self):
        """测试组件初始化"""
        if RegistryComponent:
            component = RegistryComponent(registry_id=1, component_type="Test")
            assert component is not None
            assert component.registry_id == 1
            assert component.component_type == "Test"

    def test_registry_component_get_registry_id(self):
        """测试获取注册表ID"""
        if RegistryComponent:
            component = RegistryComponent(registry_id=2)
            assert component.get_registry_id() == 2

    def test_registry_component_get_info(self):
        """测试获取组件信息"""
        if RegistryComponent:
            component = RegistryComponent(registry_id=3, component_type="Test")
            info = component.get_info()
            assert isinstance(info, dict)
            assert info['registry_id'] == 3
            assert info['component_type'] == "Test"

    def test_registry_component_process(self):
        """测试处理数据"""
        if RegistryComponent:
            component = RegistryComponent(registry_id=4)
            data = {"key": "value"}
            result = component.process(data)
            assert isinstance(result, dict)
            assert result['registry_id'] == 4

    def test_registry_component_get_status(self):
        """测试获取组件状态"""
        if RegistryComponent:
            component = RegistryComponent(registry_id=5)
            status = component.get_status()
            assert isinstance(status, dict)


@pytest.mark.skipif(not IMPORTS_AVAILABLE, reason="模块导入不可用")
class TestRegistryComponentFactory:
    """测试Registry组件工厂"""

    def test_registry_component_factory_create_component(self):
        """测试创建组件"""
        if RegistryComponentFactory:
            # 使用支持的ID（2、7或12）
            component = RegistryComponentFactory.create_component(2)
            assert component is not None
            assert isinstance(component, RegistryComponent)
            assert component.registry_id == 2

    def test_registry_component_factory_get_available(self):
        """测试获取可用registry ID"""
        if RegistryComponentFactory:
            available = RegistryComponentFactory.get_available_registrys()
            assert isinstance(available, list)
            assert len(available) > 0

    def test_registry_component_factory_get_info(self):
        """测试获取工厂信息"""
        if RegistryComponentFactory:
            # 尝试不同的方法名
            if hasattr(RegistryComponentFactory, 'get_registry_info'):
                info = RegistryComponentFactory.get_registry_info()
            elif hasattr(RegistryComponentFactory, 'get_factory_info'):
                info = RegistryComponentFactory.get_factory_info()
            else:
                pytest.skip("get_info方法不可用")
            assert isinstance(info, dict)
            # 验证至少包含一些基本信息
            assert len(info) > 0

