#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层utils/components/core/__init__.py模块测试

测试目标：提升utils/components/core/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.utils.components.core模块
"""

import pytest


class TestComponentsCoreInit:
    """测试components/core模块初始化"""
    
    def test_component_factory_import(self):
        """测试ComponentFactory导入"""
        from src.infrastructure.utils.components.core import ComponentFactory
        
        assert ComponentFactory is not None
    
    def test_icomponent_factory_import(self):
        """测试IComponentFactory接口导入"""
        from src.infrastructure.utils.components.core import IComponentFactory
        
        assert IComponentFactory is not None
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.utils.components.core import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "ComponentFactory" in __all__
        assert "IComponentFactory" in __all__
    
    def test_component_factory_usage(self):
        """测试ComponentFactory使用"""
        from src.infrastructure.utils.components.core import ComponentFactory
        
        # ComponentFactory应该是一个类
        assert isinstance(ComponentFactory, type)
    
    def test_icomponent_factory_usage(self):
        """测试IComponentFactory使用"""
        from src.infrastructure.utils.components.core import IComponentFactory
        
        # IComponentFactory应该是一个抽象基类
        assert isinstance(IComponentFactory, type)

