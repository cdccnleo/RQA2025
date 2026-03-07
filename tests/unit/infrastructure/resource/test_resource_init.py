#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层resource/__init__.py模块测试

测试目标：提升resource/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.resource模块
"""

import pytest


class TestResourceInit:
    """测试resource模块初始化"""
    
    def test_resource_manager_import(self):
        """测试ResourceManager导入"""
        from src.infrastructure.resource import ResourceManager
        
        assert ResourceManager is not None
    
    def test_gpu_manager_import(self):
        """测试GPUManager导入"""
        from src.infrastructure.resource import GPUManager
        
        assert GPUManager is not None
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.resource import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "ResourceManager" in __all__
        assert "GPUManager" in __all__
    
    def test_module_version(self):
        """测试模块版本"""
        from src.infrastructure.resource import __version__
        
        assert isinstance(__version__, str)
        assert __version__ == "1.0.0"

