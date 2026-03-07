#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层utils/interfaces/__init__.py模块测试

测试目标：提升utils/interfaces/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.utils.interfaces模块
"""

import pytest


class TestInterfacesInit:
    """测试interfaces模块初始化"""
    
    def test_module_import(self):
        """测试模块可以导入"""
        import src.infrastructure.utils.interfaces
        
        assert src.infrastructure.utils.interfaces is not None
    
    def test_module_is_package(self):
        """测试模块是一个包"""
        import src.infrastructure.utils.interfaces
        
        assert hasattr(src.infrastructure.utils.interfaces, '__path__')

