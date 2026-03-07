#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层optimization/__init__.py模块测试

测试目标：提升optimization/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.optimization模块
"""

import pytest


class TestOptimizationInit:
    """测试optimization模块初始化"""
    
    def test_module_import(self):
        """测试模块可以正常导入"""
        import src.infrastructure.optimization
        
        assert src.infrastructure.optimization is not None
    
    def test_module_has_all(self):
        """测试模块是否有__all__属性"""
        import src.infrastructure.optimization as opt_module
        
        # 检查是否有__all__属性（可能为空列表或未定义）
        if hasattr(opt_module, '__all__'):
            assert isinstance(opt_module.__all__, list)

