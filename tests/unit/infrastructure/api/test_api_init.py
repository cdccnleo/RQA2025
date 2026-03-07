#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层api/__init__.py模块测试

测试目标：提升api/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.api模块
"""

import pytest


class TestApiInit:
    """测试api模块初始化"""
    
    def test_module_version(self):
        """测试模块版本"""
        from src.infrastructure.api import __version__
        
        assert isinstance(__version__, str)
        assert __version__ == "2.0.0"
    
    def test_module_author(self):
        """测试模块作者"""
        from src.infrastructure.api import __author__
        
        assert isinstance(__author__, str)
        assert len(__author__) > 0
    
    def test_module_status(self):
        """测试模块状态"""
        from src.infrastructure.api import __status__
        
        assert isinstance(__status__, str)
        assert __status__ == "Production Ready"
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.api import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "__version__" in __all__
        assert "__author__" in __all__
        assert "__status__" in __all__
    
    def test_lazy_import_getattr(self):
        """测试延迟导入__getattr__功能"""
        # 测试__getattr__函数存在
        import src.infrastructure.api as api_module
        
        # 检查__getattr__方法是否存在
        assert hasattr(api_module, '__getattr__')
        
        # 测试访问不存在的属性会触发AttributeError
        with pytest.raises(AttributeError):
            _ = api_module.NonExistentClass

