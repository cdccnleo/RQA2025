#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层security/__init__.py模块测试

测试目标：提升security/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.security模块，重点测试懒加载机制
"""

import pytest


class TestSecurityInit:
    """测试security模块初始化"""
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.security import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        # 检查一些关键的导出项
        expected_exports = [
            'AuditEventParams',
            'UserCreationParams',
            'AccessCheckParams',
            'AuditEventManager',
            'AuditManager',
            'UserManager',
            'RoleManager',
            'SecurityComponent',
            'BaseSecurityComponent'
        ]
        for export in expected_exports:
            assert export in __all__, f"{export} should be in __all__"
    
    def test_module_has_getattr(self):
        """测试模块有__getattr__方法（懒加载）"""
        import src.infrastructure.security as security_module
        
        assert hasattr(security_module, '__getattr__')
        assert callable(getattr(security_module, '__getattr__'))
    
    def test_lazy_import_attribute_error(self):
        """测试懒加载对不存在属性的处理"""
        import src.infrastructure.security as security_module
        
        # 访问不存在的属性应该触发AttributeError
        with pytest.raises(AttributeError):
            _ = security_module.NonExistentClass
    
    def test_module_import_success(self):
        """测试模块可以正常导入"""
        import src.infrastructure.security
        
        assert src.infrastructure.security is not None

