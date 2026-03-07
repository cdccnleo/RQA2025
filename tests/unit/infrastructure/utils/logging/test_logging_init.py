#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层utils/logging/__init__.py模块测试

测试目标：提升utils/logging/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.utils.logging模块
"""

import pytest


class TestLoggingInit:
    """测试logging模块初始化"""
    
    def test_unified_logger_import(self):
        """测试UnifiedLogger导入"""
        from src.infrastructure.utils.logging import UnifiedLogger
        
        assert UnifiedLogger is not None
    
    def test_get_unified_logger_import(self):
        """测试get_unified_logger函数导入"""
        from src.infrastructure.utils.logging import get_unified_logger
        
        assert callable(get_unified_logger)
    
    def test_get_logger_import(self):
        """测试get_logger函数导入"""
        from src.infrastructure.utils.logging import get_logger
        
        assert callable(get_logger)
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.utils.logging import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "UnifiedLogger" in __all__
        assert "get_unified_logger" in __all__
        assert "get_logger" in __all__

