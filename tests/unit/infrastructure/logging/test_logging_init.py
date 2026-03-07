#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层logging/__init__.py模块测试

测试目标：提升logging/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.logging模块
"""

import pytest


class TestLoggingInit:
    """测试logging模块初始化"""
    
    def test_get_infrastructure_logger_function(self):
        """测试get_infrastructure_logger函数"""
        from src.infrastructure.logging import get_infrastructure_logger
        
        assert callable(get_infrastructure_logger)
        # 测试函数调用
        logger = get_infrastructure_logger()
        assert logger is not None
    
    def test_get_infrastructure_logger_with_name(self):
        """测试get_infrastructure_logger函数（带名称）"""
        from src.infrastructure.logging import get_infrastructure_logger
        
        logger = get_infrastructure_logger("test_logger")
        assert logger is not None
        assert logger.name == "test_logger"
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.logging import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "get_infrastructure_logger" in __all__
    
    def test_module_version(self):
        """测试模块版本"""
        from src.infrastructure.logging import __version__
        
        assert isinstance(__version__, str)
        assert __version__ == "2.0.0"

