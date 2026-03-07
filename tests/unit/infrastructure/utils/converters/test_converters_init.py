#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层utils/converters/__init__.py模块测试

测试目标：提升utils/converters/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.utils.converters模块
"""

import pytest


class TestConvertersInit:
    """测试converters模块初始化"""
    
    def test_query_result_converter_import(self):
        """测试QueryResultConverter导入"""
        from src.infrastructure.utils.converters import QueryResultConverter
        
        assert QueryResultConverter is not None
    
    def test_convert_db_to_unified_import(self):
        """测试convert_db_to_unified函数导入"""
        from src.infrastructure.utils.converters import convert_db_to_unified
        
        assert callable(convert_db_to_unified)
    
    def test_convert_unified_to_db_import(self):
        """测试convert_unified_to_db函数导入"""
        from src.infrastructure.utils.converters import convert_unified_to_db
        
        assert callable(convert_unified_to_db)
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.utils.converters import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "QueryResultConverter" in __all__
        assert "convert_db_to_unified" in __all__
        assert "convert_unified_to_db" in __all__

