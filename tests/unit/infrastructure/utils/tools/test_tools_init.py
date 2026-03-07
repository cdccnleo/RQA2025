#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施层utils/tools/__init__.py模块测试

测试目标：提升utils/tools/__init__.py的真实覆盖率
实际导入和使用src.infrastructure.utils.tools模块
"""

import pytest


class TestToolsInit:
    """测试tools模块初始化"""
    
    def test_convert_import(self):
        """测试Convert导入"""
        from src.infrastructure.utils.tools import Convert
        
        assert Convert is not None
    
    def test_get_business_date_import(self):
        """测试get_business_date函数导入"""
        from src.infrastructure.utils.tools import get_business_date
        
        assert callable(get_business_date)
    
    def test_is_trading_day_import(self):
        """测试is_trading_day函数导入"""
        from src.infrastructure.utils.tools import is_trading_day
        
        assert callable(is_trading_day)
    
    def test_convert_timezone_import(self):
        """测试convert_timezone函数导入"""
        from src.infrastructure.utils.tools import convert_timezone
        
        assert callable(convert_timezone)
    
    def test_parse_datetime_import(self):
        """测试parse_datetime函数导入"""
        from src.infrastructure.utils.tools import parse_datetime
        
        assert callable(parse_datetime)
    
    def test_calculate_returns_import(self):
        """测试calculate_returns函数导入"""
        from src.infrastructure.utils.tools import calculate_returns
        
        assert callable(calculate_returns)
    
    def test_annualized_volatility_import(self):
        """测试annualized_volatility函数导入"""
        from src.infrastructure.utils.tools import annualized_volatility
        
        assert callable(annualized_volatility)
    
    def test_sharpe_ratio_import(self):
        """测试sharpe_ratio函数导入"""
        from src.infrastructure.utils.tools import sharpe_ratio
        
        assert callable(sharpe_ratio)
    
    def test_normalize_data_import(self):
        """测试normalize_data函数导入"""
        from src.infrastructure.utils.tools import normalize_data
        
        assert callable(normalize_data)
    
    def test_denormalize_data_import(self):
        """测试denormalize_data函数导入"""
        from src.infrastructure.utils.tools import denormalize_data
        
        assert callable(denormalize_data)
    
    def test_safe_file_write_import(self):
        """测试safe_file_write函数导入"""
        from src.infrastructure.utils.tools import safe_file_write
        
        assert callable(safe_file_write)
    
    def test_file_system_import(self):
        """测试FileSystem导入"""
        from src.infrastructure.utils.tools import FileSystem
        
        assert FileSystem is not None
    
    def test_market_aware_retry_import(self):
        """测试MarketAwareRetry导入"""
        from src.infrastructure.utils.tools import MarketAwareRetry
        
        assert MarketAwareRetry is not None
    
    def test_module_all(self):
        """测试模块__all__导出"""
        from src.infrastructure.utils.tools import __all__
        
        assert isinstance(__all__, list)
        assert len(__all__) > 0
        assert "Convert" in __all__
        assert "FileSystem" in __all__
        assert "MarketAwareRetry" in __all__

