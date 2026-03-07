#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
基础设施工具层数据加载器组件测试

测试目标：提升utils/adapters/data_loaders.py的真实覆盖率
实际导入和使用src.infrastructure.utils.adapters.data_loaders模块
"""

import pytest
from unittest.mock import MagicMock


class TestDataLoader:
    """测试通用数据加载器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.adapters.data_loaders import DataLoader
        
        loader = DataLoader()
        assert loader.source_type == "csv"
        assert isinstance(loader.data, dict)
    
    def test_init_with_source_type(self):
        """测试使用源类型初始化"""
        from src.infrastructure.utils.adapters.data_loaders import DataLoader
        
        loader = DataLoader(source_type="json")
        assert loader.source_type == "json"
    
    def test_load_csv(self):
        """测试加载CSV"""
        from src.infrastructure.utils.adapters.data_loaders import DataLoader
        
        loader = DataLoader()
        result = loader.load_csv("test.csv")
        
        assert isinstance(result, dict)
    
    def test_load_json(self):
        """测试加载JSON"""
        from src.infrastructure.utils.adapters.data_loaders import DataLoader
        
        loader = DataLoader()
        result = loader.load_json("test.json")
        
        assert isinstance(result, dict)
    
    def test_load_from_db(self):
        """测试从数据库加载"""
        from src.infrastructure.utils.adapters.data_loaders import DataLoader
        
        loader = DataLoader()
        result = loader.load_from_db("SELECT * FROM test")
        
        assert isinstance(result, list)
    
    def test_transform(self):
        """测试转换数据"""
        from src.infrastructure.utils.adapters.data_loaders import DataLoader
        
        loader = DataLoader()
        data = {"key": "value"}
        result = loader.transform(data)
        
        assert result == data


class TestCryptoDataLoader:
    """测试加密货币数据加载器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.adapters.data_loaders import CryptoDataLoader
        
        loader = CryptoDataLoader()
        assert loader is not None
    
    def test_load_data(self):
        """测试加载数据"""
        from src.infrastructure.utils.adapters.data_loaders import CryptoDataLoader
        
        loader = CryptoDataLoader()
        result = loader.load_data("BTC")
        
        assert isinstance(result, dict)


class TestMacroDataLoader:
    """测试宏观经济数据加载器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.adapters.data_loaders import MacroDataLoader
        
        loader = MacroDataLoader()
        assert loader is not None
    
    def test_load_data(self):
        """测试加载数据"""
        from src.infrastructure.utils.adapters.data_loaders import MacroDataLoader
        
        loader = MacroDataLoader()
        result = loader.load_data("GDP")
        
        assert isinstance(result, dict)


class TestOptionsDataLoader:
    """测试期权数据加载器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.adapters.data_loaders import OptionsDataLoader
        
        loader = OptionsDataLoader()
        assert loader is not None
    
    def test_load_data(self):
        """测试加载数据"""
        from src.infrastructure.utils.adapters.data_loaders import OptionsDataLoader
        
        loader = OptionsDataLoader()
        result = loader.load_data("600519")
        
        assert isinstance(result, dict)


class TestBondDataLoader:
    """测试债券数据加载器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.adapters.data_loaders import BondDataLoader
        
        loader = BondDataLoader()
        assert loader is not None
    
    def test_load_data(self):
        """测试加载数据"""
        from src.infrastructure.utils.adapters.data_loaders import BondDataLoader
        
        loader = BondDataLoader()
        result = loader.load_data("BOND001")
        
        assert isinstance(result, dict)


class TestCommodityDataLoader:
    """测试商品数据加载器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.adapters.data_loaders import CommodityDataLoader
        
        loader = CommodityDataLoader()
        assert loader is not None
    
    def test_load_data(self):
        """测试加载数据"""
        from src.infrastructure.utils.adapters.data_loaders import CommodityDataLoader
        
        loader = CommodityDataLoader()
        result = loader.load_data("GOLD")
        
        assert isinstance(result, dict)


class TestForexDataLoader:
    """测试外汇数据加载器类"""
    
    def test_init(self):
        """测试初始化"""
        from src.infrastructure.utils.adapters.data_loaders import ForexDataLoader
        
        loader = ForexDataLoader()
        assert loader is not None
    
    def test_load_data(self):
        """测试加载数据"""
        from src.infrastructure.utils.adapters.data_loaders import ForexDataLoader
        
        loader = ForexDataLoader()
        result = loader.load_data("USDCNY")
        
        assert isinstance(result, dict)

