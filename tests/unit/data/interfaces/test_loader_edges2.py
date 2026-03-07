"""
边界测试：loader.py
测试边界情况和异常场景
"""
import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


import pytest
from datetime import datetime
from unittest.mock import Mock

from src.data.interfaces.loader import (
    IDataLoader,
    IMarketDataLoader,
    BaseDataLoader,
    StockDataLoader,
    CryptoDataLoader,
    ForexDataLoader,
    BondDataLoader,
    OptionsDataLoader,
    MacroDataLoader,
    CommodityDataLoader,
    IndexDataLoader,
    get_data_loader,
)


def test_idata_loader_abstract():
    """测试 IDataLoader（抽象接口）"""
    # 抽象类不能直接实例化
    with pytest.raises(TypeError):
        IDataLoader()


def test_imarket_data_loader_abstract():
    """测试 IMarketDataLoader（抽象接口）"""
    # 抽象类不能直接实例化
    with pytest.raises(TypeError):
        IMarketDataLoader()


def test_base_data_loader_init():
    """测试 BaseDataLoader（初始化）"""
    loader = BaseDataLoader("test_loader")
    
    assert loader.name == "test_loader"
    assert loader.is_connected is False


def test_base_data_loader_load_data():
    """测试 BaseDataLoader（加载数据）"""
    loader = BaseDataLoader("test_loader")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    result = loader.load_data(["AAPL", "MSFT"], start_date, end_date)
    
    assert result["symbols"] == ["AAPL", "MSFT"]
    assert result["start_date"] == start_date
    assert result["end_date"] == end_date
    assert result["status"] == "success"
    assert "data" in result


def test_base_data_loader_load_data_empty_symbols():
    """测试 BaseDataLoader（加载数据，空符号列表）"""
    loader = BaseDataLoader("test_loader")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    result = loader.load_data([], start_date, end_date)
    
    assert result["symbols"] == []
    assert result["status"] == "success"


def test_base_data_loader_load_data_with_kwargs():
    """测试 BaseDataLoader（加载数据，带额外参数）"""
    loader = BaseDataLoader("test_loader")
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 1, 31)
    
    result = loader.load_data(["AAPL"], start_date, end_date, interval="1h", source="yahoo")
    
    assert result["symbols"] == ["AAPL"]
    assert result["status"] == "success"


def test_base_data_loader_get_available_symbols():
    """测试 BaseDataLoader（获取可用符号）"""
    loader = BaseDataLoader("test_loader")
    
    symbols = loader.get_available_symbols()
    
    assert symbols == []


def test_base_data_loader_get_data_info():
    """测试 BaseDataLoader（获取数据信息）"""
    loader = BaseDataLoader("test_loader")
    
    info = loader.get_data_info("AAPL")
    
    assert info["symbol"] == "AAPL"
    assert info["available"] is False


def test_base_data_loader_get_data_info_empty_symbol():
    """测试 BaseDataLoader（获取数据信息，空符号）"""
    loader = BaseDataLoader("test_loader")
    
    info = loader.get_data_info("")
    
    assert info["symbol"] == ""
    assert info["available"] is False


def test_stock_data_loader_init():
    """测试 StockDataLoader（初始化）"""
    loader = StockDataLoader("stock_loader")
    
    assert loader.name == "stock_loader"
    assert isinstance(loader, BaseDataLoader)


def test_crypto_data_loader_init():
    """测试 CryptoDataLoader（初始化）"""
    loader = CryptoDataLoader("crypto_loader")
    
    assert loader.name == "crypto_loader"
    assert isinstance(loader, BaseDataLoader)


def test_forex_data_loader_init():
    """测试 ForexDataLoader（初始化）"""
    loader = ForexDataLoader("forex_loader")
    
    assert loader.name == "forex_loader"
    assert isinstance(loader, BaseDataLoader)


def test_bond_data_loader_init():
    """测试 BondDataLoader（初始化）"""
    loader = BondDataLoader("bond_loader")
    
    assert loader.name == "bond_loader"
    assert isinstance(loader, BaseDataLoader)


def test_options_data_loader_init():
    """测试 OptionsDataLoader（初始化）"""
    loader = OptionsDataLoader("options_loader")
    
    assert loader.name == "options_loader"
    assert isinstance(loader, BaseDataLoader)


def test_macro_data_loader_init():
    """测试 MacroDataLoader（初始化）"""
    loader = MacroDataLoader("macro_loader")
    
    assert loader.name == "macro_loader"
    assert isinstance(loader, BaseDataLoader)


def test_commodity_data_loader_init():
    """测试 CommodityDataLoader（初始化）"""
    loader = CommodityDataLoader("commodity_loader")
    
    assert loader.name == "commodity_loader"
    assert isinstance(loader, BaseDataLoader)


def test_index_data_loader_init():
    """测试 IndexDataLoader（初始化）"""
    loader = IndexDataLoader("index_loader")
    
    assert loader.name == "index_loader"
    assert isinstance(loader, BaseDataLoader)


def test_get_data_loader_stock():
    """测试 get_data_loader（股票类型）"""
    loader = get_data_loader("stock")
    
    assert isinstance(loader, StockDataLoader)


def test_get_data_loader_crypto():
    """测试 get_data_loader（加密货币类型）"""
    loader = get_data_loader("crypto")
    
    assert isinstance(loader, CryptoDataLoader)


def test_get_data_loader_forex():
    """测试 get_data_loader（外汇类型）"""
    loader = get_data_loader("forex")
    
    assert isinstance(loader, ForexDataLoader)


def test_get_data_loader_bond():
    """测试 get_data_loader（债券类型）"""
    loader = get_data_loader("bond")
    
    assert isinstance(loader, BondDataLoader)


def test_get_data_loader_options():
    """测试 get_data_loader（期权类型）"""
    loader = get_data_loader("options")
    
    assert isinstance(loader, OptionsDataLoader)


def test_get_data_loader_macro():
    """测试 get_data_loader（宏观经济类型）"""
    loader = get_data_loader("macro")
    
    assert isinstance(loader, MacroDataLoader)


def test_get_data_loader_commodity():
    """测试 get_data_loader（商品类型）"""
    loader = get_data_loader("commodity")
    
    assert isinstance(loader, CommodityDataLoader)


def test_get_data_loader_index():
    """测试 get_data_loader（指数类型）"""
    loader = get_data_loader("index")
    
    assert isinstance(loader, IndexDataLoader)


def test_get_data_loader_invalid():
    """测试 get_data_loader（无效类型）"""
    loader = get_data_loader("invalid_type")
    
    # 无效类型应该返回默认的 BaseDataLoader
    assert isinstance(loader, BaseDataLoader)


def test_get_data_loader_empty():
    """测试 get_data_loader（空字符串）"""
    loader = get_data_loader("")
    
    assert isinstance(loader, BaseDataLoader)


def test_get_data_loader_none():
    """测试 get_data_loader（None）"""
    # None 会导致 AttributeError，因为会调用 None.lower()
    with pytest.raises(AttributeError):
        get_data_loader(None)


def test_base_data_loader_multiple_instances():
    """测试 BaseDataLoader（多个实例）"""
    loader1 = BaseDataLoader("loader1")
    loader2 = BaseDataLoader("loader2")
    
    assert loader1.name == "loader1"
    assert loader2.name == "loader2"
    assert loader1 is not loader2


def test_base_data_loader_load_data_same_dates():
    """测试 BaseDataLoader（加载数据，相同日期）"""
    loader = BaseDataLoader("test_loader")
    date = datetime(2024, 1, 1)
    
    result = loader.load_data(["AAPL"], date, date)
    
    assert result["start_date"] == date
    assert result["end_date"] == date
    assert result["status"] == "success"


def test_base_data_loader_load_data_reversed_dates():
    """测试 BaseDataLoader（加载数据，日期倒序）"""
    loader = BaseDataLoader("test_loader")
    start_date = datetime(2024, 1, 31)
    end_date = datetime(2024, 1, 1)
    
    result = loader.load_data(["AAPL"], start_date, end_date)
    
    # 即使日期倒序，也应该正常处理
    assert result["start_date"] == start_date
    assert result["end_date"] == end_date
    assert result["status"] == "success"

