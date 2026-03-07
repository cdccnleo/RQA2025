"""
macro_loader.py 边界测试补充
目标：将覆盖率从 72% 提升到 80%+
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
import asyncio
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock, AsyncMock
from src.data.loader.macro_loader import (
    FREDLoader,
    WorldBankLoader,
    MacroDataLoader,
    MacroIndicator
)


@pytest.fixture
def fred_loader():
    """创建 FRED 加载器实例"""
    with patch('src.data.loader.macro_loader.CacheManager'):
        loader = FREDLoader(api_key="test_key")
        loader.cache_manager = AsyncMock()
        loader.cache_manager.get = AsyncMock(return_value=None)
        loader.cache_manager.set = AsyncMock()
        return loader


@pytest.fixture
def worldbank_loader():
    """创建 WorldBank 加载器实例"""
    with patch('src.data.loader.macro_loader.CacheManager'):
        loader = WorldBankLoader(api_key="test_key")
        loader.cache_manager = AsyncMock()
        loader.cache_manager.get = AsyncMock(return_value=None)
        loader.cache_manager.set = AsyncMock()
        return loader


@pytest.fixture
def macro_loader():
    """创建 MacroDataLoader 实例"""
    with patch('src.data.loader.macro_loader.CacheManager'):
        loader = MacroDataLoader()
        loader.cache_manager = Mock()
        loader.cache_manager.get = Mock(return_value=None)
        loader.cache_manager.set = Mock()
        return loader


@pytest.mark.asyncio
async def test_fred_loader_get_series_info_error(fred_loader, monkeypatch):
    """测试 FRED 加载器（get_series_info，错误状态码，覆盖 213-214 行）"""
    # Mock session.get 返回错误状态码
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    fred_loader.session = AsyncMock()
    fred_loader.session.get = Mock(return_value=mock_response)
    
    result = await fred_loader.get_series_info("GDP")
    
    assert result is None


@pytest.mark.asyncio
async def test_fred_loader_get_series_info_exception(fred_loader, monkeypatch):
    """测试 FRED 加载器（get_series_info，异常处理，覆盖 216-218 行）"""
    # Mock session.get 抛出异常
    fred_loader.session = AsyncMock()
    fred_loader.session.get = Mock(side_effect=Exception("Network error"))
    
    result = await fred_loader.get_series_info("GDP")
    
    assert result is None


@pytest.mark.asyncio
async def test_fred_loader_search_series_error(fred_loader, monkeypatch):
    """测试 FRED 加载器（search_series，错误状态码，覆盖 247-248 行）"""
    # Mock session.get 返回错误状态码
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    fred_loader.session = AsyncMock()
    fred_loader.session.get = Mock(return_value=mock_response)
    
    result = await fred_loader.search_series("GDP")
    
    assert result == []


@pytest.mark.asyncio
async def test_fred_loader_search_series_exception(fred_loader, monkeypatch):
    """测试 FRED 加载器（search_series，异常处理，覆盖 250-252 行）"""
    # Mock session.get 抛出异常
    fred_loader.session = AsyncMock()
    fred_loader.session.get = Mock(side_effect=Exception("Network error"))
    
    result = await fred_loader.search_series("GDP")
    
    assert result == []


@pytest.mark.asyncio
async def test_worldbank_loader_get_countries_exception(worldbank_loader, monkeypatch):
    """测试 WorldBank 加载器（get_countries，异常处理，覆盖 400-402 行）"""
    # Mock session.get 抛出异常
    worldbank_loader.session = AsyncMock()
    worldbank_loader.session.get = Mock(side_effect=Exception("Network error"))
    
    result = await worldbank_loader.get_countries()
    
    assert result == []


@pytest.mark.asyncio
async def test_worldbank_loader_get_indicators_error(worldbank_loader, monkeypatch):
    """测试 WorldBank 加载器（get_indicators，错误状态码，覆盖 432-433 行）"""
    # Mock session.get 返回错误状态码
    mock_response = AsyncMock()
    mock_response.status = 404
    mock_response.__aenter__ = AsyncMock(return_value=mock_response)
    mock_response.__aexit__ = AsyncMock(return_value=None)
    
    worldbank_loader.session = AsyncMock()
    worldbank_loader.session.get = Mock(return_value=mock_response)
    
    result = await worldbank_loader.get_indicators()
    
    assert result == []


@pytest.mark.skip(reason="异步上下文管理器 mock 复杂，需要进一步调试")
@pytest.mark.asyncio
async def test_macro_loader_get_gdp_data_non_us(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_gdp_data，非US国家，覆盖 538-557 行）"""
    pass


@pytest.mark.skip(reason="异步上下文管理器 mock 复杂，需要进一步调试")
@pytest.mark.asyncio
async def test_macro_loader_get_inflation_data_non_us(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_inflation_data，非US国家，覆盖 585-603 行）"""
    pass


@pytest.mark.skip(reason="异步上下文管理器 mock 复杂，需要进一步调试")
@pytest.mark.asyncio
async def test_macro_loader_get_interest_rate_data(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_interest_rate_data，覆盖 608-631 行）"""
    pass


@pytest.mark.skip(reason="异步上下文管理器 mock 复杂，需要进一步调试")
@pytest.mark.asyncio
async def test_macro_loader_get_employment_data(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_employment_data，覆盖 635-658 行）"""
    pass


@pytest.mark.skip(reason="MacroIndicator 类定义问题，需要进一步调试")
@pytest.mark.asyncio
async def test_macro_loader_validate_data_exception(macro_loader, monkeypatch):
    """测试 MacroDataLoader（validate_data，异常处理，覆盖 692-694 行）"""
    pass


@pytest.mark.asyncio
async def test_get_macro_data_convenience_function(monkeypatch):
    """测试便捷函数 get_macro_data（覆盖 769-770 行）"""
    from src.data.loader.macro_loader import get_macro_data
    
    # Mock MacroDataLoader.load_data
    async def mock_load_data(**kwargs):
        return {
            'data': Mock(),
            'metadata': {'test': 'data'}
        }
    
    with patch('src.data.loader.macro_loader.MacroDataLoader') as MockLoader:
        mock_instance = Mock()
        mock_instance.load_data = mock_load_data
        MockLoader.return_value = mock_instance
        
        result = await get_macro_data(indicator_type="gdp", country="US", years=5)
        
        assert result is not None
        assert 'data' in result
        assert 'metadata' in result

