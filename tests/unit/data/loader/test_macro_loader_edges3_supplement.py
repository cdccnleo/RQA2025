"""
macro_loader.py 边界测试补充 - 第3批
覆盖未覆盖的方法和异常处理路径
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
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch, MagicMock
import asyncio

from src.data.loader.macro_loader import (
    MacroDataLoader,
    FREDLoader,
    WorldBankLoader,
    get_macro_data
)

# Mock MacroIndicator 类，因为源代码中的定义无法直接实例化
class MockMacroIndicator:
    def __init__(self, indicator_id, name, value, unit, frequency, date, country, source):
        self.indicator_id = indicator_id
        self.name = name
        self.value = value
        self.unit = unit
        self.frequency = frequency
        self.date = date
        self.country = country
        self.source = source


@pytest.fixture
def macro_loader():
    """创建 MacroDataLoader 实例"""
    config = {
        'fred_api_key': 'test_fred_key',
        'worldbank_api_key': 'test_wb_key'
    }
    return MacroDataLoader(config)


@pytest.fixture
def fred_loader():
    """创建 FREDLoader 实例"""
    return FREDLoader(api_key='test_key')


@pytest.fixture
def worldbank_loader():
    """创建 WorldBankLoader 实例"""
    return WorldBankLoader(api_key='test_key')


@pytest.mark.asyncio
async def test_macro_loader_get_gdp_data_us(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_gdp_data，US，覆盖 514-535 行）"""
    # Mock MacroIndicator 类
    monkeypatch.setattr('src.data.loader.macro_loader.MacroIndicator', MockMacroIndicator)
    
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器
    mock_context = AsyncMock()
    mock_context.get_series = AsyncMock(return_value={
        'observations': [
            {'date': '2020-01-01', 'value': '1000.0'},
            {'date': '2020-04-01', 'value': '1050.0'}
        ]
    })
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 fred_loader 为一个完全 mock 的对象
    macro_loader.fred_loader = mock_context
    
    result = await macro_loader.get_gdp_data("US", years=5)
    
    assert isinstance(result, list)
    assert len(result) == 2
    # MacroIndicator 是类，检查是否有 indicator_id 属性
    assert all(hasattr(ind, 'indicator_id') for ind in result)


@pytest.mark.asyncio
async def test_macro_loader_get_gdp_data_other_country(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_gdp_data，其他国家，覆盖 537-555 行）"""
    # Mock MacroIndicator 类
    monkeypatch.setattr('src.data.loader.macro_loader.MacroIndicator', MockMacroIndicator)
    
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器
    mock_context = AsyncMock()
    mock_context.get_indicator = AsyncMock(return_value=[
        None,  # 第一个元素通常是元数据
        [
            {'date': '2020', 'value': '2000.0'},
            {'date': '2021', 'value': '2100.0'}
        ]
    ])
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 worldbank_loader 为一个完全 mock 的对象
    macro_loader.worldbank_loader = mock_context
    
    result = await macro_loader.get_gdp_data("CN", years=5)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(hasattr(ind, 'indicator_id') for ind in result)


@pytest.mark.asyncio
async def test_macro_loader_get_gdp_data_no_data(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_gdp_data，无数据，覆盖 557 行）"""
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器，返回 None
    mock_context = AsyncMock()
    mock_context.get_series = AsyncMock(return_value=None)
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 fred_loader
    macro_loader.fred_loader = mock_context
    
    result = await macro_loader.get_gdp_data("US", years=5)
    
    assert result == []


@pytest.mark.asyncio
async def test_macro_loader_get_inflation_data_us(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_inflation_data，US，覆盖 561-582 行）"""
    # Mock MacroIndicator 类
    monkeypatch.setattr('src.data.loader.macro_loader.MacroIndicator', MockMacroIndicator)
    
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器
    mock_context = AsyncMock()
    mock_context.get_series = AsyncMock(return_value={
        'observations': [
            {'date': '2020-01-01', 'value': '250.0'},
            {'date': '2020-02-01', 'value': '251.0'}
        ]
    })
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 fred_loader
    macro_loader.fred_loader = mock_context
    
    result = await macro_loader.get_inflation_data("US", years=5)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(hasattr(ind, 'indicator_id') for ind in result)


@pytest.mark.asyncio
async def test_macro_loader_get_inflation_data_other_country(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_inflation_data，其他国家，覆盖 584-600 行）"""
    # Mock MacroIndicator 类
    monkeypatch.setattr('src.data.loader.macro_loader.MacroIndicator', MockMacroIndicator)
    
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器
    mock_context = AsyncMock()
    mock_context.get_indicator = AsyncMock(return_value=[
        None,
        [
            {'date': '2020', 'value': '2.5'},
            {'date': '2021', 'value': '2.6'}
        ]
    ])
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 worldbank_loader
    macro_loader.worldbank_loader = mock_context
    
    result = await macro_loader.get_inflation_data("CN", years=5)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(hasattr(ind, 'indicator_id') for ind in result)


@pytest.mark.asyncio
async def test_macro_loader_get_interest_rate_data_us(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_interest_rate_data，US，覆盖 606-627 行）"""
    # Mock MacroIndicator 类
    monkeypatch.setattr('src.data.loader.macro_loader.MacroIndicator', MockMacroIndicator)
    
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器
    mock_context = AsyncMock()
    mock_context.get_series = AsyncMock(return_value={
        'observations': [
            {'date': '2020-01-01', 'value': '1.5'},
            {'date': '2020-02-01', 'value': '1.6'}
        ]
    })
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 fred_loader
    macro_loader.fred_loader = mock_context
    
    result = await macro_loader.get_interest_rate_data("US", years=5)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(hasattr(ind, 'indicator_id') for ind in result)


@pytest.mark.asyncio
async def test_macro_loader_get_employment_data_us(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_employment_data，US，覆盖 633-654 行）"""
    # Mock MacroIndicator 类
    monkeypatch.setattr('src.data.loader.macro_loader.MacroIndicator', MockMacroIndicator)
    
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器
    mock_context = AsyncMock()
    mock_context.get_series = AsyncMock(return_value={
        'observations': [
            {'date': '2020-01-01', 'value': '150000.0'},
            {'date': '2020-02-01', 'value': '151000.0'}
        ]
    })
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 fred_loader
    macro_loader.fred_loader = mock_context
    
    result = await macro_loader.get_employment_data("US", years=5)
    
    assert isinstance(result, list)
    assert len(result) == 2
    assert all(hasattr(ind, 'indicator_id') for ind in result)


@pytest.mark.asyncio
async def test_macro_loader_validate_data_exception(macro_loader, monkeypatch):
    """测试 MacroDataLoader（validate_data，异常处理，覆盖 692-694 行）"""
    # 创建一个会触发异常的 MockIndicator
    # 需要在访问 value 属性时抛出异常，因为 indicator_id 和 name 的检查在前
    class MockIndicator:
        def __init__(self):
            self.indicator_id = "TEST"
            self.name = "Test"
            self._value = 100.0
            self.unit = "Unit"
            self.frequency = "Monthly"
            self.date = datetime.now()
            self.country = "US"
            self.source = "Test"
        
        # 让访问 value 属性时抛出异常（在数值检查时触发）
        @property
        def value(self):
            raise Exception("验证异常")
    
    # 使用一个会触发异常的 indicator
    indicator = MockIndicator()
    indicators = [indicator]
    
    result = await macro_loader.validate_data(indicators)
    
    assert 'errors' in result
    assert result['invalid_records'] > 0
    assert len(result['errors']) > 0
    # 检查错误消息中是否包含异常信息
    assert any("验证异常" in err for err in result['errors'])


@pytest.mark.asyncio
async def test_get_macro_data_function(macro_loader, monkeypatch):
    """测试 get_macro_data 便捷函数（覆盖 767-773 行）"""
    # Mock MacroDataLoader
    mock_loader = AsyncMock()
    mock_loader.load_data = AsyncMock(return_value={
        'data': pd.DataFrame({'value': [100, 200]}),
        'metadata': {'indicator_type': 'gdp', 'country': 'US'}
    })
    
    with patch('src.data.loader.macro_loader.MacroDataLoader', return_value=mock_loader):
        result = await get_macro_data("gdp", "US", 10)
    
    assert 'data' in result
    assert 'metadata' in result


@pytest.mark.asyncio
async def test_macro_loader_get_gdp_data_dot_value(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_gdp_data，value 为 '.'，覆盖 524 行）"""
    # Mock MacroIndicator 类
    monkeypatch.setattr('src.data.loader.macro_loader.MacroIndicator', MockMacroIndicator)
    
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器
    mock_context = AsyncMock()
    mock_context.get_series = AsyncMock(return_value={
        'observations': [
            {'date': '2020-01-01', 'value': '.'},  # 应该被跳过
            {'date': '2020-04-01', 'value': '1050.0'}  # 应该被包含
        ]
    })
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 fred_loader
    macro_loader.fred_loader = mock_context
    
    result = await macro_loader.get_gdp_data("US", years=5)
    
    # 只有非 '.' 的值应该被包含
    assert len(result) == 1


@pytest.mark.asyncio
async def test_macro_loader_get_gdp_data_worldbank_short_response(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_gdp_data，WorldBank 响应太短，覆盖 541 行）"""
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器
    mock_context = AsyncMock()
    mock_context.get_indicator = AsyncMock(return_value=[None])  # 长度只有1
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 worldbank_loader
    macro_loader.worldbank_loader = mock_context
    
    result = await macro_loader.get_gdp_data("CN", years=5)
    
    # 应该返回空列表
    assert result == []


@pytest.mark.asyncio
async def test_macro_loader_get_inflation_data_worldbank_no_value(macro_loader, monkeypatch):
    """测试 MacroDataLoader（get_inflation_data，WorldBank 数据无 value，覆盖 590 行）"""
    # Mock MacroIndicator 类
    monkeypatch.setattr('src.data.loader.macro_loader.MacroIndicator', MockMacroIndicator)
    
    # 先初始化 loader
    await macro_loader.initialize()
    
    # 创建一个完全 mock 的上下文管理器
    mock_context = AsyncMock()
    mock_context.get_indicator = AsyncMock(return_value=[
        None,
        [
            {'date': '2020'},  # 没有 value
            {'date': '2021', 'value': '2.6'}  # 有 value
        ]
    ])
    mock_context.__aenter__ = AsyncMock(return_value=mock_context)
    mock_context.__aexit__ = AsyncMock(return_value=None)
    
    # 直接替换 worldbank_loader
    macro_loader.worldbank_loader = mock_context
    
    result = await macro_loader.get_inflation_data("CN", years=5)
    
    # 只有有 value 的数据应该被包含
    assert len(result) == 1

