"""
测试中国市场适配器实现

测试目标：为 china/adapters/ 中的适配器实现编写完整的单元测试
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
from unittest.mock import Mock, patch
import pandas as pd

from src.data.china.adapters import (
    AStockAdapter,
    STARMarketAdapter,
    ChinaStockAdapter  # 别名
)


class TestAStockAdapter:
    """测试 A股适配器"""

    @pytest.fixture
    def adapter(self):
        """创建 A股适配器实例"""
        return AStockAdapter()

    @pytest.fixture
    def adapter_with_config(self):
        """创建带配置的 A股适配器实例"""
        return AStockAdapter(config={'timeout': 30, 'retries': 3})

    def test_adapter_initialization(self, adapter):
        """测试适配器初始化"""
        assert adapter.config == {}
        assert adapter._is_connected is False
        assert hasattr(adapter, 'logger')

    def test_adapter_initialization_with_config(self, adapter_with_config):
        """测试带配置的适配器初始化"""
        assert adapter_with_config.config['timeout'] == 30
        assert adapter_with_config.config['retries'] == 3

    def test_connect(self, adapter):
        """测试连接"""
        result = adapter.connect()
        assert result is True
        assert adapter._is_connected is True
        assert adapter.is_connected() is True

    def test_disconnect(self, adapter):
        """测试断开连接"""
        adapter._is_connected = True
        result = adapter.disconnect()
        assert result is True
        assert adapter._is_connected is False
        assert adapter.is_connected() is False

    def test_get_data_basic(self, adapter):
        """测试获取基础数据"""
        data = adapter.get_data('600519')
        assert data['symbol'] == '600519'
        assert data['market'] == 'A股'
        assert 'data_type' in data

    def test_get_data_with_kwargs(self, adapter):
        """测试获取数据带参数"""
        data = adapter.get_data('600519', data_type='financial')
        assert data['symbol'] == '600519'
        assert data['data_type'] == 'financial'

    def test_get_stock_basic(self, adapter):
        """测试获取股票基础信息"""
        result = adapter.get_stock_basic('600519')
        assert isinstance(result, pd.DataFrame)

    def test_get_stock_basic_no_symbol(self, adapter):
        """测试获取股票基础信息（无代码）"""
        result = adapter.get_stock_basic()
        assert isinstance(result, pd.DataFrame)

    def test_get_daily_quotes(self, adapter):
        """测试获取日线行情"""
        result = adapter.get_daily_quotes('600519', '2024-01-01', '2024-01-31')
        assert isinstance(result, pd.DataFrame)

    def test_get_financial_data_annual(self, adapter):
        """测试获取年度财务数据"""
        result = adapter.get_financial_data('600519', 'annual')
        assert isinstance(result, pd.DataFrame)

    def test_get_financial_data_quarterly(self, adapter):
        """测试获取季度财务数据"""
        result = adapter.get_financial_data('600519', 'quarterly')
        assert isinstance(result, pd.DataFrame)

    def test_china_stock_adapter_alias(self, adapter):
        """测试 ChinaStockAdapter 别名"""
        adapter2 = ChinaStockAdapter()
        assert isinstance(adapter2, AStockAdapter)


class TestSTARMarketAdapter:
    """测试科创板适配器"""

    @pytest.fixture
    def adapter(self):
        """创建科创板适配器实例"""
        return STARMarketAdapter()

    @pytest.fixture
    def adapter_with_config(self):
        """创建带配置的科创板适配器实例"""
        return STARMarketAdapter(config={'timeout': 30})

    def test_adapter_initialization(self, adapter):
        """测试适配器初始化"""
        assert adapter.config == {}
        assert adapter._is_connected is False
        assert adapter.market_type == 'STAR'

    def test_adapter_inherits_from_astock(self, adapter):
        """测试适配器继承自 AStockAdapter"""
        assert isinstance(adapter, AStockAdapter)

    def test_connect(self, adapter):
        """测试连接"""
        result = adapter.connect()
        assert result is True
        assert adapter._is_connected is True

    def test_disconnect(self, adapter):
        """测试断开连接"""
        adapter._is_connected = True
        result = adapter.disconnect()
        assert result is True
        assert adapter._is_connected is False

    def test_get_data_star_market(self, adapter):
        """测试获取科创板数据"""
        data = adapter.get_data('688001')
        assert data['symbol'] == '688001'
        assert data['market'] == '科创板'
        assert data['market_type'] == 'STAR'

    def test_get_star_market_data(self, adapter):
        """测试获取科创板特有数据"""
        data = adapter.get_star_market_data('688001')
        assert data['symbol'] == '688001'
        assert data['market_type'] == 'STAR'
        assert data['has_after_hours_trading'] is True

    def test_get_after_hours_trading(self, adapter):
        """测试获取盘后固定价格交易数据"""
        result = adapter.get_after_hours_trading('688001')
        assert isinstance(result, pd.DataFrame)

    def test_get_red_chip_info(self, adapter):
        """测试获取红筹企业信息"""
        data = adapter.get_red_chip_info('688001')
        assert data['symbol'] == '688001'
        assert data['is_red_chip'] is True
        assert 'red_chip_type' in data
        assert data['red_chip_type'] == 'VCDR'

    def test_star_adapter_inheritance(self, adapter):
        """测试科创板适配器继承的方法"""
        # 测试继承自 AStockAdapter 的方法
        basic = adapter.get_stock_basic('688001')
        assert isinstance(basic, pd.DataFrame)
        
        quotes = adapter.get_daily_quotes('688001', '2024-01-01', '2024-01-31')
        assert isinstance(quotes, pd.DataFrame)


class TestAdaptersIntegration:
    """测试适配器集成场景"""

    def test_multiple_adapters_independent(self):
        """测试多个适配器实例相互独立"""
        a_stock = AStockAdapter()
        star = STARMarketAdapter()
        
        a_stock.connect()
        assert a_stock.is_connected() is True
        assert star.is_connected() is False

    def test_adapter_config_preserved(self):
        """测试适配器配置保持不变"""
        config = {'custom': 'value'}
        adapter = AStockAdapter(config)
        assert adapter.config['custom'] == 'value'

    def test_adapter_logger_initialized(self):
        """测试适配器日志初始化"""
        adapter = AStockAdapter()
        assert adapter.logger is not None
        assert adapter.logger.name in ['AStockAdapter', 'STARMarketAdapter']

