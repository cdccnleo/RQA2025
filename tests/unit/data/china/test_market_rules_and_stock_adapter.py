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


import sys
from types import ModuleType
from datetime import date

import pytest

# 提前注入 BaseDataAdapter 依赖，避免导入 src.data.china.stock 时失败
if "src.data.base_adapter" not in sys.modules:
    base_adapter_module = ModuleType("src.data.base_adapter")

    class _BaseDataAdapter:
        def __init__(self, config=None):
            self.config = config or {}

        def _validate_config(self):
            return True

        def transform(self, raw_data):
            return raw_data

        def connect(self):
            return True

        def disconnect(self):
            return None

        def is_connected(self):
            return True

    base_adapter_module.BaseDataAdapter = _BaseDataAdapter
    sys.modules["src.data.base_adapter"] = base_adapter_module

from src.data.china.market import ChinaMarketRules
from src.data.china.stock import ChinaStockDataAdapter, ChinaDataAdapter


class TestChinaMarketRules:
    @pytest.mark.parametrize(
        "symbol, expected",
        [
            ("600000", 0.1),
            (" 600519 ", 0.1),
            ("688001", 0.2),
            ("300123", 0.2),
        ],
    )
    def test_get_price_limit_handles_board_types(self, symbol, expected):
        assert ChinaMarketRules.get_price_limit(symbol) == pytest.approx(expected)

    def test_is_t1_restricted_valid_and_invalid_inputs(self):
        assert ChinaMarketRules.is_t1_restricted("600000") is True

        with pytest.raises(AttributeError):
            ChinaMarketRules.is_t1_restricted(None)

        with pytest.raises(AttributeError):
            ChinaMarketRules.is_t1_restricted(600000)  # type: ignore[arg-type]

    def test_get_star_market_rules_returns_expected_payload(self):
        rules = ChinaMarketRules.get_star_market_rules("688001")

        assert rules["after_hours_trading"] is True
        assert rules["price_limits"] == pytest.approx(0.2)
        assert rules["listing_requirements"]["market_cap"] == 1_000_000_000

        assert ChinaMarketRules.get_star_market_rules("600000") == {}


class TestChinaStockDataAdapter:
    def test_adapter_type_and_default_config(self):
        adapter = ChinaStockDataAdapter()

        assert adapter.adapter_type == "china_stock"
        assert adapter.config["market"] == "A股"

    def test_adapter_custom_config_preserved(self):
        adapter = ChinaStockDataAdapter(config={"market": "港股", "env": "test"})

        assert adapter.config["market"] == "港股"
        assert adapter.config["env"] == "test"

    def test_connection_and_validation_methods(self):
        adapter = ChinaStockDataAdapter()

        assert adapter.connect() is True
        assert adapter.is_connected() is True
        assert adapter.disconnect() is None
        assert adapter.validate({"any": "data"}) is True
        assert adapter.transform({"raw": "data"}) is None

    def test_data_access_methods_return_stubbed_payloads(self):
        adapter = ChinaStockDataAdapter()

        assert adapter.load() is None
        assert adapter._validate_config() is True  # 验证基础配置逻辑路径

        basic = adapter.get_stock_basic("600000")
        assert basic["code"] == "600000"

        quotes = adapter.get_daily_quotes("600000", date(2024, 1, 1), date(2024, 1, 31))
        assert isinstance(quotes, list)
        assert quotes[0]["date"] == "待实现"

        factors = adapter.get_adj_factors("600000")
        assert factors == {"待实现": 1.0}

    def test_china_data_adapter_alias(self):
        assert ChinaDataAdapter is ChinaStockDataAdapter

