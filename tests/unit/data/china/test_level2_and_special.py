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

import pytest

# 提前注入 BaseDataAdapter 依赖，避免导入 src.data.china 时失败
base_adapter_module = sys.modules.get("src.data.base_adapter")
if not base_adapter_module:
    base_adapter_module = ModuleType("src.data.base_adapter")
    sys.modules["src.data.base_adapter"] = base_adapter_module


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

from src.data.china.level2 import ChinaLevel2Processor
from src.data.china.special import SpecialStockHandler


class TestChinaLevel2Processor:
    def test_sequence_builds_dataframe_with_bids_and_asks(self):
        processor = ChinaLevel2Processor(config={"depth": 5})
        order_book = {
            "symbol": "600000",
            "timestamp": "2025-01-01 09:30:00",
            "bids": [
                {"price": 10.0, "volume": 100},
                {"price": 9.9, "volume": 200},
            ],
            "asks": [
                {"price": 10.1, "volume": 150},
                {"price": 10.2, "volume": 120},
            ],
        }

        df = processor.sequence(order_book)

        assert not df.empty
        assert set(df["side"]) == {"bid", "ask"}
        assert len(df) == 4
        assert (df["symbol"] == "600000").all()
        assert (df["timestamp"] == "2025-01-01 09:30:00").all()

    def test_sequence_invalid_input_returns_empty(self):
        processor = ChinaLevel2Processor()

        df_missing_fields = processor.sequence({"symbol": "600000"})
        df_not_dict = processor.sequence([])

        assert df_missing_fields.empty
        assert df_not_dict.empty

    def test_process_tick_returns_dataframe(self):
        processor = ChinaLevel2Processor()
        tick_data = {
            "symbol": "600000",
            "timestamp": "2025-01-01 09:30:00",
            "price": 10.05,
            "volume": 300,
            "side": "buy",
            "order_id": "abc123",
        }

        df = processor.process_tick(tick_data)

        assert len(df) == 1
        row = df.iloc[0]
        assert row["price"] == 10.05
        assert row["volume"] == 300
        assert row["side"] == "buy"
        assert row["order_id"] == "abc123"

    def test_process_tick_invalid_data_returns_empty(self):
        processor = ChinaLevel2Processor()

        assert processor.process_tick({}).empty
        assert processor.process_tick("invalid").empty  # type: ignore[arg-type]
        assert processor.process_tick({"symbol": "600000"}).empty

    def test_calculate_market_depth_returns_expected_metrics(self):
        processor = ChinaLevel2Processor()
        order_book = {
            "bids": [
                {"price": 9.9, "volume": 100},
                {"price": 9.8, "volume": 50},
            ],
            "asks": [
                {"price": 10.1, "volume": 120},
                {"price": 10.2, "volume": 80},
            ],
        }

        metrics = processor.calculate_market_depth(order_book)

        assert metrics["total_bid_volume"] == 150
        assert metrics["total_ask_volume"] == 200
        assert metrics["bid_levels"] == 2
        assert metrics["ask_levels"] == 2
        assert metrics["bid_ask_spread"] == pytest.approx(0.4)
        assert metrics["min_bid_price"] == 9.8
        assert metrics["max_ask_price"] == 10.2

    def test_calculate_market_depth_invalid_input_returns_empty(self):
        processor = ChinaLevel2Processor()

        assert processor.calculate_market_depth(None) == {}  # type: ignore[arg-type]
        assert processor.calculate_market_depth("invalid") == {}  # type: ignore[arg-type]

    def test_calculate_market_depth_missing_bids_or_asks_returns_empty(self):
        processor = ChinaLevel2Processor()

        assert processor.calculate_market_depth({"bids": []}) == {}
        assert processor.calculate_market_depth({"asks": []}) == {}


class TestSpecialStockHandler:
    def test_is_special_stock_recognizes_prefixes(self):
        handler = SpecialStockHandler()

        assert handler.is_special_stock("ST600000")
        assert handler.is_special_stock("*ST123456")
        assert handler.is_special_stock("688001")
        assert not handler.is_special_stock("600000")

    def test_get_special_rules_returns_expected_values(self):
        handler = SpecialStockHandler()

        st_rules = handler.get_special_rules("ST600000")
        star_rules = handler.get_special_rules("688001")
        other_rules = handler.get_special_rules("600000")

        assert st_rules["price_limit"] == 0.05
        assert st_rules["disclosure"] == "enhanced"

        assert star_rules["price_limit"] == 0.2
        assert star_rules["after_hours"] is True

        assert other_rules == {}

    def test_filter_special_stocks_groups_by_category(self):
        handler = SpecialStockHandler()
        stocks = ["ST600000", "*ST000001", "688001", "600519"]

        grouped = handler.filter_special_stocks(stocks)

        assert grouped["ST"] == ["ST600000", "*ST000001"]
        assert grouped["STAR"] == ["688001"]
        assert grouped["other"] == ["600519"]

