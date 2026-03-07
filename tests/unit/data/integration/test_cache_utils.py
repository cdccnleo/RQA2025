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


import pandas as pd

from src.data.integration.enhanced_data_integration_modules import cache_utils


class _DummyCache:
    def __init__(self):
        self.storage = {}
        self.set_calls = []

    def get(self, key):
        return self.storage.get(key)

    def set(self, key, value, ttl=None):
        self.storage[key] = value
        self.set_calls.append({"key": key, "ttl": ttl, "value": value})


def test_check_cache_helpers_return_only_hits():
    cache = _DummyCache()
    df_stock = pd.DataFrame({"value": [1]})
    df_index = pd.DataFrame({"value": [2]})
    df_financial = pd.DataFrame({"value": [3]})

    cache.storage["stock_RQA_2024-01-01_2024-01-31_daily"] = df_stock
    cache.storage["index_SH000300_2024-01-01_2024-01-31_daily"] = df_index
    cache.storage["financial_RQA_2024-01-01_2024-01-31_income"] = df_financial

    stocks = cache_utils.check_cache_for_symbols(
        cache, ["RQA", "XYZ"], "2024-01-01", "2024-01-31", "daily"
    )
    assert stocks == {"RQA": df_stock}

    indices = cache_utils.check_cache_for_indices(
        cache, ["SH000300", "NDX"], "2024-01-01", "2024-01-31", "daily"
    )
    assert indices == {"SH000300": df_index}

    financials = cache_utils.check_cache_for_financial(
        ["RQA", "ABC"], "2024-01-01", "2024-01-31", "income", cache
    )
    assert financials == {"RQA": df_financial}


def test_cache_helpers_use_consistent_keys_and_ttl():
    cache = _DummyCache()
    df = pd.DataFrame({"value": [1]})

    cache_utils.cache_data(cache, "RQA", df, "2024-01-01", "2024-01-31", "daily")
    cache_utils.cache_index_data(cache, "SH000300", df, "2024-01-01", "2024-01-31", "daily")
    cache_utils.cache_financial_data(cache, "RQA", df, "2024-01-01", "2024-01-31", "income")

    expected_keys = {
        "stock_RQA_2024-01-01_2024-01-31_daily",
        "index_SH000300_2024-01-01_2024-01-31_daily",
        "financial_RQA_2024-01-01_2024-01-31_income",
    }
    assert {call["key"] for call in cache.set_calls} == expected_keys
    assert all(call["ttl"] == 3600 for call in cache.set_calls)

    for call in cache.set_calls:
        assert cache.storage[call["key"]] is call["value"]

