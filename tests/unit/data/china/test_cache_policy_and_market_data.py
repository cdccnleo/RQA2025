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

import numpy as np
import pandas as pd
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

from src.data.china.cache_policy import ChinaCachePolicy
from src.data.china import market_data as market_module
from src.data.china import china_data_adapter as adapter_module
from src.data.china.market_data import MarketData


class _DummySecrets:
    """提供与代码中调用一致的 np.secrets API。"""

    @staticmethod
    def randn(size):
        return np.random.randn(size)

    @staticmethod
    def randint(low, high, size):
        return np.random.randint(low, high, size)

    @staticmethod
    def uniform(low, high, size):
        return np.random.uniform(low, high, size)


def test_china_cache_policy_known_type():
    policy = ChinaCachePolicy.get_policy("level2")
    assert policy.ttl == 15
    assert policy.max_ttl == 120
    assert policy.refresh_interval == 5


def test_china_cache_policy_default_type():
    policy = ChinaCachePolicy.get_policy("unknown_type")
    assert policy.ttl == 3600
    assert policy.max_ttl == 86400
    assert policy.refresh_interval == 600


@pytest.fixture
def market_with_secrets(monkeypatch):
    monkeypatch.setattr(np, "secrets", _DummySecrets(), raising=False)
    return MarketData()


def test_market_data_get_data_success(market_with_secrets):
    df = market_with_secrets.get_data("test_key")
    assert isinstance(df, pd.DataFrame)
    assert {"timestamp", "price", "volume"}.issubset(df.columns)
    assert len(df) == 100


def test_market_data_get_fundamental_data_success(market_with_secrets):
    df = market_with_secrets.get_fundamental_data("test_key")
    assert isinstance(df, pd.DataFrame)
    assert {"timestamp", "pe_ratio", "pb_ratio", "roe"}.issubset(df.columns)
    assert len(df) == 50


def test_market_data_get_technical_data_success(market_with_secrets):
    df = market_with_secrets.get_technical_data("test_key")
    assert isinstance(df, pd.DataFrame)
    assert {"timestamp", "ma_5", "ma_10", "rsi"}.issubset(df.columns)


def test_market_data_methods_log_error_when_missing_secrets(monkeypatch):
    # 确保 np.secrets 不存在，从而触发异常路径
    monkeypatch.delattr(np, "secrets", raising=False)

    captured = []

    class DummyLogger:
        def error(self, msg):
            captured.append(msg)

    # 替换模块级 logger，新的 MarketData 实例会引用它
    monkeypatch.setattr(market_module, "logger", DummyLogger())
    data = market_module.MarketData()

    assert data.get_data("key") is None
    assert data.get_fundamental_data("key") is None
    assert data.get_technical_data("key") is None
    # 至少记录过一次错误信息
    assert captured, "Expected errors to be logged when np.secrets is missing"


def test_market_data_handles_dataframe_error(monkeypatch):
    monkeypatch.setattr(np, "secrets", _DummySecrets(), raising=False)

    errors = []

    class DummyLogger:
        def error(self, msg):
            errors.append(msg)

    monkeypatch.setattr(market_module, "logger", DummyLogger())

    def raise_dataframe(*args, **kwargs):
        raise ValueError("df boom")

    monkeypatch.setattr(market_module.pd, "DataFrame", raise_dataframe)

    data = market_module.MarketData()
    assert data.get_data("boom") is None
    assert errors and "df boom" in errors[-1]


def test_china_data_adapter_smoke():
    adapter = adapter_module.ChinaDataAdapter()
    assert adapter.__doc__ == "空壳中国数据适配器，待实现"

