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
from pydantic import ValidationError

from src.data.interfaces.api import DataStorageRequest, DataRequest


def test_data_storage_request_rejects_none_data():
    with pytest.raises(ValidationError):
        DataStorageRequest(data=None)  # storage_type 有默认值


def test_data_storage_request_accepts_valid_payload():
    req = DataStorageRequest(data={"rows": [1, 2, 3]}, metadata={"source": "unit"})
    assert req.storage_type == "database"
    assert req.metadata == {"source": "unit"}


def test_data_request_defaults_and_fields():
    req = DataRequest(symbol="AAPL", start_date="2024-01-01", end_date="2024-01-31")
    assert req.data_type == "ohlcv"
    assert req.source == "default"
    assert isinstance(req.symbol, str)


