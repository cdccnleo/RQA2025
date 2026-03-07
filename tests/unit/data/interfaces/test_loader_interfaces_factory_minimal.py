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


from datetime import datetime
from src.data.interfaces.loader import get_data_loader, BaseDataLoader, StockDataLoader


def test_loader_factory_and_base_behavior():
    stock = get_data_loader("stock")
    assert isinstance(stock, StockDataLoader)
    unknown = get_data_loader("unknown")
    assert isinstance(unknown, BaseDataLoader)
    res = unknown.load_data(["000001"], datetime(2020, 1, 1), datetime(2020, 1, 2))
    assert res["status"] == "success" and res["symbols"] == ["000001"]


