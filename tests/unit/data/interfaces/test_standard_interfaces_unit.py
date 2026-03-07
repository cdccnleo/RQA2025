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


from src.data.interfaces.standard_interfaces import DataRequest, DataResponse, DataSourceType


def test_data_request_auto_request_id_and_defaults():
    req = DataRequest(source="api", query={"q": "x"})
    assert req.source == "api"
    assert isinstance(req.request_id, str) and len(req.request_id) > 0
    # 可选字段为空时保持 None
    assert req.filters is None
    assert req.limit is None
    assert req.offset is None


def test_data_response_defaults_and_assignment():
    resp = DataResponse(data=[1, 2, 3])
    assert resp.status == "success"
    assert resp.message is None
    assert resp.metadata is None
    # 指定 metadata 与 request_id
    resp2 = DataResponse(data={"k": "v"}, status="success", message=None, request_id="rid", metadata={"m": 1})
    assert resp2.request_id == "rid"
    assert resp2.metadata == {"m": 1}


def test_data_source_type_members():
    # 枚举值存在性
    assert DataSourceType.DATABASE.value == "database"
    assert DataSourceType.API.value == "api"
    assert DataSourceType.FILE.value == "file"
    assert DataSourceType.STREAM.value == "stream"
    assert DataSourceType.CACHE.value == "cache"
    assert DataSourceType.STOCK.value == "stock"
    assert DataSourceType.CRYPTO.value == "crypto"
    assert DataSourceType.NEWS.value == "news"
    assert DataSourceType.MACRO.value == "macro"


