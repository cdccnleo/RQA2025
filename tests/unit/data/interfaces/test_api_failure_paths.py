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
from fastapi.testclient import TestClient

from src.data.interfaces.api import app


def test_store_failure_path_returns_500(monkeypatch):
    client = TestClient(app)

    # 模拟 DataManagerSingleton.store_data 抛出异常
    from src.data.interfaces import api as api_module
    # 直接替换模块级 data_manager 实例的 store_data 为抛错
    class _DM:
        def store_data(self, *args, **kwargs):  # pragma: no cover - callable only
            raise RuntimeError("store failed")
    api_module.data_manager = _DM()

    resp = client.post(
        "/api / v1 / data / store",
        json={"data": [{"a": 1}], "metadata": {"source": "x"}},
    )
    # 失败路径应返回 500
    assert resp.status_code >= 500
    body = resp.json()
    assert "error" in body or "detail" in body


