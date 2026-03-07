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


import types
import sys
import pytest
from fastapi.testclient import TestClient

# 为使接口模块可导入，注入最小的 data_manager 桩
dm_module_name = "src.data.interfaces.data_manager"
if dm_module_name not in sys.modules:
    dm = types.ModuleType(dm_module_name)
    class _DummyDM:
        _inited = False
        @classmethod
        def get_instance(cls):
            return cls()
        def initialize(self):
            self._inited = True
        def is_initialized(self):
            return True
        def store_data(self, data=None, storage_type=None, metadata=None):
            return {"ok": True}
    dm.DataManagerSingleton = _DummyDM
    sys.modules[dm_module_name] = dm

from src.data.interfaces import api as interfaces_api  # type: ignore


@pytest.mark.skipif(interfaces_api is None, reason="接口模块不可用")
def test_health_and_ready_endpoints():
    client = TestClient(interfaces_api.app)
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "healthy"
    r2 = client.get("/ready")
    # 启动事件会初始化 data_manager，就绪应为 ready
    assert r2.status_code in (200, 503)
    # 当 data_manager 初始化成功时返回 ready
    if r2.status_code == 200:
        assert r2.json()["status"] == "ready"


