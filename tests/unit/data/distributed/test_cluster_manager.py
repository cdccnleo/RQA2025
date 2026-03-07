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

if "src.models" not in sys.modules:
    models_module = ModuleType("src.models")

    class _DummyModel:
        def __init__(self, *args, **kwargs):
            self.data = kwargs.get("data")
            self.metadata = kwargs.get("metadata", {})

        def get_metadata(self, user_only: bool = False):
            return dict(self.metadata)

    models_module.DataModel = _DummyModel
    models_module.SimpleDataModel = _DummyModel
    sys.modules["src.models"] = models_module

if "src.interfaces" not in sys.modules:
    interfaces_module = ModuleType("src.interfaces")
    interfaces_module.IDistributedDataLoader = object
    sys.modules["src.interfaces"] = interfaces_module

from src.data.distributed.cluster_manager import ClusterManager, ClusterStatus


def test_register_and_unregister_updates_resources():
    manager = ClusterManager(config={"cluster_name": "DataGrid"})

    assert manager.get_cluster_info()["name"] == "DataGrid"

    manager.add_node("node-1", {"cpu_usage": 0.7, "memory_usage": 1.2})
    manager.register_node("node-2", {"cpu_usage": 1.0})

    assert len(manager.get_node_list()) == 2

    stats = manager.get_cluster_stats()
    assert stats["total_nodes"] == 2
    assert pytest.approx(stats["total_cpu"], rel=1e-3) == 1.7
    assert pytest.approx(stats["total_memory"], rel=1e-3) == 1.2
    assert manager.status == ClusterStatus.ACTIVE.value

    assert manager.remove_node("node-2") is True
    manager.unregister_node("node-1")

    stats = manager.get_cluster_stats()
    assert stats["total_nodes"] == 0
    assert stats["total_cpu"] == pytest.approx(0.0)


def test_update_status_and_get_status_summary():
    manager = ClusterManager()

    manager.cluster_info.active_nodes = 3
    manager.update_cluster_status(ClusterStatus.MAINTENANCE)

    status = manager.get_status()
    assert status["status"] == ClusterStatus.MAINTENANCE.value
    assert status["active_nodes"] == 3
    assert manager.status == "maintenance"

    assert manager.remove_node("missing") is False


def test_cluster_stats_handles_zero_nodes():
    manager = ClusterManager()

    stats = manager.get_cluster_stats()
    assert stats["total_nodes"] == 0
    assert stats["average_cpu"] == 0.0
    assert stats["average_memory"] == 0.0

