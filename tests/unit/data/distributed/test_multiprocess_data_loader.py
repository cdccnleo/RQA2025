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

# MultiprocessDataLoader 依赖 src.interfaces.IDistributedDataLoader，但该接口在代码库中缺失。
# 为了隔离测试，我们在导入模块前注入一个最小的桩对象。
if "src.interfaces" not in sys.modules:
    interfaces_module = ModuleType("src.interfaces")
    interfaces_module.IDistributedDataLoader = object
    sys.modules["src.interfaces"] = interfaces_module

if "src.models" not in sys.modules:
    models_module = ModuleType("src.models")

    class _DummyModel:
        def __init__(self, *args, **kwargs):
            self.data = kwargs.get("data")
            self.metadata = kwargs.get("metadata", {})

        def get_metadata(self, user_only=False):
            return dict(self.metadata)

    models_module.DataModel = _DummyModel
    models_module.SimpleDataModel = _DummyModel
    sys.modules["src.models"] = models_module

from src.data.distributed.multiprocess_loader import MultiprocessDataLoader


def sample_worker(task):
    return task["value"] * 2


def test_distribute_load_uses_configured_pool(monkeypatch):
    captured = {}

    class DummyPool:
        def __init__(self, processes=None):
            captured["processes"] = processes

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            captured["closed"] = True

        def map(self, fn, tasks):
            captured["mapped_tasks"] = tasks
            return [fn(task) for task in tasks]

    monkeypatch.setattr("src.data.distributed.multiprocess_loader.Pool", DummyPool)
    loader = MultiprocessDataLoader(worker_fn=sample_worker, num_workers=2)
    tasks = [{"value": 1}, {"value": 3}]

    result = loader.distribute_load(tasks)

    assert result == [2, 6]
    assert captured["processes"] == 2
    assert captured["mapped_tasks"] == tasks
    assert captured["closed"] is True


def test_aggregate_results_supports_custom_function():
    loader = MultiprocessDataLoader(worker_fn=sample_worker, num_workers=1)
    results = [1, 2, 3]

    assert loader.aggregate_results(results) == results
    assert loader.aggregate_results(results, aggregate_fn=sum) == 6


def test_load_distributed_builds_expected_tasks(monkeypatch):
    loader = MultiprocessDataLoader(worker_fn=sample_worker, num_workers=1)
    captured = {}

    def fake_distribute(tasks, **kwargs):
        captured["tasks"] = tasks
        captured["kwargs"] = kwargs
        return ["ok"]

    monkeypatch.setattr(loader, "distribute_load", fake_distribute)

    result = loader.load_distributed(
        start_date="2024-01-01", end_date="2024-01-02", frequency="1d", priority="high"
    )

    assert result == ["ok"]
    assert captured["tasks"] == [
        {"start_date": "2024-01-01", "end_date": "2024-01-02", "frequency": "1d"}
    ]
    assert captured["kwargs"] == {"priority": "high"}


def test_get_node_info_reports_worker_metadata():
    loader = MultiprocessDataLoader(worker_fn=sample_worker, num_workers=4)

    info = loader.get_node_info()

    assert info["node_type"] == "multiprocess"
    assert info["num_workers"] == 4
    assert info["worker_function"] == "sample_worker"


def test_get_cluster_status_exposes_basic_stats():
    loader = MultiprocessDataLoader(worker_fn=sample_worker, num_workers=3)

    status = loader.get_cluster_status()

    assert status["status"] == "active"
    assert status["node_count"] == 1
    assert status["worker_count"] == 3

