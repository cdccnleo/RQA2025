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

        def get_metadata(self, user_only=False):
            return dict(self.metadata)

    models_module.DataModel = _DummyModel
    models_module.SimpleDataModel = _DummyModel
    sys.modules["src.models"] = models_module

if "src.interfaces" not in sys.modules:
    interfaces_module = ModuleType("src.interfaces")
    interfaces_module.IDistributedDataLoader = object
    sys.modules["src.interfaces"] = interfaces_module

from src.data.distributed.load_balancer import LoadBalancer, LoadBalancingStrategy


def test_round_robin_selection_cycles_through_nodes():
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
    nodes = ["node-a", "node-b", "node-c"]

    selections = [balancer.select_node(nodes, {}) for _ in range(4)]

    assert selections == ["node-a", "node-b", "node-c", "node-a"]


def test_least_connections_prefers_lowest_active_tasks():
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    nodes = {
        "node-a": {"active_tasks": 3},
        "node-b": {"active_tasks": 1},
        "node-c": {"active_tasks": 2},
    }

    selected = balancer.select_node(list(nodes), nodes)

    assert selected == "node-b"


def test_weighted_round_robin_prefers_lower_usage_nodes():
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
    nodes = {
        "heavy": {"cpu_usage": 0.9, "memory_usage": 0.9},
        "light": {"cpu_usage": 0.1, "memory_usage": 0.1},
    }

    selected = balancer.select_node(list(nodes), nodes)

    assert selected == "light"


def test_least_response_time_prefers_faster_node():
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_RESPONSE_TIME)
    available = ["slow", "fast"]
    balancer.update_node_stats("slow", response_time=0.8)
    balancer.update_node_stats("fast", response_time=0.2)

    selected = balancer.select_node(available, nodes={})

    assert selected == "fast"


def test_random_strategy_uses_secrets_choice(monkeypatch):
    chosen = {"value": None}

    def fake_choice(options):
        chosen["value"] = options[-1]
        return options[-1]

    monkeypatch.setattr(
        "src.data.distributed.load_balancer.secrets.choice", fake_choice
    )
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.RANDOM)

    selected = balancer.select_node(["left", "right"], nodes={})

    assert selected == "right"
    assert chosen["value"] == "right"


def test_update_and_reset_node_stats():
    balancer = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
    balancer.update_node_stats("node-a", response_time=0.4, success=True)
    balancer.update_node_stats("node-a", response_time=0.6, success=False)

    stats = balancer.get_node_stats("node-a")
    assert stats is not None
    assert pytest.approx(stats["average_response_time"], rel=1e-3) == 0.5
    assert stats["successful_requests"] == 1
    assert stats["failed_requests"] == 1

    balancer.reset_node_stats("node-a")
    assert balancer.get_node_stats("node-a") is None

    balancer.update_node_stats("node-b", response_time=0.3, success=True)
    balancer.reset_node_stats()
    assert balancer.get_all_node_stats() == {}

