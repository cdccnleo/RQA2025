import importlib
from typing import List

import pytest

from src.infrastructure.cache.distributed import unified_sync
from src.infrastructure.cache.distributed.unified_sync import UnifiedSync


@pytest.fixture(autouse=True)
def reset_global_sync_instance():
    yield
    unified_sync._sync_instance = None


def test_unified_sync_disabled_returns_defaults():
    sync = UnifiedSync(enable_distributed_sync=False)

    assert sync.is_sync_enabled() is False
    assert sync.register_sync_node("node-1", "127.0.0.1", 8000) is False
    assert sync.unregister_sync_node("node-1") is False
    assert sync.start_auto_sync() is False
    assert sync.stop_auto_sync() is False
    assert sync.sync_config_to_nodes(["node-1"]) == {"success": False, "message": "分布式同步功能未启用"}
    assert sync.sync_config_data({"key": "value"}, ["node-1"]) is False
    assert sync.sync_data({"payload": 1}) is False
    assert sync.get_sync_status()["enabled"] is False
    assert sync.get_distributed_sync_status()["enabled"] is False
    assert sync.get_sync_history() == []
    assert sync.get_conflicts() == []
    assert sync.resolve_conflicts() == {"success": False, "message": "分布式同步功能未启用"}
    assert sync.add_sync_callback("event", lambda *_: None) is False
    assert sync.remove_sync_callback("event") is False


def test_unified_sync_basic_operations_with_placeholder_service():
    sync = UnifiedSync(enable_distributed_sync=True)

    assert sync.is_sync_enabled() is True
    assert sync.register_sync_node("node-1", "127.0.0.1", 9000) is True
    assert sync._sync_service.nodes["node-1"]["address"] == "127.0.0.1"

    status = sync.get_sync_status()
    assert status["enabled"] is True
    assert status["nodes"] == 1

    assert sync.start_auto_sync() is True
    assert sync.sync_config_to_nodes(["node-1"])["success"] is True
    assert sync.sync_data({"value": 123}, ["node-1"]) is True
    history = sync.get_sync_history()
    assert history

    callback_triggered: List[str] = []

    def dummy_callback(event=None):
        callback_triggered.append("called")

    assert sync.add_sync_callback("sync_event", dummy_callback) is True
    assert sync.remove_sync_callback("sync_event") is False  # 占位服务按事件名无法移除

    conflicts = sync.get_conflicts()
    assert conflicts == []
    assert sync.resolve_conflicts()["success"] is True

    assert sync.resolve_conflict("conflict-key", {"resolved": True}) is False
    assert sync.stop_auto_sync() is True


def test_global_start_stop_sync():
    module = importlib.reload(unified_sync)

    assert module._sync_instance is None
    assert module.start_sync() is True
    assert isinstance(module._sync_instance, module.UnifiedSync)
    assert module.stop_sync() is True
    assert module._sync_instance is not None  # stop_sync 不会自动清空实例


