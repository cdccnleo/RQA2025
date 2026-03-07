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


import asyncio
import importlib
import sys
import time
from datetime import datetime, timedelta
from types import ModuleType, SimpleNamespace

import pytest

if "src.models" not in sys.modules:
    models_module = ModuleType("src.models")

    class _DummyModel:
        def __init__(self, data=None, frequency="1d", metadata=None, **kwargs):
            self.data = data
            self.frequency = frequency
            self.metadata = metadata or {}

        def get_metadata(self, user_only=False):
            return dict(self.metadata)

    models_module.DataModel = _DummyModel
    models_module.SimpleDataModel = _DummyModel
    sys.modules["src.models"] = models_module

if "src.interfaces" not in sys.modules:
    interfaces_module = ModuleType("src.interfaces")
    interfaces_module.IDistributedDataLoader = object
    sys.modules["src.interfaces"] = interfaces_module

from src.data.distributed.distributed_data_loader import (
    DistributedDataLoader,
    LoadBalancingStrategy,
    NodeInfo,
    NodeStatus,
    TaskStatus,
)


@pytest.fixture
def loader(monkeypatch):
    def noop_start(self):
        self._start_time = time.time()
        self._monitor_thread = None

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.DistributedDataLoader._start_monitoring_thread",
        noop_start,
    )
    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.np",
        SimpleNamespace(
            secrets=SimpleNamespace(
                randn=lambda n: [0.0] * n,
                uniform=lambda *args, **kwargs: [0.0],
            )
        ),
        raising=False,
    )
    return DistributedDataLoader(
        config={"load_balancing_strategy": LoadBalancingStrategy.ROUND_ROBIN}
    )


def make_node(node_id="node-1", status=NodeStatus.ONLINE, active_tasks=0):
    return NodeInfo(
        node_id=node_id,
        host="localhost",
        port=6379,
        status=status,
        cpu_usage=0.1,
        memory_usage=0.1,
        active_tasks=active_tasks,
        max_tasks=5,
        last_heartbeat=datetime.now(),
    )


def test_create_task_registers_in_queue(loader):
    task_id = loader._create_task("market", {"symbol": "AAA"}, priority=3)
    assert task_id in loader.tasks
    assert loader.tasks[task_id].priority == 3
    assert loader.task_queue[-1] == task_id
    assert loader.stats["total_tasks"] == 1


def test_register_node_updates_stats(loader):
    node = make_node()
    loader.register_node(node)
    assert loader.nodes[node.node_id] is node
    assert loader.stats["total_nodes"] == 1
    assert loader.stats["active_nodes"] == 1


def test_get_cluster_status_reflects_nodes_and_tasks(loader):
    node = make_node(node_id="node-A")
    loader.register_node(node)
    task_id = loader._create_task("source", {}, priority=1)
    loader.tasks[task_id].status = TaskStatus.RUNNING

    status = loader.get_cluster_status()
    assert status["nodes"]["node-A"] == NodeStatus.ONLINE.value
    assert status["tasks"][task_id] == TaskStatus.RUNNING.value
    assert status["pending_tasks"] == 0
    assert status["running_tasks"] == 1


@pytest.mark.asyncio
async def test_load_data_distributed_completes_with_stubs(loader, monkeypatch):
    loader.register_node(make_node("node-1"))

    async def fake_select(task_id):
        return "node-1"

    async def fake_assign(task_id, node_id):
        loader.tasks[task_id].assigned_node = node_id
        loader.tasks[task_id].status = TaskStatus.RUNNING

    async def fake_execute(task_id):
        loader.stats["completed_tasks"] += 1
        return SimpleNamespace(data="ok", metadata={"task_id": task_id})

    loader._select_node_for_task = fake_select
    loader._assign_task_to_node = fake_assign
    loader._execute_task = fake_execute

    result = await loader.load_data_distributed("test", {"param": 1})
    assert result.data == "ok"
    assert loader.stats["completed_tasks"] == 1


@pytest.mark.asyncio
async def test_handle_task_failure_marks_task(loader):
    loader.register_node(make_node("node-fail", active_tasks=1))
    task_id = loader._create_task("fail_source", {}, priority=1)
    loader.tasks[task_id].assigned_node = "node-fail"

    await loader._handle_task_failure(task_id, "boom")
    assert loader.tasks[task_id].status == TaskStatus.FAILED
    assert loader.tasks[task_id].error == "boom"
    assert loader.stats["failed_tasks"] == 1
    assert loader.nodes["node-fail"].active_tasks == 0


@pytest.mark.asyncio
async def test_select_node_round_robin(loader):
    loader.register_node(make_node("node-1"))
    loader.register_node(make_node("node-2"))
    task_id = loader._create_task("src", {}, priority=1)

    selected_first = await loader._select_node_for_task(task_id)
    selected_second = await loader._select_node_for_task(task_id)

    assert selected_first == "node-1"
    assert selected_second == "node-2"


@pytest.mark.asyncio
async def test_select_node_without_available_nodes_raises(loader):
    task_id = loader._create_task("src", {}, priority=1)
    with pytest.raises(RuntimeError):
        await loader._select_node_for_task(task_id)


@pytest.mark.asyncio
async def test_assign_task_updates_node_state(loader):
    node = make_node("node-assign")
    loader.register_node(node)
    task_id = loader._create_task("assign_src", {}, priority=1)

    await loader._assign_task_to_node(task_id, "node-assign")
    assert loader.tasks[task_id].assigned_node == "node-assign"
    assert loader.tasks[task_id].status == TaskStatus.RUNNING
    assert loader.nodes["node-assign"].active_tasks == 1


@pytest.mark.asyncio
async def test_execute_task_success_updates_stats(loader, monkeypatch):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.asyncio.sleep",
        fast_sleep,
    )
    node = make_node("node-run", active_tasks=1)
    loader.register_node(node)
    task_id = loader._create_task("exec", {}, priority=1)
    loader.tasks[task_id].assigned_node = "node-run"

    result = await loader._execute_task(task_id)
    assert loader.tasks[task_id].status == TaskStatus.COMPLETED
    assert loader.nodes["node-run"].active_tasks == 0
    assert loader.stats["completed_tasks"] == 1
    assert result is loader.tasks[task_id].result


@pytest.mark.asyncio
async def test_execute_task_failure_triggers_handler(loader, monkeypatch):
    async def fast_sleep(_):
        return None

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.asyncio.sleep",
        fast_sleep,
    )

    def failing_dataframe(*args, **kwargs):
        raise RuntimeError("df boom")

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.pd.DataFrame",
        failing_dataframe,
    )

    called = {}

    async def capture_failure(task_id, error):
        called["task_id"] = task_id
        called["error"] = error

    monkeypatch.setattr(
        loader,
        "_handle_task_failure",
        capture_failure,
    )

    node = make_node("node-failure", active_tasks=1)
    loader.register_node(node)
    task_id = loader._create_task("exec_fail", {}, priority=1)
    loader.tasks[task_id].assigned_node = "node-failure"

    with pytest.raises(RuntimeError):
        await loader._execute_task(task_id)

    assert called["task_id"] == task_id
    assert "df boom" in called["error"]


def test_update_stats_computes_average_and_throughput(loader, monkeypatch):
    loader.stats["completed_tasks"] = 2
    loader.stats["average_response_time"] = 1.0
    loader._start_time = time.time() - 10

    loader._update_stats(2.0)
    assert loader.stats["average_response_time"] == pytest.approx(1.5)
    assert loader.stats["throughput"] == pytest.approx(
        loader.stats["completed_tasks"] / 10, rel=1e-2
    )


def test_check_node_health_marks_offline(loader):
    stale_node = make_node("stale")
    stale_node.last_heartbeat = datetime.now() - timedelta(minutes=10)
    loader.register_node(stale_node)
    loader._check_node_health()
    assert loader.nodes["stale"].status == NodeStatus.OFFLINE


def test_update_monitoring_stats_counts_active_nodes(loader):
    loader.register_node(make_node("n1", status=NodeStatus.ONLINE))
    loader.register_node(make_node("n2", status=NodeStatus.OFFLINE))
    loader._update_monitoring_stats()
    assert loader.stats["active_nodes"] == 1


def test_start_monitoring_thread_invokes_checks(monkeypatch):
    module = importlib.reload(
        sys.modules["src.data.distributed.distributed_data_loader"]
    )
    loader = module.DistributedDataLoader(
        config={"load_balancing_strategy": LoadBalancingStrategy.ROUND_ROBIN}
    )
    loader.register_node(
        module.NodeInfo(
            node_id="monitor-node",
            host="localhost",
            port=1,
            status=module.NodeStatus.ONLINE,
            cpu_usage=0.1,
            memory_usage=0.1,
            active_tasks=0,
            max_tasks=1,
            last_heartbeat=datetime.now(),
        )
    )
    calls = {"health": 0, "stats": 0}

    def fake_thread(target, daemon=True):
        class DummyThread:
            def __init__(self):
                self._target = target

            def start(self):
                self._target()

            def is_alive(self):
                return False

        return DummyThread()

    monkeypatch.setattr(module.threading, "Thread", fake_thread)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)

    def health_wrapper():
        calls["health"] += 1

    def stats_wrapper():
        calls["stats"] += 1
        loader._stop_monitoring = True

    monkeypatch.setattr(loader, "_check_node_health", health_wrapper, raising=False)
    monkeypatch.setattr(loader, "_update_monitoring_stats", stats_wrapper, raising=False)

    loader._start_monitoring_thread()
    loader.shutdown()
    assert calls["health"] == 1
    assert calls["stats"] >= 1


def test_shutdown_joins_monitor_thread(loader, monkeypatch):
    loader._monitor_thread = SimpleNamespace(
        is_alive=lambda: True,
        join=lambda timeout=None: None,
    )
    loader.shutdown()
    assert loader._stop_monitoring is True


def test_load_balancer_strategies():
    lb = DistributedDataLoader().load_balancer
    nodes = {
        "n1": make_node("n1", active_tasks=2),
        "n2": make_node("n2", active_tasks=1),
    }
    available = ["n1", "n2"]
    assert lb.select_node(available, nodes) == "n1"
    assert lb.select_node(available, nodes) == "n2"

    lb.strategy = LoadBalancingStrategy.LEAST_CONNECTIONS
    assert lb.select_node(available, nodes) == "n2"

    lb.strategy = LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN
    nodes["n1"].cpu_usage = 0.9
    nodes["n1"].memory_usage = 0.9
    nodes["n2"].cpu_usage = 0.1
    nodes["n2"].memory_usage = 0.1
    assert lb.select_node(available, nodes) == "n2"


def test_load_balancer_stats_tracking():
    lb = DistributedDataLoader().load_balancer
    lb.update_node_stats("node", response_time=0.5, success=True)
    lb.update_node_stats("node", response_time=1.5, success=False)
    stats = lb.get_node_stats("node")
    assert stats["total_requests"] == 2
    assert stats["successful_requests"] == 1
    assert stats["failed_requests"] == 1
    assert stats["average_response_time"] == pytest.approx(1.0)


def test_loader_init_handles_logger_failure(monkeypatch):
    def noop_start(self):
        self._start_time = time.time()
        self._monitor_thread = None

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.DistributedDataLoader._start_monitoring_thread",
        noop_start,
    )

    logged = {"count": 0}

    def boom(*args, **kwargs):
        logged["count"] += 1
        raise RuntimeError("log failure")

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.logger.info", boom
    )

    loader = DistributedDataLoader()
    # 初始化过程中可能有多个地方调用logger.info，至少验证调用了一次
    assert logged["count"] >= 1
    loader.shutdown()


def test_loader_del_swallows_shutdown_error(monkeypatch):
    def noop_start(self):
        self._start_time = time.time()
        self._monitor_thread = None

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.DistributedDataLoader._start_monitoring_thread",
        noop_start,
    )
    loader = DistributedDataLoader()

    def boom():
        raise RuntimeError("shutdown failure")

    monkeypatch.setattr(loader, "shutdown", boom)
    loader.__del__()  # should not raise


def test_create_distributed_data_loader_factory():
    from src.data.distributed import distributed_data_loader as module

    loader = module.create_distributed_data_loader(
        {"load_balancing_strategy": LoadBalancingStrategy.LEAST_CONNECTIONS}
    )
    assert isinstance(loader, module.DistributedDataLoader)
    assert loader.load_balancer.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS


@pytest.mark.asyncio
async def test_convenience_load_data_function(monkeypatch):
    from src.data.distributed import distributed_data_loader as module

    shutdown_called = {"value": False}

    class DummyLoader:
        def __init__(self):
            self.shutdown = lambda: shutdown_called.update(value=True)

        async def load_data_distributed(self, *args, **kwargs):
            return "ok"

    monkeypatch.setattr(
        module,
        "create_distributed_data_loader",
        lambda config=None: DummyLoader(),
    )

    result = await module.load_data_distributed("src", {"a": 1})
    assert result == "ok"
    assert shutdown_called["value"] is True


def test_monitoring_thread_handles_internal_exception(monkeypatch):
    module = importlib.reload(
        sys.modules["src.data.distributed.distributed_data_loader"]
    )

    def fake_thread(target, daemon=True):
        class DummyThread:
            def __init__(self):
                self._target = target

            def start(self):
                self._target()

            def is_alive(self):
                return False

        return DummyThread()

    monkeypatch.setattr(module.threading, "Thread", fake_thread)
    monkeypatch.setattr(module.time, "sleep", lambda *_: None)

    loader = module.DistributedDataLoader(
        config={"load_balancing_strategy": module.LoadBalancingStrategy.ROUND_ROBIN}
    )
    loader.register_node(
        module.NodeInfo(
            node_id="monitor-node",
            host="localhost",
            port=1,
            status=module.NodeStatus.ONLINE,
            cpu_usage=0.1,
            memory_usage=0.1,
            active_tasks=0,
            max_tasks=1,
            last_heartbeat=datetime.now(),
        )
    )

    errors = {"count": 0}

    def failing_health():
        loader._stop_monitoring = True
        raise RuntimeError("boom")

    def capture_error(message):
        errors["count"] += 1

    monkeypatch.setattr(loader, "_check_node_health", failing_health, raising=False)
    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.logger.error",
        capture_error,
        raising=False,
    )

    loader._start_monitoring_thread()
    loader.shutdown()
    assert errors["count"] >= 1


def test_check_node_health_no_nodes(loader):
    loader.nodes.clear()
    loader._check_node_health()  # should simply return


def test_check_node_health_handles_logger_warning_failure(loader, monkeypatch):
    stale_node = make_node("stale")
    stale_node.last_heartbeat = datetime.now() - timedelta(minutes=10)
    loader.register_node(stale_node)

    def boom(*args, **kwargs):
        raise RuntimeError("warn failure")

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.logger.warning",
        boom,
    )

    loader._check_node_health()
    assert loader.nodes["stale"].status == NodeStatus.OFFLINE


def test_shutdown_handles_join_and_logger_errors(loader, monkeypatch):
    class DummyThread:
        def is_alive(self):
            return True

        def join(self, timeout=None):
            raise RuntimeError("join failure")

    loader._monitor_thread = DummyThread()

    def boom(*args, **kwargs):
        raise RuntimeError("shutdown log")

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.logger.info",
        boom,
    )

    loader.shutdown()
    assert loader._stop_monitoring is True


def test_load_balancer_select_node_handles_no_nodes(monkeypatch):
    def noop_start(self):
        self._start_time = time.time()
        self._monitor_thread = None

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.DistributedDataLoader._start_monitoring_thread",
        noop_start,
    )
    lb = DistributedDataLoader().load_balancer

    with pytest.raises(ValueError):
        lb.select_node([], {})

    with pytest.raises(ValueError):
        lb._round_robin_select([])

    lb.strategy = None
    assert lb.select_node(["node-x"], {}) == "node-x"


def test_load_balancer_get_stats_without_updates(monkeypatch):
    def noop_start(self):
        self._start_time = time.time()
        self._monitor_thread = None

    monkeypatch.setattr(
        "src.data.distributed.distributed_data_loader.DistributedDataLoader._start_monitoring_thread",
        noop_start,
    )
    lb = DistributedDataLoader().load_balancer
    assert lb.get_node_stats("missing") is None

