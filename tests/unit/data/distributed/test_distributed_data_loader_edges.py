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
import pytest
from datetime import datetime, timedelta

from src.data.distributed.distributed_data_loader import (
    DistributedDataLoader,
    NodeInfo,
    NodeStatus,
    LoadBalancer,
)
from src.data.distributed.load_balancer import LoadBalancingStrategy


def _make_node(node_id: str, status: NodeStatus = NodeStatus.ONLINE, active: int = 0, max_tasks: int = 2,
               cpu: float = 0.1, mem: float = 0.1) -> NodeInfo:
    return NodeInfo(
        node_id=node_id,
        host="127.0.0.1",
        port=8000,
        status=status,
        cpu_usage=cpu,
        memory_usage=mem,
        active_tasks=active,
        max_tasks=max_tasks,
        last_heartbeat=datetime.now(),
    )


@pytest.mark.asyncio
async def test_select_node_no_available_raises():
    ddl = DistributedDataLoader()
    # 创建任务以便调用选择逻辑
    tid = ddl._create_task("ds", {}, 1)
    with pytest.raises(RuntimeError):
        await ddl._select_node_for_task(tid)
    ddl.shutdown()


@pytest.mark.asyncio
async def test_end_to_end_load_with_fast_sleep(monkeypatch):
    ddl = DistributedDataLoader()
    # 注册一个在线节点
    ddl.register_node(_make_node("n1"))
    # 加速异步sleep
    async def fast_sleep(_secs):
        return None
    monkeypatch.setattr(asyncio, "sleep", fast_sleep, raising=True)
    result = await ddl.load_data_distributed("market_ds", {"window": 5}, priority=1)
    assert result is not None
    # 集群状态包含已完成任务
    st = ddl.get_cluster_status()
    assert st["stats"]["completed_tasks"] >= 1
    ddl.shutdown()


@pytest.mark.asyncio
async def test_handle_task_failure_and_stats():
    ddl = DistributedDataLoader()
    ddl.register_node(_make_node("n1"))
    tid = ddl._create_task("ds", {}, 1)
    # 分配任务到节点，再触发失败
    await ddl._assign_task_to_node(tid, "n1")
    await ddl._handle_task_failure(tid, "boom")
    st = ddl.get_cluster_status()
    assert st["stats"]["failed_tasks"] >= 1
    assert st["tasks"][tid] == "failed"
    ddl.shutdown()


def test_load_balancer_strategies_selection():
    # Round Robin
    lb = LoadBalancer(strategy=LoadBalancingStrategy.ROUND_ROBIN)
    nodes = {"a": _make_node("a"), "b": _make_node("b")}
    order = [lb.select_node(["a", "b"], nodes) for _ in range(3)]
    assert order[0] == "a" and order[1] == "b" and order[2] == "a"
    # Least Connections
    lb2 = LoadBalancer(strategy=LoadBalancingStrategy.LEAST_CONNECTIONS)
    nodes2 = {"a": _make_node("a", active=3), "b": _make_node("b", active=1)}
    assert lb2.select_node(["a", "b"], nodes2) == "b"
    # Weighted Round Robin（根据cpu+mem权重挑选更“轻”的）
    lb3 = LoadBalancer(strategy=LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN)
    nodes3 = {"a": _make_node("a", cpu=0.9, mem=0.9), "b": _make_node("b", cpu=0.1, mem=0.1)}
    assert lb3.select_node(["a", "b"], nodes3) == "b"


def test_monitor_thread_exits_without_nodes(monkeypatch):
    ddl = DistributedDataLoader()
    # 无节点场景，监控线程应尽快退出，不阻塞关闭
    ddl.shutdown()
    # 多次调用也应安全
    ddl.shutdown()


