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
from datetime import datetime
from src.data.distributed.distributed_data_loader import (
    DistributedDataLoader, NodeInfo, NodeStatus
)


def _run(coro):
    """运行异步函数的辅助函数，兼容并行执行"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


def make_node(node_id: str, max_tasks: int = 1) -> NodeInfo:
    return NodeInfo(
        node_id=node_id,
        host="127.0.0.1",
        port=8000,
        status=NodeStatus.ONLINE,
        cpu_usage=0.1,
        memory_usage=0.1,
        active_tasks=0,
        max_tasks=max_tasks,
        last_heartbeat=datetime.now(),
    )


def test_two_concurrent_tasks_distributed_to_two_nodes_and_counters_released():
    loader = DistributedDataLoader(config={})
    try:
        loader.register_node(make_node("n1", max_tasks=1))
        loader.register_node(make_node("n2", max_tasks=1))

        async def run_two():
            t1 = asyncio.create_task(loader.load_data_distributed("s1", {}))
            t2 = asyncio.create_task(loader.load_data_distributed("s2", {}))
            r1, r2 = await asyncio.gather(t1, t2)
            return r1, r2

        r1, r2 = _run(run_two())
        assert r1 is not None and r2 is not None
        # active_tasks 应回收为0
        for n in loader.nodes.values():
            assert n.active_tasks == 0
        st = loader.get_cluster_status()["stats"]
        assert st["completed_tasks"] == 2
    finally:
        loader.shutdown()


