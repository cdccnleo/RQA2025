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
from datetime import datetime, timedelta
from src.data.distributed.distributed_data_loader import (
    DistributedDataLoader, NodeInfo, NodeStatus
)


def make_node(node_id: str, active: int = 0, max_tasks: int = 2, status: NodeStatus = NodeStatus.ONLINE) -> NodeInfo:
    return NodeInfo(
        node_id=node_id,
        host="127.0.0.1",
        port=8000,
        status=status,
        cpu_usage=0.1,
        memory_usage=0.1,
        active_tasks=active,
        max_tasks=max_tasks,
        last_heartbeat=datetime.now(),
    )


def test_basic_success_flow_updates_stats_and_task_states():
    loader = DistributedDataLoader(config={})
    try:
        # 注册两个在线节点
        loader.register_node(make_node("n1"))
        loader.register_node(make_node("n2"))

        async def _run():
            result = await loader.load_data_distributed("ds", {"p": 1})
            return result

        result = asyncio.get_event_loop().run_until_complete(_run())
        assert result is not None
        # 统计与任务状态
        status = loader.get_cluster_status()
        assert status["stats"]["total_tasks"] == 1
        assert status["stats"]["completed_tasks"] == 1
        # active_tasks 应回收为0
        for n in loader.nodes.values():
            assert n.active_tasks == 0
    finally:
        loader.shutdown()


def test_health_check_marks_offline_on_heartbeat_timeout():
    loader = DistributedDataLoader(config={})
    try:
        old = datetime.now() - timedelta(minutes=10)
        node = make_node("n1")
        node.last_heartbeat = old
        loader.register_node(node)
        # 触发一次内部健康检查
        loader._check_node_health()  # type: ignore[attr-defined]
        assert loader.nodes["n1"].status == NodeStatus.OFFLINE
    finally:
        loader.shutdown()


