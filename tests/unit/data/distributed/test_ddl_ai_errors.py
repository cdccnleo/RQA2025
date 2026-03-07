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
import builtins
import types
import pandas as pd
import pytest
from datetime import datetime
def _run(coro):
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)
from src.data.distributed.distributed_data_loader import (
    DistributedDataLoader, NodeInfo, NodeStatus
)


def make_node(node_id: str, active: int = 0, max_tasks: int = 1, status: NodeStatus = NodeStatus.ONLINE) -> NodeInfo:
    return NodeInfo(
        node_id=node_id,
        host="127.0.0.1",
        port=8000,
        status=status,
        cpu_usage=0.2,
        memory_usage=0.2,
        active_tasks=active,
        max_tasks=max_tasks,
        last_heartbeat=datetime.now(),
    )


def test_select_node_raises_when_no_available_nodes():
    loader = DistributedDataLoader(config={})
    try:
        # 未注册节点
        with pytest.raises(RuntimeError, match="No available nodes"):
            _run(loader.load_data_distributed("ds", {}))
    finally:
        loader.shutdown()


def test_execute_task_failure_path_updates_stats(monkeypatch):
    loader = DistributedDataLoader(config={})
    try:
        loader.register_node(make_node("n1", max_tasks=5))

        # 让 pandas.DataFrame 抛出异常以触发失败分支
        real_df = pd.DataFrame
        def boom(*args, **kwargs):
            raise RuntimeError("df-boom")
        monkeypatch.setattr(pd, "DataFrame", boom)

        async def _run():
            with pytest.raises(RuntimeError, match="df-boom"):
                await loader.load_data_distributed("ds", {})

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(_run())
        st = loader.get_cluster_status()["stats"]
        assert st["total_tasks"] == 1
        assert st["failed_tasks"] == 1
        # 恢复以免污染其他测试（保险起见）
        monkeypatch.setattr(pd, "DataFrame", real_df)
    finally:
        loader.shutdown()


