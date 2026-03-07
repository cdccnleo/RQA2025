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
import time
import pandas as pd
import pytest
from datetime import datetime
from src.data.distributed.distributed_data_loader import (
    DistributedDataLoader, NodeInfo, NodeStatus
)


def make_node(node_id: str) -> NodeInfo:
    return NodeInfo(
        node_id=node_id,
        host="127.0.0.1",
        port=8000,
        status=NodeStatus.ONLINE,
        cpu_usage=0.1,
        memory_usage=0.1,
        active_tasks=0,
        max_tasks=1,
        last_heartbeat=datetime.now(),
    )


def test_monitor_thread_exits_quickly_when_no_nodes_and_shutdown_is_fast():
    loader = DistributedDataLoader(config={})
    # 未注册任何节点，监控线程应很快退出；shutdown 不应阻塞
    t0 = time.time()
    loader.shutdown()
    assert time.time() - t0 < 1.5


def test_handle_task_failure_releases_node_counter_and_not_negative(monkeypatch):
    loader = DistributedDataLoader(config={})
    try:
        loader.register_node(make_node("n1"))
        real_df = pd.DataFrame
        def boom(*args, **kwargs):
            raise RuntimeError("df-boom-2")
        monkeypatch.setattr(pd, "DataFrame", boom)
        async def _run():
            with pytest.raises(RuntimeError, match="df-boom-2"):
                await loader.load_data_distributed("s", {})
        asyncio.get_event_loop().run_until_complete(_run())
        # active_tasks 应回收为0
        assert loader.nodes["n1"].active_tasks == 0
        # 恢复
        monkeypatch.setattr(pd, "DataFrame", real_df)
    finally:
        loader.shutdown()


