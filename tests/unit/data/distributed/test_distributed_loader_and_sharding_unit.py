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
import sys
import pandas as pd
import pytest


# 先注入最小的 src.models 以满足被测模块导入
models_stub = type(sys)("src.models")

class _SimpleDataModel:
    def __init__(self, data=None, metadata=None):
        self.data = data
        self.metadata = metadata or {}

# 兼容被测文件中使用的名称
models_stub.DataModel = _SimpleDataModel
models_stub.SimpleDataModel = _SimpleDataModel
sys.modules.setdefault("src.models", models_stub)

# 注入最小的 src.interfaces 以满足 distributed 包 __init__ 中的依赖
interfaces_stub = type(sys)("src.interfaces")
class IDistributedDataLoader:  # 占位接口
    pass
interfaces_stub.IDistributedDataLoader = IDistributedDataLoader
sys.modules.setdefault("src.interfaces", interfaces_stub)

from src.data.distributed.distributed_data_loader import (  # noqa: E402
    DistributedDataLoader,
    NodeInfo,
    NodeStatus,
    LoadBalancer,
)
from src.data.distributed.sharding_manager import (  # noqa: E402
    DataShardingManager,
    ShardingStrategy,
)


@pytest.mark.asyncio
async def test_distributed_loader_round_robin_and_execution(monkeypatch):
    loader = DistributedDataLoader()
    try:
        now = datetime.now()
        # 注册两个在线节点
        loader.register_node(NodeInfo(
            node_id="n1", host="h1", port=1,
            status=NodeStatus.ONLINE, cpu_usage=0.1, memory_usage=0.1,
            active_tasks=0, max_tasks=4, last_heartbeat=now
        ))
        loader.register_node(NodeInfo(
            node_id="n2", host="h2", port=2,
            status=NodeStatus.ONLINE, cpu_usage=0.1, memory_usage=0.2,
            active_tasks=0, max_tasks=4, last_heartbeat=now
        ))

        # 加速：避免真实 1s sleep
        async def _fast_sleep(*_args, **_kwargs):
            return None
        monkeypatch.setattr("src.data.distributed.distributed_data_loader.asyncio.sleep", _fast_sleep)

        result = await loader.load_data_distributed("sourceX", {"q": 1})
        assert hasattr(result, "data")
        df = result.data
        assert isinstance(df, pd.DataFrame) and not df.empty

        status = loader.get_cluster_status()
        assert status["stats"]["completed_tasks"] >= 1
        assert status["active_nodes"] >= 1
    finally:
        loader.shutdown()


@pytest.mark.asyncio
async def test_sharding_manager_hash_range_time():
    mgr = DataShardingManager()

    # hash-based
    shards_h = await mgr.create_shards("ds", {"k": 1}, ShardingStrategy.HASH_BASED, {"num_shards": 3, "shard_key": "id"})
    assert len(shards_h) == 3
    idx = mgr.get_shard_by_key("abc", 3)
    assert 0 <= idx < 3

    # range-based
    shards_r = await mgr.create_shards("ds", {"k": 1}, ShardingStrategy.RANGE_BASED, {"ranges": [{"start": "2024-01-01", "end": "2024-06-01"}]})
    assert len(shards_r) == 1

    # time-based
    shards_t = await mgr.create_shards("ds", {"k": 1}, ShardingStrategy.TIME_BASED, {"time_period": "monthly", "start_date": "2024-01-01", "end_date": "2024-03-01"})
    assert len(shards_t) >= 2


