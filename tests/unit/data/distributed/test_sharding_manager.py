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
from datetime import datetime
from types import ModuleType

import pytest

if "src.models" not in sys.modules:
    models_module = ModuleType("src.models")

    class _DummyModel:
        def __init__(self, *args, **kwargs):
            self.data = kwargs.get("data")
            self.metadata = kwargs.get("metadata", {})

        def get_metadata(self, user_only: bool = False):
            return dict(self.metadata)

    models_module.DataModel = _DummyModel
    models_module.SimpleDataModel = _DummyModel
    sys.modules["src.models"] = models_module

if "src.interfaces" not in sys.modules:
    interfaces_module = ModuleType("src.interfaces")
    interfaces_module.IDistributedDataLoader = object
    sys.modules["src.interfaces"] = interfaces_module

from src.data.distributed.sharding_manager import (
    DataShardingManager,
    ShardInfo,
    ShardingStrategy,
)


@pytest.mark.asyncio
async def test_hash_based_sharding_respects_parameters():
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="orders",
        parameters={"symbol": "AAA"},
        strategy=ShardingStrategy.HASH_BASED,
        shard_parameters={"num_shards": 3, "shard_key": "symbol"},
    )

    assert len(shards) == 3
    assert shards[0]["parameters"]["total_shards"] == 3
    assert shards[1]["metadata"]["shard_key"] == "symbol"


@pytest.mark.asyncio
async def test_range_based_sharding_uses_default_ranges():
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="history",
        parameters={"market": "CN"},
        strategy=ShardingStrategy.RANGE_BASED,
    )

    assert len(shards) == 4
    assert shards[0]["metadata"]["range"]["start"].strip().startswith("2020")
    assert shards[-1]["metadata"]["range"]["end"].strip().startswith("2024")


@pytest.mark.asyncio
async def test_time_based_sharding_generates_monthly_segments():
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="ticks",
        parameters={"market": "US"},
        strategy=ShardingStrategy.TIME_BASED,
        shard_parameters={
            "time_period": "monthly",
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
        },
    )

    assert len(shards) == 3
    assert shards[0]["metadata"]["start_date"] == "2024-01-01"
    assert shards[-1]["metadata"]["end_date"] == "2024-04-01"


@pytest.mark.asyncio
async def test_time_based_sharding_quarterly_and_yearly_support():
    manager = DataShardingManager()

    quarterly = await manager.create_shards(
        data_source="ticks",
        parameters={"market": "US"},
        strategy=ShardingStrategy.TIME_BASED,
        shard_parameters={
            "time_period": "quarterly",
            "start_date": "2024-01-01",
            "end_date": "2024-04-01",
        },
    )
    assert len(quarterly) == 1
    assert quarterly[0]["metadata"]["end_date"] == "2024-04-01"

    yearly = await manager.create_shards(
        data_source="ticks",
        parameters={"market": "US"},
        strategy=ShardingStrategy.TIME_BASED,
        shard_parameters={
            "time_period": "yearly",
            "start_date": "2024-01-01",
            "end_date": "2026-01-01",
        },
    )
    assert len(yearly) == 2
    assert yearly[-1]["metadata"]["end_date"] == "2026-01-01"


@pytest.mark.asyncio
async def test_time_based_sharding_falls_back_for_unknown_period():
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="ticks",
        parameters={"market": "US"},
        strategy=ShardingStrategy.TIME_BASED,
        shard_parameters={
            "time_period": "weekly-ish",
            "start_date": "2024-01-01",
            "end_date": "2024-02-01",
        },
    )

    # fallback应仍按月分片
    assert len(shards) == 1
    assert shards[0]["metadata"]["time_period"] == "weekly-ish"
    assert shards[0]["metadata"]["end_date"] == "2024-02-01"


def test_get_shard_by_key_is_deterministic():
    manager = DataShardingManager()

    first = manager.get_shard_by_key("alpha", 5)
    second = manager.get_shard_by_key("alpha", 5)

    assert first == second
    assert 0 <= first < 5


def test_register_and_unregister_shard_records_metadata():
    manager = DataShardingManager()
    shard = ShardInfo(
        shard_id="shard-1",
        data_source="orders",
        approach=ShardingStrategy.CUSTOM,
        parameters={"range": "A-M"},
        assigned_nodes=["node-1"],
        status="ready",
        created_at=datetime.utcnow(),
        metadata={"owner": "qa"},
    )

    manager.register_shard(shard)
    assert manager.get_shard_info("shard-1") is shard
    assert manager.get_shards_by_data_source("orders") == [shard]

    manager.unregister_shard("shard-1")
    assert manager.get_shard_info("shard-1") is None
    assert manager.get_all_shards() == []
    manager.register_shard(shard)
    assert manager.approach(ShardingStrategy.CUSTOM) == [shard]
    manager.unregister_shard("shard-1")


@pytest.mark.asyncio
async def test_create_shards_rejects_unsupported_strategy():
    manager = DataShardingManager()

    with pytest.raises(ValueError):
        await manager.create_shards(
            data_source="orders",
            parameters={},
            strategy=ShardingStrategy.CUSTOM,
        )

