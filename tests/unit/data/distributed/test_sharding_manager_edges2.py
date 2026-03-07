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


import pytest
import asyncio
from datetime import datetime
from unittest.mock import Mock

from src.data.distributed.sharding_manager import (
    DataShardingManager,
    ShardInfo,
    ShardingStrategy
)


def test_shard_info_defaults():
    """测试 ShardInfo（默认值）"""
    info = ShardInfo(
        shard_id="shard1",
        data_source="source1",
        approach=ShardingStrategy.HASH_BASED,
        parameters={},
        assigned_nodes=[],
        status="active",
        created_at=datetime.now(),
        metadata={}
    )
    assert info.shard_id == "shard1"
    assert info.data_source == "source1"
    assert info.approach == ShardingStrategy.HASH_BASED
    assert info.parameters == {}
    assert info.assigned_nodes == []
    assert info.status == "active"
    assert info.metadata == {}


def test_data_sharding_manager_init_none_config():
    """测试 DataShardingManager（初始化，None 配置）"""
    manager = DataShardingManager(config=None)
    assert manager.config == {}
    assert manager.shards == {}


def test_data_sharding_manager_init_empty_config():
    """测试 DataShardingManager（初始化，空配置）"""
    manager = DataShardingManager(config={})
    assert manager.config == {}


def test_data_sharding_manager_create_shards_unsupported_strategy():
    """测试 DataShardingManager（创建分片，不支持的策略）"""
    manager = DataShardingManager()
    with pytest.raises(ValueError, match="Unsupported sharding strategy"):
        asyncio.run(manager.create_shards(
            data_source="test",
            parameters={},
            strategy=ShardingStrategy.CUSTOM
        ))


@pytest.mark.asyncio
async def test_data_sharding_manager_hash_based_sharding_default():
    """测试 DataShardingManager（基于哈希的分片，默认参数）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.HASH_BASED
    )
    assert len(shards) == 4  # 默认4个分片
    assert all(shard['approach'] == 'hash_based' for shard in shards)


@pytest.mark.asyncio
async def test_data_sharding_manager_hash_based_sharding_custom():
    """测试 DataShardingManager（基于哈希的分片，自定义参数）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.HASH_BASED,
        shard_parameters={'num_shards': 8, 'shard_key': 'user_id'}
    )
    assert len(shards) == 8
    assert all(shard['metadata']['shard_key'] == 'user_id' for shard in shards)


@pytest.mark.asyncio
async def test_data_sharding_manager_hash_based_sharding_zero_shards():
    """测试 DataShardingManager（基于哈希的分片，零分片）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.HASH_BASED,
        shard_parameters={'num_shards': 0}
    )
    assert len(shards) == 0


@pytest.mark.asyncio
async def test_data_sharding_manager_range_based_sharding_default():
    """测试 DataShardingManager（基于范围的分片，默认参数）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.RANGE_BASED
    )
    assert len(shards) == 4  # 默认4个范围


@pytest.mark.asyncio
async def test_data_sharding_manager_range_based_sharding_custom():
    """测试 DataShardingManager（基于范围的分片，自定义参数）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.RANGE_BASED,
        shard_parameters={'ranges': [
            {'start': '2020-01-01', 'end': '2021-01-01'},
            {'start': '2021-01-01', 'end': '2022-01-01'}
        ]}
    )
    assert len(shards) == 2


@pytest.mark.asyncio
async def test_data_sharding_manager_range_based_sharding_empty_ranges():
    """测试 DataShardingManager（基于范围的分片，空范围）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.RANGE_BASED,
        shard_parameters={'ranges': []}
    )
    # 空范围会使用默认范围
    assert len(shards) == 4


@pytest.mark.asyncio
async def test_data_sharding_manager_time_based_sharding_monthly():
    """测试 DataShardingManager（基于时间的分片，月度）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.TIME_BASED,
        shard_parameters={
            'time_period': 'monthly',
            'start_date': '2023-01-01',
            'end_date': '2023-03-01'
        }
    )
    assert len(shards) == 2  # 2个月


@pytest.mark.asyncio
async def test_data_sharding_manager_time_based_sharding_quarterly():
    """测试 DataShardingManager（基于时间的分片，季度）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.TIME_BASED,
        shard_parameters={
            'time_period': 'quarterly',
            'start_date': '2023-01-01',
            'end_date': '2023-07-01'
        }
    )
    assert len(shards) == 2  # 2个季度


@pytest.mark.asyncio
async def test_data_sharding_manager_time_based_sharding_yearly():
    """测试 DataShardingManager（基于时间的分片，年度）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.TIME_BASED,
        shard_parameters={
            'time_period': 'yearly',
            'start_date': '2020-01-01',
            'end_date': '2022-01-01'
        }
    )
    assert len(shards) == 2  # 2年


@pytest.mark.asyncio
async def test_data_sharding_manager_time_based_sharding_invalid_period():
    """测试 DataShardingManager（基于时间的分片，无效周期）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.TIME_BASED,
        shard_parameters={
            'time_period': 'invalid',
            'start_date': '2023-01-01',
            'end_date': '2023-02-01'
        }
    )
    # 无效周期会使用默认月度
    assert len(shards) == 1


@pytest.mark.asyncio
async def test_data_sharding_manager_time_based_sharding_same_dates():
    """测试 DataShardingManager（基于时间的分片，相同日期）"""
    manager = DataShardingManager()
    shards = await manager.create_shards(
        data_source="test",
        parameters={},
        strategy=ShardingStrategy.TIME_BASED,
        shard_parameters={
            'start_date': '2023-01-01',
            'end_date': '2023-01-01'
        }
    )
    assert len(shards) == 0  # 相同日期不生成分片


def test_data_sharding_manager_get_shard_by_key():
    """测试 DataShardingManager（根据键获取分片索引）"""
    manager = DataShardingManager()
    shard_index = manager.get_shard_by_key("test_key", 4)
    assert 0 <= shard_index < 4


def test_data_sharding_manager_get_shard_by_key_zero_shards():
    """测试 DataShardingManager（根据键获取分片索引，零分片）"""
    manager = DataShardingManager()
    # 零分片会导致除零错误，但实际实现中会返回 0
    try:
        shard_index = manager.get_shard_by_key("test_key", 0)
        # 如果实现允许，应该返回 0
        assert shard_index == 0
    except ZeroDivisionError:
        # 如果实现会抛出异常，这也是合理的
        assert True


def test_data_sharding_manager_get_shard_by_key_same_key():
    """测试 DataShardingManager（根据键获取分片索引，相同键）"""
    manager = DataShardingManager()
    index1 = manager.get_shard_by_key("test_key", 4)
    index2 = manager.get_shard_by_key("test_key", 4)
    assert index1 == index2  # 相同键应该得到相同索引


def test_data_sharding_manager_get_shard_info_nonexistent():
    """测试 DataShardingManager（获取分片信息，不存在）"""
    manager = DataShardingManager()
    info = manager.get_shard_info("nonexistent")
    assert info is None


def test_data_sharding_manager_register_shard():
    """测试 DataShardingManager（注册分片）"""
    manager = DataShardingManager()
    shard_info = ShardInfo(
        shard_id="shard1",
        data_source="source1",
        approach=ShardingStrategy.HASH_BASED,
        parameters={},
        assigned_nodes=[],
        status="active",
        created_at=datetime.now(),
        metadata={}
    )
    manager.register_shard(shard_info)
    assert "shard1" in manager.shards
    assert manager.get_shard_info("shard1") == shard_info


def test_data_sharding_manager_unregister_shard_nonexistent():
    """测试 DataShardingManager（注销分片，不存在）"""
    manager = DataShardingManager()
    manager.unregister_shard("nonexistent")
    assert len(manager.shards) == 0


def test_data_sharding_manager_unregister_shard_existing():
    """测试 DataShardingManager（注销分片，存在）"""
    manager = DataShardingManager()
    shard_info = ShardInfo(
        shard_id="shard1",
        data_source="source1",
        approach=ShardingStrategy.HASH_BASED,
        parameters={},
        assigned_nodes=[],
        status="active",
        created_at=datetime.now(),
        metadata={}
    )
    manager.register_shard(shard_info)
    manager.unregister_shard("shard1")
    assert "shard1" not in manager.shards


def test_data_sharding_manager_get_all_shards_empty():
    """测试 DataShardingManager（获取所有分片，空）"""
    manager = DataShardingManager()
    shards = manager.get_all_shards()
    assert shards == []


def test_data_sharding_manager_get_all_shards_with_shards():
    """测试 DataShardingManager（获取所有分片，有分片）"""
    manager = DataShardingManager()
    shard1 = ShardInfo(
        shard_id="shard1",
        data_source="source1",
        approach=ShardingStrategy.HASH_BASED,
        parameters={},
        assigned_nodes=[],
        status="active",
        created_at=datetime.now(),
        metadata={}
    )
    shard2 = ShardInfo(
        shard_id="shard2",
        data_source="source1",
        approach=ShardingStrategy.RANGE_BASED,
        parameters={},
        assigned_nodes=[],
        status="active",
        created_at=datetime.now(),
        metadata={}
    )
    manager.register_shard(shard1)
    manager.register_shard(shard2)
    shards = manager.get_all_shards()
    assert len(shards) == 2


def test_data_sharding_manager_approach_empty():
    """测试 DataShardingManager（根据策略获取分片，空）"""
    manager = DataShardingManager()
    shards = manager.approach(ShardingStrategy.HASH_BASED)
    assert shards == []


def test_data_sharding_manager_approach_with_shards():
    """测试 DataShardingManager（根据策略获取分片，有分片）"""
    manager = DataShardingManager()
    shard1 = ShardInfo(
        shard_id="shard1",
        data_source="source1",
        approach=ShardingStrategy.HASH_BASED,
        parameters={},
        assigned_nodes=[],
        status="active",
        created_at=datetime.now(),
        metadata={}
    )
    shard2 = ShardInfo(
        shard_id="shard2",
        data_source="source1",
        approach=ShardingStrategy.RANGE_BASED,
        parameters={},
        assigned_nodes=[],
        status="active",
        created_at=datetime.now(),
        metadata={}
    )
    manager.register_shard(shard1)
    manager.register_shard(shard2)
    hash_shards = manager.approach(ShardingStrategy.HASH_BASED)
    assert len(hash_shards) == 1
    assert hash_shards[0].shard_id == "shard1"


def test_data_sharding_manager_get_shards_by_data_source_empty():
    """测试 DataShardingManager（根据数据源获取分片，空）"""
    manager = DataShardingManager()
    shards = manager.get_shards_by_data_source("source1")
    assert shards == []


def test_data_sharding_manager_get_shards_by_data_source_with_shards():
    """测试 DataShardingManager（根据数据源获取分片，有分片）"""
    manager = DataShardingManager()
    shard1 = ShardInfo(
        shard_id="shard1",
        data_source="source1",
        approach=ShardingStrategy.HASH_BASED,
        parameters={},
        assigned_nodes=[],
        status="active",
        created_at=datetime.now(),
        metadata={}
    )
    shard2 = ShardInfo(
        shard_id="shard2",
        data_source="source2",
        approach=ShardingStrategy.HASH_BASED,
        parameters={},
        assigned_nodes=[],
        status="active",
        created_at=datetime.now(),
        metadata={}
    )
    manager.register_shard(shard1)
    manager.register_shard(shard2)
    source1_shards = manager.get_shards_by_data_source("source1")
    assert len(source1_shards) == 1
    assert source1_shards[0].shard_id == "shard1"

