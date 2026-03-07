# -*- coding: utf-8 -*-
"""
分区管理器真实实现测试
测试 PartitionManager 的核心功能
"""

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
import pandas as pd
from datetime import datetime

from src.data.lake.partition_manager import (
    PartitionManager,
    PartitionConfig,
    PartitionStrategy
)


@pytest.fixture
def partition_manager():
    """创建分区管理器实例"""
    return PartitionManager()


@pytest.fixture
def sample_dataframe():
    """创建示例DataFrame"""
    return pd.DataFrame({
        'date': pd.date_range('2024-01-01', periods=10, freq='D'),
        'symbol': ['AAPL'] * 10,
        'value': [100 + i for i in range(10)]
    })


def test_partition_manager_initialization(partition_manager):
    """测试分区管理器初始化"""
    assert partition_manager.config is not None
    assert partition_manager.config.approach == PartitionStrategy.DATE


def test_get_partition_info_date_strategy(partition_manager, sample_dataframe):
    """测试日期分区策略"""
    config = PartitionConfig(
        approach=PartitionStrategy.DATE,
        partition_key='date'
    )
    partition_manager.config = config
    
    partition_info = partition_manager.get_partition_info(sample_dataframe)
    
    assert 'date' in partition_info
    assert partition_info['date'] is not None


def test_get_partition_info_hash_strategy(partition_manager, sample_dataframe):
    """测试哈希分区策略"""
    config = PartitionConfig(
        approach=PartitionStrategy.HASH,
        partition_key='symbol',
        num_partitions=10
    )
    partition_manager.config = config
    
    partition_info = partition_manager.get_partition_info(sample_dataframe)
    
    assert 'hash' in partition_info
    assert partition_info['hash'].startswith('part_')


def test_get_partition_info_custom_strategy(partition_manager, sample_dataframe):
    """测试自定义分区策略"""
    config = PartitionConfig(
        approach=PartitionStrategy.CUSTOM,
        partition_key='symbol'
    )
    partition_manager.config = config
    
    partition_info = partition_manager.get_partition_info(sample_dataframe)
    
    assert 'custom' in partition_info
    assert partition_info['custom'] == 'AAPL'


def test_get_partition_info_range_strategy(partition_manager, sample_dataframe):
    """测试范围分区策略"""
    config = PartitionConfig(
        approach=PartitionStrategy.RANGE,
        partition_key='value',
        range_bins=[105, 110, 115]
    )
    partition_manager.config = config
    
    partition_info = partition_manager.get_partition_info(sample_dataframe)
    
    assert 'range' in partition_info
    assert partition_info['range'].startswith('bin_')


def test_get_partition_info_missing_key(partition_manager, sample_dataframe):
    """测试缺少分区键的情况"""
    config = PartitionConfig(
        approach=PartitionStrategy.DATE,
        partition_key='nonexistent'
    )
    partition_manager.config = config
    
    partition_info = partition_manager.get_partition_info(sample_dataframe)
    
    assert partition_info == {}


def test_get_partition_path(partition_manager):
    """测试获取分区路径"""
    partition_info = {'date': '2024-01-01', 'symbol': 'AAPL'}
    
    path = partition_manager.get_partition_path(partition_info)
    
    assert 'date=2024-01-01' in path
    assert 'symbol=AAPL' in path


def test_get_partition_path_empty(partition_manager):
    """测试空分区信息"""
    path = partition_manager.get_partition_path({})
    
    assert path == ""


def test_optimize_partitions(partition_manager, sample_dataframe):
    """测试优化分区大小"""
    partitions = partition_manager.optimize_partitions(
        sample_dataframe,
        target_size_mb=1
    )
    
    assert len(partitions) > 0
    assert all(isinstance(p, pd.DataFrame) for p in partitions)


def test_optimize_partitions_empty_data(partition_manager):
    """测试优化空数据"""
    empty_df = pd.DataFrame()
    partitions = partition_manager.optimize_partitions(empty_df)
    
    assert partitions == []


def test_normalize_strategy(partition_manager):
    """测试策略规范化"""
    # 测试字符串策略
    strategy = partition_manager._normalize_strategy("date")
    assert strategy == PartitionStrategy.DATE
    
    # 测试枚举策略
    strategy = partition_manager._normalize_strategy(PartitionStrategy.HASH)
    assert strategy == PartitionStrategy.HASH
    
    # 测试无效策略（应返回默认值）
    strategy = partition_manager._normalize_strategy("invalid")
    assert strategy == PartitionStrategy.DATE


def test_list_partitions(partition_manager, tmp_path):
    """测试列出分区"""
    # 创建测试分区目录
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    (dataset_path / "date=2024-01-01").mkdir()
    (dataset_path / "date=2024-01-02").mkdir()
    
    partitions = partition_manager.list_partitions(str(dataset_path))
    
    assert len(partitions) == 2
    assert all('path' in p for p in partitions)


def test_list_partitions_nonexistent(partition_manager, tmp_path):
    """测试列出不存在数据集的分区"""
    nonexistent_path = tmp_path / "nonexistent"
    
    partitions = partition_manager.list_partitions(str(nonexistent_path))
    
    assert partitions == []


def test_get_partition_stats(partition_manager, tmp_path):
    """测试获取分区统计信息"""
    # 创建测试分区目录和文件
    dataset_path = tmp_path / "dataset"
    dataset_path.mkdir()
    partition_dir = dataset_path / "date=2024-01-01"
    partition_dir.mkdir()
    (partition_dir / "file1.parquet").write_text("test data")
    
    stats = partition_manager.get_partition_stats(str(dataset_path))
    
    assert isinstance(stats, dict)
    assert 'total_partitions' in stats
    assert 'total_files' in stats
    assert stats['total_partitions'] > 0


def test_get_partition_stats_nonexistent(partition_manager, tmp_path):
    """测试获取不存在数据集的分区统计"""
    nonexistent_path = tmp_path / "nonexistent"
    
    stats = partition_manager.get_partition_stats(str(nonexistent_path))
    
    assert isinstance(stats, dict)
    assert stats['total_partitions'] == 0


def test_parse_partition_path(partition_manager):
    """测试解析分区路径"""
    partition_info = partition_manager._parse_partition_path("date=2024-01-01")
    
    assert partition_info == {'date': '2024-01-01'}


def test_parse_partition_path_invalid(partition_manager):
    """测试解析无效分区路径"""
    partition_info = partition_manager._parse_partition_path("invalid_path")
    
    assert partition_info == {}

