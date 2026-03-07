"""
边界测试：partition_manager.py
测试边界情况和异常场景
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
from pathlib import Path
import tempfile
import shutil
from src.data.lake.partition_manager import (
    PartitionStrategy,
    PartitionConfig,
    PartitionManager
)


def test_partition_strategy_enum():
    """测试 PartitionStrategy（枚举值）"""
    assert PartitionStrategy.DATE.value == "date"
    assert PartitionStrategy.HASH.value == "hash"
    assert PartitionStrategy.CUSTOM.value == "custom"
    assert PartitionStrategy.RANGE.value == "range"


def test_partition_config_init_default():
    """测试 PartitionConfig（初始化，默认值）"""
    config = PartitionConfig()
    
    assert config.approach == PartitionStrategy.DATE
    assert config.partition_key is None
    assert config.num_partitions == 100
    assert config.date_format == "%Y-%m-%d"
    assert config.range_bins is None


def test_partition_config_init_custom():
    """测试 PartitionConfig（初始化，自定义值）"""
    config = PartitionConfig(
        approach=PartitionStrategy.HASH,
        partition_key="user_id",
        num_partitions=50,
        date_format="%Y/%m/%d",
        range_bins=[10.0, 20.0, 30.0]
    )
    
    assert config.approach == PartitionStrategy.HASH
    assert config.partition_key == "user_id"
    assert config.num_partitions == 50
    assert config.date_format == "%Y/%m/%d"
    assert config.range_bins == [10.0, 20.0, 30.0]


def test_partition_manager_init_default():
    """测试 PartitionManager（初始化，默认配置）"""
    manager = PartitionManager()
    
    assert manager.config.approach == PartitionStrategy.DATE
    assert manager.config.partition_key is None


def test_partition_manager_init_custom():
    """测试 PartitionManager（初始化，自定义配置）"""
    config = PartitionConfig(
        approach=PartitionStrategy.HASH,
        partition_key="id"
    )
    manager = PartitionManager(config)
    
    assert manager.config.approach == PartitionStrategy.HASH
    assert manager.config.partition_key == "id"


def test_partition_manager_get_partition_info_no_key():
    """测试 PartitionManager（获取分区信息，无分区键）"""
    manager = PartitionManager()
    data = pd.DataFrame({"col1": [1, 2, 3]})
    
    result = manager.get_partition_info(data)
    
    assert result == {}


def test_partition_manager_get_partition_info_key_not_in_data():
    """测试 PartitionManager（获取分区信息，键不在数据中）"""
    config = PartitionConfig(partition_key="nonexistent")
    manager = PartitionManager(config)
    data = pd.DataFrame({"col1": [1, 2, 3]})
    
    result = manager.get_partition_info(data)
    
    assert result == {}


def test_partition_manager_get_partition_info_date():
    """测试 PartitionManager（获取分区信息，日期分区）"""
    config = PartitionConfig(
        approach=PartitionStrategy.DATE,
        partition_key="date"
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "date": [datetime(2023, 1, 1), datetime(2023, 1, 2)],
        "value": [1, 2]
    })
    
    result = manager.get_partition_info(data)
    
    assert "date" in result
    assert result["date"] == "2023-01-01"


def test_partition_manager_get_partition_info_date_string():
    """测试 PartitionManager（获取分区信息，日期分区，字符串格式）"""
    config = PartitionConfig(
        approach=PartitionStrategy.DATE,
        partition_key="date"
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "date": ["2023-01-01", "2023-01-02"],
        "value": [1, 2]
    })
    
    result = manager.get_partition_info(data)
    
    assert "date" in result


def test_partition_manager_get_partition_info_date_na():
    """测试 PartitionManager（获取分区信息，日期分区，NA值）"""
    config = PartitionConfig(
        approach=PartitionStrategy.DATE,
        partition_key="date"
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "date": [None, None],
        "value": [1, 2]
    })
    
    result = manager.get_partition_info(data)
    
    assert result == {}


def test_partition_manager_get_partition_info_hash():
    """测试 PartitionManager（获取分区信息，哈希分区）"""
    config = PartitionConfig(
        approach=PartitionStrategy.HASH,
        partition_key="id",
        num_partitions=10
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "id": [1, 2, 3],
        "value": [10, 20, 30]
    })
    
    result = manager.get_partition_info(data)
    
    assert "hash" in result
    assert result["hash"].startswith("part_")


def test_partition_manager_get_partition_info_hash_na():
    """测试 PartitionManager（获取分区信息，哈希分区，NA值）"""
    config = PartitionConfig(
        approach=PartitionStrategy.HASH,
        partition_key="id"
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "id": [None, None],
        "value": [1, 2]
    })
    
    result = manager.get_partition_info(data)
    
    assert result == {}


def test_partition_manager_get_partition_info_custom():
    """测试 PartitionManager（获取分区信息，自定义分区）"""
    config = PartitionConfig(
        approach=PartitionStrategy.CUSTOM,
        partition_key="category"
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "category": ["A", "B", "C"],
        "value": [1, 2, 3]
    })
    
    result = manager.get_partition_info(data)
    
    assert "custom" in result
    assert result["custom"] == "A"


def test_partition_manager_get_partition_info_custom_na():
    """测试 PartitionManager（获取分区信息，自定义分区，NA值）"""
    config = PartitionConfig(
        approach=PartitionStrategy.CUSTOM,
        partition_key="category"
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "category": [None, None],
        "value": [1, 2]
    })
    
    result = manager.get_partition_info(data)
    
    assert result == {}


def test_partition_manager_get_partition_info_range():
    """测试 PartitionManager（获取分区信息，范围分区）"""
    config = PartitionConfig(
        approach=PartitionStrategy.RANGE,
        partition_key="value",
        range_bins=[10.0, 20.0, 30.0]
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "value": [5.0, 15.0, 25.0],
        "other": [1, 2, 3]
    })
    
    result = manager.get_partition_info(data)
    
    assert "range" in result
    assert result["range"].startswith("bin_")


def test_partition_manager_get_partition_info_range_no_bins():
    """测试 PartitionManager（获取分区信息，范围分区，无bins）"""
    config = PartitionConfig(
        approach=PartitionStrategy.RANGE,
        partition_key="value"
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "value": [1, 2, 3],
        "other": [1, 2, 3]
    })
    
    result = manager.get_partition_info(data)
    
    assert result == {}


def test_partition_manager_get_partition_info_range_na():
    """测试 PartitionManager（获取分区信息，范围分区，NA值）"""
    config = PartitionConfig(
        approach=PartitionStrategy.RANGE,
        partition_key="value",
        range_bins=[10.0, 20.0]
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "value": [None, None],
        "other": [1, 2]
    })
    
    result = manager.get_partition_info(data)
    
    assert result == {}


def test_partition_manager_get_partition_info_range_out_of_bounds():
    """测试 PartitionManager（获取分区信息，范围分区，超出范围）"""
    config = PartitionConfig(
        approach=PartitionStrategy.RANGE,
        partition_key="value",
        range_bins=[10.0, 20.0]
    )
    manager = PartitionManager(config)
    data = pd.DataFrame({
        "value": [100.0],  # 超出所有范围
        "other": [1]
    })
    
    result = manager.get_partition_info(data)
    
    assert "range" in result
    # 应该归入最后一个分区
    assert result["range"] == "bin_002"


def test_partition_manager_normalize_strategy_enum():
    """测试 PartitionManager（规范化策略，枚举）"""
    manager = PartitionManager()
    
    result = manager._normalize_strategy(PartitionStrategy.HASH)
    
    assert result == PartitionStrategy.HASH


def test_partition_manager_normalize_strategy_string():
    """测试 PartitionManager（规范化策略，字符串）"""
    manager = PartitionManager()
    
    result = manager._normalize_strategy("hash")
    
    assert result == PartitionStrategy.HASH


def test_partition_manager_normalize_strategy_invalid():
    """测试 PartitionManager（规范化策略，无效值）"""
    manager = PartitionManager()
    
    result = manager._normalize_strategy("invalid_strategy")
    
    # 应该返回默认的 DATE 策略
    assert result == PartitionStrategy.DATE


def test_partition_manager_get_partition_path_empty():
    """测试 PartitionManager（获取分区路径，空）"""
    manager = PartitionManager()
    
    result = manager.get_partition_path({})
    
    assert result == ""


def test_partition_manager_get_partition_path_single():
    """测试 PartitionManager（获取分区路径，单个）"""
    manager = PartitionManager()
    
    result = manager.get_partition_path({"date": "2023-01-01"})
    
    assert result == "date=2023-01-01"


def test_partition_manager_get_partition_path_multiple():
    """测试 PartitionManager（获取分区路径，多个）"""
    manager = PartitionManager()
    
    result = manager.get_partition_path({
        "date": "2023-01-01",
        "region": "us"
    })
    
    assert "date=2023-01-01" in result
    assert "region=us" in result
    assert "/" in result


def test_partition_manager_list_partitions_not_exists():
    """测试 PartitionManager（列出分区，路径不存在）"""
    manager = PartitionManager()
    
    result = manager.list_partitions("/nonexistent/path")
    
    assert result == []


def test_partition_manager_list_partitions_empty():
    """测试 PartitionManager（列出分区，空目录）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PartitionManager()
        
        result = manager.list_partitions(tmpdir)
        
        assert result == []


def test_partition_manager_list_partitions_with_dirs():
    """测试 PartitionManager（列出分区，有目录）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PartitionManager()
        
        # 创建分区目录
        partition_dir = Path(tmpdir) / "date=2023-01-01"
        partition_dir.mkdir()
        (partition_dir / "file1.txt").write_text("test")
        
        result = manager.list_partitions(tmpdir)
        
        assert len(result) == 1
        assert result[0]["date"] == "2023-01-01"
        assert result[0]["file_count"] == 1


def test_partition_manager_parse_partition_path_valid():
    """测试 PartitionManager（解析分区路径，有效）"""
    manager = PartitionManager()
    
    result = manager._parse_partition_path("date=2023-01-01")
    
    assert result == {"date": "2023-01-01"}


def test_partition_manager_parse_partition_path_invalid():
    """测试 PartitionManager（解析分区路径，无效）"""
    manager = PartitionManager()
    
    result = manager._parse_partition_path("invalid_path")
    
    assert result == {}


def test_partition_manager_optimize_partitions_empty():
    """测试 PartitionManager（优化分区，空数据）"""
    manager = PartitionManager()
    data = pd.DataFrame()
    
    result = manager.optimize_partitions(data)
    
    assert result == []


def test_partition_manager_optimize_partitions_small():
    """测试 PartitionManager（优化分区，小数据）"""
    manager = PartitionManager()
    data = pd.DataFrame({"col1": [1, 2, 3], "col2": [4, 5, 6]})
    
    result = manager.optimize_partitions(data, target_size_mb=100)
    
    assert len(result) >= 1
    assert all(isinstance(df, pd.DataFrame) for df in result)


def test_partition_manager_optimize_partitions_large():
    """测试 PartitionManager（优化分区，大数据）"""
    manager = PartitionManager()
    # 创建较大的数据框
    data = pd.DataFrame({
        "col1": range(10000),
        "col2": range(10000, 20000)
    })
    
    result = manager.optimize_partitions(data, target_size_mb=1)
    
    assert len(result) >= 1
    assert all(isinstance(df, pd.DataFrame) for df in result)


def test_partition_manager_get_partition_stats_not_exists():
    """测试 PartitionManager（获取分区统计，路径不存在）"""
    manager = PartitionManager()
    
    result = manager.get_partition_stats("/nonexistent/path")
    
    assert result["total_partitions"] == 0
    assert result["total_files"] == 0
    assert result["total_size_mb"] == 0


def test_partition_manager_get_partition_stats_empty():
    """测试 PartitionManager（获取分区统计，空目录）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PartitionManager()
        
        result = manager.get_partition_stats(tmpdir)
        
        assert result["total_partitions"] == 0
        assert result["total_files"] == 0


def test_partition_manager_get_partition_stats_with_partitions():
    """测试 PartitionManager（获取分区统计，有分区）"""
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = PartitionManager()
        
        # 创建分区目录和文件
        partition1 = Path(tmpdir) / "date=2023-01-01"
        partition1.mkdir()
        (partition1 / "file1.txt").write_text("test content")
        
        partition2 = Path(tmpdir) / "date=2023-01-02"
        partition2.mkdir()
        (partition2 / "file2.txt").write_text("test content 2")
        
        result = manager.get_partition_stats(tmpdir)
        
        assert result["total_partitions"] == 2
        assert result["total_files"] == 2
        assert "largest_partition" in result
        assert "smallest_partition" in result
