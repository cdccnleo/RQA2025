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


import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.lake.data_lake_manager import DataLakeManager, LakeConfig
from src.data.lake.metadata_manager import MetadataManager
from src.data.lake.partition_manager import PartitionConfig, PartitionManager, PartitionStrategy


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "trade_date": ["2025-01-01", "2025-01-01"],
            "symbol": ["RQA", "RQA"],
            "price": [10.5, 11.0],
        }
    )


def test_data_lake_store_and_load_with_date_partition(tmp_path, sample_dataframe):
    base_path = tmp_path / "lake"
    config = LakeConfig(
        base_path=str(base_path),
        compression="csv",
        metadata_enabled=True,
        approach="date",
    )
    manager = DataLakeManager(config)

    metadata_payload = {
        "dataset_name": "market_quotes",
        "source": "unit-test",
        "quality_score": 0.99,
    }

    stored_path = manager.store_data(
        sample_dataframe,
        "market_quotes",
        partition_key="trade_date",
        metadata=metadata_payload,
    )

    assert Path(stored_path).exists()
    assert "date=2025-01-01" in stored_path

    metadata_files = list((base_path / "metadata").glob("market_quotes_data_*.json"))
    assert metadata_files, "元数据文件应被创建以符合架构要求"

    partition_file = base_path / "partitions" / "market_quotes_partitions.json"
    partition_data = json.loads(partition_file.read_text(encoding="utf-8"))
    assert "date=2025-01-01" in partition_data
    assert stored_path in partition_data["date=2025-01-01"]

    loaded = manager.load_data("market_quotes")
    assert len(loaded) == len(sample_dataframe)
    assert loaded["price"].tolist() == pytest.approx(sample_dataframe["price"].tolist())

    datasets = manager.list_datasets()
    assert datasets == ["market_quotes"]

    info = manager.get_dataset_info("market_quotes")
    assert info["total_rows"] == len(sample_dataframe)
    assert any(partition.get("date") == "2025-01-01" for partition in info["partitions"])


def test_data_lake_partition_filter(tmp_path):
    base_path = tmp_path / "lake"
    config = LakeConfig(
        base_path=str(base_path),
        compression="csv",
        metadata_enabled=False,
        approach="date",
    )
    manager = DataLakeManager(config)

    df_first = pd.DataFrame({"trade_date": ["2025-01-01"], "value": [1]})
    df_second = pd.DataFrame({"trade_date": ["2025-01-02"], "value": [2]})

    manager.store_data(df_first, "daily_metrics", partition_key="trade_date")
    manager.store_data(df_second, "daily_metrics", partition_key="trade_date")

    filtered = manager.load_data(
        "daily_metrics",
        partition_filter={"date": "2025-01-02"},
    )
    assert len(filtered) == 1
    assert filtered.iloc[0]["value"] == 2


def test_metadata_manager_lifecycle(tmp_path):
    metadata_path = tmp_path / "metadata"
    manager = MetadataManager(metadata_path=str(metadata_path))

    raw_data = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "amount": [100.0, 200.5, 150.25],
            "status": ["open", "closed", "open"],
        }
    )

    schema = manager.create_schema(
        "orders",
        raw_data,
        description="订单数据集",
        tags=["orders", "finance"],
        owner="data-team",
        access_level="restricted",
    )
    assert schema.dataset_name == "orders"
    assert metadata_path.joinpath("orders_schema.json").exists()

    loaded_schema = manager.get_schema("orders")
    assert loaded_schema is not None
    assert loaded_schema.description == "订单数据集"
    assert loaded_schema.tags == ["orders", "finance"]

    assert "orders" in manager.list_schemas()

    updated = manager.update_schema("orders", description="订单数据集（更新）")
    assert updated
    assert manager.get_schema("orders").description == "订单数据集（更新）"

    validation_missing = manager.validate_schema("orders", pd.DataFrame({"id": [1]}))
    assert validation_missing["valid"] is False
    assert any("Missing columns" in err for err in validation_missing["errors"])

    validation_extra = manager.validate_schema(
        "orders",
        pd.DataFrame({"id": [1], "amount": [99.0], "status": ["open"], "extra": ["x"]}),
    )
    assert validation_extra["valid"] is True
    assert any("Extra columns" in warn for warn in validation_extra["warnings"])

    stats = manager.get_schema_stats()
    assert stats["total_datasets"] == 1
    assert stats["total_columns"] == 3
    assert stats["access_levels"]["restricted"] == 1

    matches = manager.search_schemas("orders")
    assert matches and matches[0].dataset_name == "orders"

    assert manager.delete_schema("orders") is True
    assert manager.get_schema("orders") is None


def test_partition_manager_strategies(tmp_path):
    data = pd.DataFrame(
        {
            "trade_date": ["2025-01-01", "2025-01-02"],
            "symbol": ["RQA", "RQA"],
            "price": [10.0, 12.0],
        }
    )

    date_manager = PartitionManager(
        PartitionConfig(
            approach=PartitionStrategy.DATE,
            partition_key="trade_date",
            date_format="%Y-%m-%d",
        )
    )
    assert date_manager.get_partition_info(data) == {"date": "2025-01-01"}

    hash_manager = PartitionManager(
        PartitionConfig(
            approach=PartitionStrategy.HASH,
            partition_key="symbol",
            num_partitions=10,
        )
    )
    hash_info = hash_manager.get_partition_info(data)
    assert "hash" in hash_info and hash_info["hash"].startswith("part_")

    custom_manager = PartitionManager(
        PartitionConfig(
            approach=PartitionStrategy.CUSTOM,
            partition_key="symbol",
        )
    )
    assert custom_manager.get_partition_info(data) == {"custom": "RQA"}

    range_manager = PartitionManager(
        PartitionConfig(
            approach=PartitionStrategy.RANGE,
            partition_key="price",
            range_bins=[10.0, 11.0, 12.0],
        )
    )
    assert range_manager.get_partition_info(data)["range"] == "bin_000"

    partition_path = date_manager.get_partition_path({"date": "2025-01-01"})
    assert partition_path == "date=2025-01-01"

    dataset_dir = tmp_path / "dataset"
    first_partition = dataset_dir / "date=2025-01-01"
    first_partition.mkdir(parents=True)
    first_file = first_partition / "data.csv"
    first_file.write_text("col\nvalue\n", encoding="utf-8")

    second_partition = dataset_dir / "date=2025-01-02"
    second_partition.mkdir()
    (second_partition / "data.csv").write_text("col\nvalue\n", encoding="utf-8")

    partitions = date_manager.list_partitions(str(dataset_dir))
    assert len(partitions) == 2

    stats = date_manager.get_partition_stats(str(dataset_dir))
    assert stats["total_partitions"] == 2
    assert stats["total_files"] == 2
    assert stats["partition_distribution"]["date"] == 2

    optimized = date_manager.optimize_partitions(data, target_size_mb=0.000001)
    assert optimized, "分区优化应至少返回一个分片"

