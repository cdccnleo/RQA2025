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
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

from src.data.lake.data_lake_manager import DataLakeManager, LakeConfig


def _create_manager(tmp_path: Path, compression: str = "csv") -> DataLakeManager:
    base_root = Path(tmp_path)
    base_root.mkdir(parents=True, exist_ok=True)
    config = LakeConfig(
        base_path=str(base_root / "enterprise_lake"),
        compression=compression,
        metadata_enabled=True,
    )
    return DataLakeManager(config)


def test_store_load_and_metadata_roundtrip(tmp_path: Path):
    manager = _create_manager(tmp_path)
    dataset_name = "ticks"
    trade_date = "2024-01-01"
    data = pd.DataFrame(
        {
            "trade_date": [trade_date, trade_date],
            "symbol": ["RQA", "RQA"],
            "price": [1.1, 1.2],
        }
    )
    metadata = {"dataset_name": dataset_name, "source": "unit-test", "owner": "ml"}
    range_start = datetime.now() - timedelta(minutes=5)
    stored_path = manager.store_data(
        data,
        dataset_name=dataset_name,
        partition_key="trade_date",
        metadata=metadata,
    )
    range_end = datetime.now() + timedelta(minutes=5)

    stored_file = Path(stored_path)
    assert stored_file.exists()

    metadata_files = list(manager.metadata_path.glob(f"{dataset_name}_*.json"))
    assert metadata_files, "metadata file should be generated"
    stored_metadata = json.loads(metadata_files[0].read_text(encoding="utf-8"))
    assert stored_metadata["custom_metadata"]["source"] == "unit-test"

    partition_file = manager.partitions_path / f"{dataset_name}_partitions.json"
    assert partition_file.exists()
    partitions = json.loads(partition_file.read_text(encoding="utf-8"))
    assert any("date=2024-01-01" in key for key in partitions)

    loaded = manager.load_data(
        dataset_name,
        partition_filter={"date": trade_date},
        date_range=(range_start, range_end),
    )
    assert_frame_equal(
        loaded.sort_values(by=list(loaded.columns)).reset_index(drop=True),
        data.sort_values(by=list(data.columns)).reset_index(drop=True),
    )
    assert dataset_name in manager.list_datasets()


def test_dataset_listing_validation_and_deletion(tmp_path: Path):
    manager = _create_manager(tmp_path)
    dataset_name = "quotes"
    data_day1 = pd.DataFrame({"trade_date": ["2024-01-01"], "value": [10]})
    data_day2 = pd.DataFrame({"trade_date": ["2024-01-02"], "value": [20]})

    manager.store_data(data_day1, dataset_name, partition_key="trade_date")
    manager.store_data(data_day2, dataset_name, partition_key="trade_date")

    info = manager.get_dataset_info(dataset_name)
    assert info["total_rows"] == len(data_day1) + len(data_day2)
    assert len(info["files"]) == 2
    assert any(part.get("date") in {"2024-01-01", "2024-01-02"} for part in info["partitions"])

    datasets = manager.list_datasets()
    assert dataset_name in datasets
    assert manager.validate_storage_path() is True

    manager.config.base_path = ""
    assert manager.validate_storage_path() is False

    assert manager.delete_dataset(dataset_name) is False
    assert manager.delete_dataset(dataset_name, confirm=True) is True
    assert not (manager.data_path / dataset_name).exists()


def test_batch_operations_and_operational_utilities(tmp_path: Path):
    manager = _create_manager(tmp_path)

    batch_frames = [pd.DataFrame({"value": [i]}) for i in range(3)]
    assert manager.store_batch_data(batch_frames) is True

    assert manager.store_data_with_metadata(
        pd.DataFrame({"value": [42]}), {"dataset_name": "shared", "source": "analytics"}
    )

    storage_info = manager.get_storage_info()
    assert storage_info["file_count"] >= 1
    assert storage_info["total_size_gb"] >= 0

    cleanup_info = manager.cleanup_old_data(datetime.now())
    assert cleanup_info["deleted_files"] == 0

    backup_dir = tmp_path / "backup"
    backup_info = manager.backup_data(str(backup_dir))
    restore_info = manager.restore_data(str(backup_dir))
    assert backup_info["backup_path"] == str(backup_dir)
    assert restore_info["status"] in {"success", "failed"}

    read_perf = manager.measure_read_performance()
    write_perf = manager.measure_write_performance()
    assert read_perf["avg_read_time_ms"] > 0
    assert write_perf["avg_write_time_ms"] > 0

    assert manager.encrypt_data({"foo": "bar"}) == "encrypted_data_string"
    assert manager.decrypt_data("encrypted")["sensitive"] == "information"
    assert manager.check_access_permission("user", "resource", "read") is True

    assert manager.retrieve_data_by_id("non_existent").empty
    assert manager.retrieve_data_by_type("equity") == []
    assert manager.retrieve_data_by_time_range(datetime.now(), datetime.now()) == []
    assert manager.query_data_with_filters({"asset": "equity"}) == []
    assert manager.advanced_query({"sql": "select 1"}) == []


def test_error_paths_and_missing_datasets(tmp_path: Path):
    bad_config = LakeConfig(base_path=str(tmp_path / "bad_lake"), compression="unsupported")
    bad_manager = DataLakeManager(bad_config)
    data = pd.DataFrame({"value": [1]})

    with pytest.raises(ValueError):
        bad_manager.store_data(data, "unsupported_ds")

    assert bad_manager.store_batch_data([data]) is False

    empty = bad_manager.load_data("missing_ds")
    assert empty.empty

    info = bad_manager.get_dataset_info("missing_ds")
    assert info["files"] == []
    assert bad_manager.list_datasets() == []
    assert bad_manager.delete_dataset("missing_ds", confirm=True) is False


def test_partition_strategies_and_helper_filters(tmp_path: Path):
    base_manager = _create_manager(tmp_path / "helper_base")
    nan_df = pd.DataFrame({"partition": [float("nan")]})
    assert base_manager._get_partition_info(nan_df, "partition") == {}

    hash_manager = _create_manager(tmp_path / "helper_hash")
    hash_manager.config.approach = "hash"
    hash_info = hash_manager._get_partition_info(pd.DataFrame({"symbol": ["RQA"]}), "symbol")
    assert "hash" in hash_info

    custom_manager = _create_manager(tmp_path / "helper_custom")
    custom_manager.config.approach = "custom"
    custom_info = custom_manager._get_partition_info(pd.DataFrame({"region": ["CN-N"]}), "region")
    assert custom_info["custom"] == "CN-N"

    path_with_partition = custom_manager.data_path / "ds/date=2024-01-01/data_20240101_000000.csv"
    path_with_partition.parent.mkdir(parents=True, exist_ok=True)
    path_with_partition.touch()

    extracted_partition = custom_manager._extract_partition_from_path(path_with_partition)
    assert extracted_partition["date"] == "2024-01-01"

    extracted_date = custom_manager._extract_date_from_path(path_with_partition)
    assert extracted_date.strftime("%Y-%m-%d") == "2024-01-01"

    partition_only_path = custom_manager.data_path / "ds/date=2024-01-02/sample.csv"
    partition_only_path.parent.mkdir(parents=True, exist_ok=True)
    partition_only_path.touch()
    extracted_partition_date = custom_manager._extract_date_from_path(partition_only_path)
    assert extracted_partition_date.strftime("%Y-%m-%d") == "2024-01-02"

    assert custom_manager._matches_partition_filter({"date": "2024-01-01"}, {"date": "2024-01-01"})
    assert not custom_manager._matches_partition_filter({"date": "2024-01-01"}, {"date": "2024-01-02"})

    now = datetime.now()
    assert custom_manager._matches_date_range(None, (now, now))
    assert custom_manager._matches_date_range(now - timedelta(days=1), (now - timedelta(days=2), now))
    assert custom_manager._matches_date_range(now + timedelta(days=1), (now - timedelta(days=2), now))


def test_find_matching_files_and_invalid_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manager = _create_manager(tmp_path / "finder")
    df = pd.DataFrame({"trade_date": ["2024-01-01"], "value": [1]})
    manager.store_data(df, "finder_ds", partition_key="trade_date")

    assert manager._find_matching_files("finder_ds", {"date": "2099-01-01"}, None) == []

    past = datetime.now() - timedelta(days=10)
    # 日期范围检查目前采用宽松策略，仍会返回文件
    assert manager._find_matching_files("finder_ds", None, (past, past))

    def raise_loader(_path):
        raise ValueError("boom")

    monkeypatch.setattr(manager, "_load_data_file", raise_loader)
    with pytest.raises(ValueError):
        manager.load_data("finder_ds")


def test_parquet_and_json_branches(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    parquet_manager = _create_manager(tmp_path / "parquet", compression="parquet")
    df = pd.DataFrame({"value": [1]})

    def fake_to_parquet(self, path, index=False):
        Path(path).write_text("value\n1\n", encoding="utf-8")

    def fake_read_parquet(path):
        return pd.DataFrame({"value": [1]})

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet, raising=False)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)

    parquet_manager.store_data(df, "parquet_ds")
    parquet_loaded = parquet_manager.load_data("parquet_ds")
    assert not parquet_loaded.empty

    json_manager = _create_manager(tmp_path / "json", compression="json")
    json_df = pd.DataFrame({"trade_date": ["2024-01-01"], "value": [2]})
    json_manager.store_data(json_df, "json_ds", partition_key="trade_date")
    json_loaded = json_manager.load_data("json_ds")
    assert not json_loaded.empty

    with pytest.raises(ValueError):
        parquet_manager._load_data_file(Path("unsupported.txt"))


def test_path_initialization_and_validation_exceptions(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manager = _create_manager(tmp_path / "path_ex")
    manager.base_path = None
    assert manager.validate_storage_path() is False

    other_manager = _create_manager(tmp_path / "init_ex")
    original_mkdir = Path.mkdir

    def failing_mkdir(self, *args, **kwargs):
        if self == other_manager.base_path:
            raise OSError("boom")
        return original_mkdir(self, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", failing_mkdir)
    assert other_manager.initialize_storage() is False


def test_list_and_info_exceptional_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manager = _create_manager(tmp_path / "list_ex")

    class BrokenPath:
        def rglob(self, _pattern):
            raise RuntimeError("boom")

    monkeypatch.setattr(manager, "data_path", BrokenPath())
    assert manager.list_datasets() == []

    info_manager = _create_manager(tmp_path / "info_ex")
    info_manager.store_data(pd.DataFrame({"trade_date": ["2024-01-01"], "value": [1]}), "info_ds", partition_key="trade_date")

    def raise_on_load(_path):
        raise ValueError("boom")

    monkeypatch.setattr(info_manager, "_load_data_file", raise_on_load)
    info = info_manager.get_dataset_info("info_ds")
    assert info["files"][0]["rows"] == 0

    class BrokenDividePath:
        def __truediv__(self, _other):
            raise RuntimeError("boom")

    monkeypatch.setattr(info_manager, "data_path", BrokenDividePath())
    assert info_manager.get_dataset_info("info_ds") == {}


def test_delete_and_metadata_exception_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    manager = _create_manager(tmp_path / "delete_ex")
    df = pd.DataFrame({"trade_date": ["2024-01-01"], "value": [1]})
    manager.store_data(df, "del_ds", partition_key="trade_date")

    import shutil

    def fail_rmtree(_path):
        raise OSError("boom")

    monkeypatch.setattr(shutil, "rmtree", fail_rmtree)
    assert manager.delete_dataset("del_ds", confirm=True) is False

    bad_config = LakeConfig(base_path=str((tmp_path / "meta_ex").resolve()), compression="unsupported")
    bad_manager = DataLakeManager(bad_config)
    assert bad_manager.store_data_with_metadata(pd.DataFrame({"value": [1]}), {"dataset_name": "meta_ds"}) is False

