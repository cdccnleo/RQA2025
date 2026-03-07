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


import os
import shutil
import sys
import tempfile
from pathlib import Path

import pandas as pd
import pytest

module = sys.modules.setdefault("src.infrastructure.utils.exceptions", type(sys)("exceptions"))
if not hasattr(module, "DataVersionError"):
    setattr(module, "DataVersionError", RuntimeError)

try:
    from src.data.models import DataModel  # type: ignore
except ImportError:
    class DataModel:
        def __init__(self, data=None, frequency="1d", metadata=None):
            self.data = data
            self.frequency = frequency
            self._user_metadata = dict(metadata or {})

        def get_metadata(self, user_only=False):
            return dict(self._user_metadata)


from src.data.version_control import DataVersionManager


@pytest.fixture(autouse=True)
def patch_parquet(monkeypatch):
    storage = {}

    def fake_to_parquet(self, path, *args, **kwargs):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        storage[path] = self.copy()
        path.touch()

    def fake_read_parquet(path, *args, **kwargs):
        path = Path(path)
        if path not in storage:
            raise FileNotFoundError(path)
        return storage[path].copy()

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fake_to_parquet)
    monkeypatch.setattr(pd, "read_parquet", fake_read_parquet)


@pytest.fixture()
def version_manager_and_data():
    temp_dir = tempfile.mkdtemp()
    version_dir = os.path.join(temp_dir, "versions")
    version_manager = DataVersionManager(version_dir)

    test_data1 = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "name": ["Alice", "Bob", "Charlie"],
            "age": [25, 30, 35],
        }
    )
    test_metadata1 = {"source": "test", "created_at": "2023-01-01"}
    test_model1 = DataModel(test_data1, "1d", test_metadata1)

    test_data2 = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "name": ["Alice", "Bob", "Charlie", "David"],
            "age": [25, 30, 35, 40],
            "gender": ["F", "M", "M", "M"],
        }
    )
    test_metadata2 = {"source": "test_updated", "created_at": "2023-01-02", "updated_by": "tester"}
    test_model2 = DataModel(test_data2, "1d", test_metadata2)

    yield (
        version_manager,
        version_dir,
        test_model1,
        test_data1,
        test_metadata1,
        test_model2,
        test_data2,
        test_metadata2,
    )
    shutil.rmtree(temp_dir, ignore_errors=True)


def test_create_version(version_manager_and_data):
    version_manager, version_dir, test_model1, _, _, _, _, _ = version_manager_and_data
    version = version_manager.create_version(
        test_model1,
        description="Test version",
        tags=["test"],
        creator="tester",
    )
    assert version is not None
    assert version_manager.current_version == version
    version_file = Path(version_dir) / f"{version}.parquet"
    assert version_file.exists()
    version_info = version_manager.get_version_info(version)
    assert version_info is not None
    assert version_info["description"] == "Test version"
    assert version_info["tags"] == ["test"]
    assert version_info["creator"] == "tester"


def test_get_version(version_manager_and_data):
    version_manager, _, test_model1, test_data1, test_metadata1, _, _, _ = version_manager_and_data
    version = version_manager.create_version(
        test_model1,
        description="Test version",
        tags=["test"],
        creator="tester",
    )
    model = version_manager.get_version(version)
    assert model is not None
    pd.testing.assert_frame_equal(model.data, test_data1)
    assert model.get_metadata(user_only=True) == test_metadata1


def test_list_versions(version_manager_and_data):
    version_manager, _, test_model1, _, _, test_model2, _, _ = version_manager_and_data
    version1 = version_manager.create_version(
        test_model1,
        description="Test version 1",
        tags=["test", "v1"],
        creator="tester1",
    )
    version2 = version_manager.create_version(
        test_model2,
        description="Test version 2",
        tags=["test", "v2"],
        creator="tester2",
    )
    versions = version_manager.list_versions()
    assert len(versions) == 2
    versions_v1 = version_manager.list_versions(tags=["v1"])
    assert len(versions_v1) == 1
    assert versions_v1[0]["version_id"] == version1
    versions_tester2 = version_manager.list_versions(creator="tester2")
    assert len(versions_tester2) == 1
    assert versions_tester2[0]["version_id"] == version2


def test_delete_version(version_manager_and_data):
    version_manager, version_dir, test_model1, _, _, test_model2, _, _ = version_manager_and_data
    version1 = version_manager.create_version(
        test_model1,
        description="Test version 1",
        tags=["test", "v1"],
        creator="tester1",
    )
    version2 = version_manager.create_version(
        test_model2,
        description="Test version 2",
        tags=["test", "v2"],
        creator="tester2",
    )
    result = version_manager.delete_version(version1)
    assert result is True
    assert version_manager.get_version(version1) is None
    version_file = Path(version_dir) / f"{version1}.parquet"
    assert not version_file.exists()
    assert version1 not in version_manager.metadata["versions"]
    with pytest.raises(module.DataVersionError):
        version_manager.delete_version(version2)


def test_rollback(version_manager_and_data):
    version_manager, _, test_model1, test_data1, test_metadata1, test_model2, _, _ = version_manager_and_data
    version1 = version_manager.create_version(
        test_model1,
        description="Test version 1",
        tags=["test", "v1"],
        creator="tester1",
    )
    version_manager.create_version(
        test_model2,
        description="Test version 2",
        tags=["test", "v2"],
        creator="tester2",
    )
    rollback_model = version_manager.rollback_to_version(version1)
    assert rollback_model is not None
    pd.testing.assert_frame_equal(rollback_model.data, test_data1)
    assert rollback_model.get_metadata(user_only=True) == test_metadata1
    new_version_id = version_manager.current_version
    version_info = version_manager.get_version_info(new_version_id)
    assert version_info is not None
    assert "rollback" in version_info["tags"]


def test_compare_versions(version_manager_and_data):
    version_manager, _, test_model1, _, _, test_model2, _, _ = version_manager_and_data
    version1 = version_manager.create_version(
        test_model1,
        description="Test version 1",
        tags=["test", "v1"],
        creator="tester1",
    )
    version2 = version_manager.create_version(
        test_model2,
        description="Test version 2",
        tags=["test", "v2"],
        creator="tester2",
    )
    diff = version_manager.compare_versions(version1, version2)
    assert "metadata_diff" in diff
    assert "added" in diff["metadata_diff"]
    assert "updated_by" in diff["metadata_diff"]["added"]
    assert "changed" in diff["metadata_diff"]
    assert "source" in diff["metadata_diff"]["changed"]
    assert "data_diff" in diff
    assert diff["data_diff"]["shape_diff"]["rows"] == 1
    assert diff["data_diff"]["shape_diff"]["columns"] == 1
    assert diff["data_diff"]["columns_diff"]["added"] == ["gender"]
    assert diff["data_diff"]["columns_diff"]["removed"] == []
    assert "value_diff" in diff["data_diff"]

