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
import sys
from pathlib import Path

import pandas as pd
import pytest

module = sys.modules.setdefault("src.infrastructure.utils.exceptions", type(sys)("exceptions"))
if not hasattr(module, "DataVersionError"):
    class _DataVersionError(RuntimeError):
        ...

    setattr(module, "DataVersionError", _DataVersionError)

from src.data.version_control.version_manager import (
    DataVersionManager,
    DataVersionError,
)


class DummyDataModel:
    def __init__(self, data: pd.DataFrame, frequency: str = "1d", metadata=None):
        self.data = data
        self.frequency = frequency
        self._user_metadata = dict(metadata or {})

    def get_metadata(self, user_only: bool = False):
        return dict(self._user_metadata)


def _build_model(seed: int = 0, extra_cols=None):
    base = {"id": [1, 2], "value": [seed + 10, seed + 20]}
    if extra_cols:
        base.update(extra_cols)
    df = pd.DataFrame(base)
    metadata = {"batch": seed, "owner": f"user{seed}"}
    return DummyDataModel(df, "1d", metadata)


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


@pytest.fixture
def version_manager(tmp_path):
    version_dir = tmp_path / "versions"
    return DataVersionManager(str(version_dir))


def test_load_metadata_recovers_from_corruption(tmp_path):
    version_dir = tmp_path / "versions"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "version_metadata.json").write_text("{invalid json")

    manager = DataVersionManager(str(version_dir))
    assert manager.metadata["versions"] == {}
    assert manager.metadata["branches"] == {"main": None}


def test_history_and_lineage_corruption(tmp_path):
    version_dir = tmp_path / "versions"
    version_dir.mkdir(parents=True, exist_ok=True)
    (version_dir / "version_history.json").write_text("{broken")
    (version_dir / "version_lineage.json").write_text("{broken")

    manager = DataVersionManager(str(version_dir))
    assert manager.history == []
    assert manager.lineage == {}


def test_get_version_returns_none_when_file_missing(version_manager):
    vid = version_manager.create_version(
        _build_model(seed=1),
        description="missing file scenario",
    )
    (version_manager.version_dir / f"{vid}.parquet").unlink()

    assert version_manager.get_version(vid) is None


def test_list_versions_filters_by_branch_and_limit(version_manager):
    version_manager.create_version(_build_model(seed=1), description="main")
    feature_version = version_manager.create_version(
        _build_model(seed=2, extra_cols={"feature": [1, 2]}),
        description="feature",
        branch="feature",
        creator="alice",
    )

    branch_versions = version_manager.list_versions(branch="feature")
    assert len(branch_versions) == 1
    assert branch_versions[0]["version_id"] == feature_version

    latest_only = version_manager.list_versions(limit=1)
    assert len(latest_only) == 1
    assert latest_only[0]["version_id"] == feature_version


def test_delete_version_updates_lineage(version_manager):
    v1 = version_manager.create_version(
        _build_model(seed=1),
        description="experiment base",
        branch="experiment",
    )
    v2 = version_manager.create_version(
        _build_model(seed=2, extra_cols={"extra": [5, 6]}),
        description="follow up",
    )

    version_manager.lineage = {
        "root": [v1],
        v1: [v2],
        v2: [],
    }
    version_manager.lineage_file.write_text(json.dumps(version_manager.lineage))

    assert version_manager.delete_version(v1) is True
    assert v1 not in version_manager.lineage
    assert "root" in version_manager.lineage
    assert v2 in version_manager.lineage["root"]
    assert version_manager.metadata["branches"]["experiment"] is None


def test_delete_current_version_raises(version_manager):
    current = version_manager.create_version(_build_model(seed=3), description="current")
    with pytest.raises(DataVersionError):
        version_manager.delete_version(current)


def test_get_lineage_returns_ancestors(version_manager):
    parent = version_manager.create_version(_build_model(seed=4), description="parent")
    child = version_manager.create_version(
        _build_model(seed=5, extra_cols={"flag": [0, 1]}),
        description="child",
    )
    version_manager.lineage = {parent: [child], child: []}

    lineage = version_manager.get_lineage(child)
    assert lineage["version_id"] == child
    assert lineage["ancestors"][0]["version_id"] == parent


def test_export_version_missing_file_returns_false(version_manager, tmp_path):
    assert version_manager.export_version("not_exists", tmp_path / "export.parquet") is False


def test_import_version_missing_file_returns_none(version_manager, tmp_path):
    assert version_manager.import_version(tmp_path / "missing.parquet") is None


def test_import_version_success_creates_new_entry(version_manager, tmp_path):
    data = pd.DataFrame({"id": [1], "value": [42]})
    import_path = tmp_path / "external.parquet"
    data.to_parquet(import_path)

    version_id = version_manager.import_version(import_path)
    assert version_id is not None
    assert "imported" in version_manager.metadata["versions"][version_id]["tags"]


def test_update_metadata_unknown_version_returns_false(version_manager):
    assert version_manager.update_metadata("unknown", {"owner": "qa"}) is False


def test_compare_versions_missing_target_raises(version_manager):
    existing = version_manager.create_version(_build_model(seed=6), description="base")
    with pytest.raises(DataVersionError):
        version_manager.compare_versions(existing, "missing")


def test_rollback_to_missing_version_returns_none(version_manager):
    assert version_manager.rollback_to_version("missing") is None

