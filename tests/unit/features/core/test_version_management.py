import json
import time
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from src.features.core.version_management import FeatureVersionManager, FeatureVersion


@pytest.fixture(autouse=True)
def patch_parquet_and_datetime(monkeypatch):
    from src.features.core import version_management as vm

    monkeypatch.setattr(
        pd.DataFrame,
        "to_parquet",
        lambda self, path, index=True: self.to_pickle(path),
        raising=False,
    )
    monkeypatch.setattr(pd, "read_parquet", lambda path: pd.read_pickle(path), raising=False)

    real_datetime = vm.datetime

    class FixedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            real = real_datetime.now(tz)
            return cls(
                real.year,
                real.month,
                real.day,
                real.hour,
                real.minute,
                real.second,
                real.microsecond,
                real.tzinfo,
            )

        @classmethod
        def fromisoformat(cls, date_string):
            real = real_datetime.fromisoformat(date_string)
            return cls(
                real.year,
                real.month,
                real.day,
                real.hour,
                real.minute,
                real.second,
                real.microsecond,
                real.tzinfo,
            )

        def strftime(self, fmt):
            try:
                return super().strftime(fmt)
            except ValueError:
                return super().strftime("%Y%m%d_%H%M%S")

    monkeypatch.setattr(vm, "datetime", FixedDateTime, raising=False)


@pytest.fixture
def manager(tmp_path):
    return FeatureVersionManager(storage_dir=str(tmp_path))


def _make_features(seed: int = 0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.normal(size=5),
            "f2": rng.integers(0, 10, size=5),
        }
    )


def test_create_and_get_version(manager, tmp_path):
    features = _make_features()
    version_id = manager.create_version(features, description="baseline", creator="tester")
    assert version_id in manager.versions

    stored = manager.get_version(version_id)
    pd.testing.assert_frame_equal(stored, features)

    info_path = tmp_path / version_id / "version_info.json"
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    assert info["description"] == "baseline"


def test_list_versions_filters(manager):
    v1 = manager.create_version(_make_features(1), description="v1", creator="alice")
    v2 = manager.create_version(_make_features(2), description="v2", creator="bob")

    filtered = manager.list_versions(creator="alice")
    assert [v.version_id for v in filtered] == [v1]

    manager.delete_version(v1)
    deleted = manager.list_versions(status="deleted")
    assert deleted and deleted[0].version_id == v1


def test_compare_versions_detects_changes(manager):
    base = _make_features(3)
    v1 = manager.create_version(base, description="base")

    modified = base.copy()
    modified["f2"] = modified["f2"] + 1
    modified["f3"] = 1.0
    v2 = manager.create_version(modified, parent_version=v1, description="modified")

    result = manager.compare_versions(v1, v2)
    assert "f3" in result["only_in_v2"]
    assert "f2" in result["value_changes"]
    assert result["feature_count_diff"] == manager.versions[v2].feature_count - manager.versions[v1].feature_count


def test_rollback_creates_new_version(manager):
    features = _make_features(4)
    v1 = manager.create_version(features, description="first")
    v2 = manager.create_version(_make_features(5), description="second")

    rollback_id = manager.rollback_version(v1, description="fix regression")
    assert rollback_id in manager.versions
    lineage = manager.get_version_lineage(rollback_id)
    assert lineage["ancestors"][0]["version_id"] == v1


def test_delete_version_marks_status(manager):
    version_id = manager.create_version(_make_features(6))
    assert manager.delete_version(version_id) is True
    assert manager.versions[version_id].status == "deleted"
    assert manager.delete_version("unknown") is False


def test_get_version_lineage_contains_descendants(manager):
    base = manager.create_version(_make_features(7))
    child = manager.create_version(_make_features(8), parent_version=base)
    lineage = manager.get_version_lineage(base)
    assert lineage["descendants"][0]["version_id"] == child


def test_record_changes_tracks_added_removed_modified(manager):
    parent = manager.create_version(_make_features(9))
    new_features = _make_features(10)
    new_features["extra"] = 1.0
    new_version = manager.create_version(new_features, parent_version=parent)
    changes = manager.changes[new_version]
    types = {change.change_type for change in changes}
    assert {"added", "modified"} <= types


def test_get_version_stats(manager):
    old = manager.create_version(_make_features(11))
    manager.versions[old].timestamp -= timedelta(days=1)
    manager.create_version(_make_features(12))
    stats = manager.get_version_stats()
    assert stats["total_versions"] == 2
    assert stats["active_versions"] == 2
    assert stats["oldest_version"] <= stats["newest_version"]


def test_load_existing_index(tmp_path):
    mgr = FeatureVersionManager(storage_dir=str(tmp_path))
    version_id = mgr.create_version(_make_features(13))
    # 重新加载
    reloaded = FeatureVersionManager(storage_dir=str(tmp_path))
    assert version_id in reloaded.versions

