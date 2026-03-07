import pytest

from src.infrastructure.versioning.data.data_version_manager import (
    DataVersionManager,
    VersionInfo,
)
from src.infrastructure.versioning.core.version import Version


def test_save_and_get_version(tmp_path):
    manager = DataVersionManager(base_path=tmp_path / "data_versions")

    v1 = manager.save_version(
        "dataset",
        {"value": 1},
        metadata={"owner": "qa"},
        tags=["stable"],
    )
    assert isinstance(v1, Version)
    assert str(v1) == "1.0.0"

    retrieved = manager.get_version("dataset")
    assert retrieved == {"value": 1}

    info = manager.get_version_info("dataset", v1)
    assert isinstance(info, VersionInfo)
    assert info.metadata["owner"] == "qa"
    assert info.tags == ["stable"]

    v2 = manager.create_version(
        "dataset",
        {"value": 2},
        metadata={"owner": "qa"},
        tags=["candidate"],
    )
    assert str(v2) == "1.0.1"
    assert manager.get_version("dataset", v1) == {"value": 1}
    assert manager.get_version("dataset", v2) == {"value": 2}

    history = manager.get_version_history("dataset")
    assert len(history) == 2
    assert [str(item.version) for item in history] == ["1.0.0", "1.0.1"]


def test_list_versions_and_checksum(tmp_path):
    manager = DataVersionManager(base_path=tmp_path / "data_versions")

    manager.save_version("a", {"x": 1})
    manager.save_version("a", {"x": 2})
    manager.save_version("b", ["alpha", "beta"])

    assert manager.list_versions("a") == ["1.0.0", "1.0.1"]
    all_versions = manager.list_versions()
    assert set(all_versions) == {"1.0.0", "1.0.1", "1.0.0"}

    info = manager.get_version_info("b", "1.0.0")
    assert info is not None
    checksum_again = manager._calculate_checksum(info.data)
    assert checksum_again == info.checksum


def test_cleanup_old_versions(tmp_path, monkeypatch):
    manager = DataVersionManager(base_path=tmp_path / "data_versions")

    v1 = manager.save_version("dataset", {"payload": 1})
    v2 = manager.save_version("dataset", {"payload": 2})

    # 将第一个版本设置为旧时间戳
    info_v1 = manager.get_version_info("dataset", v1)
    info_v2 = manager.get_version_info("dataset", v2)
    assert info_v1 and info_v2

    old_timestamp = info_v1.timestamp.replace(year=info_v1.timestamp.year - 1)
    info_v1.timestamp = old_timestamp

    removed = manager.cleanup_old_versions(days=30)
    assert removed == 1

    remaining_versions = manager.list_versions("dataset")
    assert remaining_versions == ["1.0.1"]
    assert manager.get_version("dataset") == {"payload": 2}


def test_missing_versions_behaviour(tmp_path):
    manager = DataVersionManager(base_path=tmp_path / "data_versions")

    assert manager.get_version("missing") is None
    assert manager.get_version_info("missing", "1.0.0") is None
    assert manager.list_versions("missing") == []

    with pytest.raises(ValueError):
        manager.save_version("dataset", {"value": 1}, version="invalid.version")

