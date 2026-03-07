from datetime import datetime
from pathlib import Path
import sys
import types

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_VERSIONING_PATH = _PROJECT_ROOT / "src" / "infrastructure" / "versioning"

_ORIGINAL_VERSIONING = sys.modules.get("src.infrastructure.versioning")
_ORIGINAL_VERSIONING_DATA = sys.modules.get("src.infrastructure.versioning.data")
_ORIGINAL_VERSIONING_DATA_MANAGER = sys.modules.get(
    "src.infrastructure.versioning.data.data_version_manager"
)

if "src.infrastructure.versioning" not in sys.modules:
    versioning_stub = types.ModuleType("src.infrastructure.versioning")
    versioning_stub.__path__ = [str(_VERSIONING_PATH)]
    sys.modules["src.infrastructure.versioning"] = versioning_stub

if "src.infrastructure.versioning.data" not in sys.modules:
    data_stub = types.ModuleType("src.infrastructure.versioning.data")
    sys.modules["src.infrastructure.versioning.data"] = data_stub

if "src.infrastructure.versioning.data.data_version_manager" not in sys.modules:
    data_manager_stub = types.ModuleType(
        "src.infrastructure.versioning.data.data_version_manager"
    )
    sys.modules["src.infrastructure.versioning.data.data_version_manager"] = data_manager_stub

from src.infrastructure.versioning.config.config_version_manager import (
    ConfigCleanupRequest,
    ConfigVersionComparisonRequest,
    ConfigVersionCreateRequest,
    ConfigVersionInfo,
    ConfigVersionManager,
    ConfigVersionStorage,
    ConfigVersionStorageConfig,
    ConfigVersionValidator,
    ConfigVersionComparator,
)
from src.infrastructure.versioning.core.version import Version

if _ORIGINAL_VERSIONING is None:
    sys.modules.pop("src.infrastructure.versioning", None)
else:
    sys.modules["src.infrastructure.versioning"] = _ORIGINAL_VERSIONING

if _ORIGINAL_VERSIONING_DATA is None:
    sys.modules.pop("src.infrastructure.versioning.data", None)
else:
    sys.modules["src.infrastructure.versioning.data"] = _ORIGINAL_VERSIONING_DATA

if _ORIGINAL_VERSIONING_DATA_MANAGER is None:
    sys.modules.pop("src.infrastructure.versioning.data.data_version_manager", None)
else:
    sys.modules[
        "src.infrastructure.versioning.data.data_version_manager"
    ] = _ORIGINAL_VERSIONING_DATA_MANAGER


def test_config_version_manager_creation_and_comparison(tmp_path):
    """按照基础设施架构文档验证配置版本管理流程"""
    manager = ConfigVersionManager(config_dir=tmp_path / "configs")

    invalid_request = ConfigVersionCreateRequest(config_name="service", config_data={})
    with pytest.raises(ValueError):
        manager.create_version_new(invalid_request)

    base_request = ConfigVersionCreateRequest(
        config_name="service",
        config_data={"threshold": 1, "enabled": True},
        creator="qa",
        description="baseline",
        tags=["stable"],
    )
    v1 = manager.create_version_new(base_request)
    assert str(v1) == "0.0.1"
    stored = manager.get_config("service")
    assert stored == {"threshold": 1, "enabled": True}

    info = manager.get_version_info("service", v1)
    assert info is not None
    assert info.creator == "qa"
    assert info.tags == ["stable"]
    assert len(info.config_hash) == 64

    explicit_request = ConfigVersionCreateRequest(
        config_name="service",
        config_data={"threshold": 2, "enabled": False},
        description="release",
        tags=["release"],
        auto_increment=False,
        explicit_version="1.0.0",
    )
    v2 = manager.create_version_new(explicit_request)
    assert str(v2) == "1.0.0"
    assert manager.get_config_version("service") == "1.0.0"
    assert manager.list_versions("service")[-1] == Version("1.0.0")

    comparison = manager.compare_versions_new(
        ConfigVersionComparisonRequest(
            config_name="service",
            version1="0.0.1",
            version2="1.0.0",
            comparison_type="full",
            include_values=True,
        )
    )
    assert comparison["summary"]["modified_count"] == 2
    assert "config1" in comparison["values"]
    assert comparison["values"]["config2"]["threshold"] == 2


def test_config_version_manager_cleanup_workflow(tmp_path):
    manager = ConfigVersionManager(config_dir=tmp_path / "configs_cleanup")

    for idx in range(3):
        manager.create_version_new(
            ConfigVersionCreateRequest(
                config_name="service",
                config_data={"value": idx},
                description=f"revision-{idx}",
            )
        )

    assert len(manager.list_versions("service")) == 3

    dry_run_request = ConfigCleanupRequest(config_name="service", keep_count=1, dry_run=True)
    assert manager.cleanup_versions(dry_run_request) == 2
    assert len(manager.list_versions("service")) == 3

    removed = manager.cleanup_versions(ConfigCleanupRequest(config_name="service", keep_count=1))
    assert removed == 2
    remaining_versions = manager.list_versions("service")
    assert len(remaining_versions) == 1
    latest = remaining_versions[0]
    assert manager.storage.load_config_data("service", Version("0.0.1")) is None
    assert manager.storage.history_file.exists()
    assert manager.cleanup_versions(ConfigCleanupRequest(config_name="missing")) == 0


def test_config_version_storage_history_roundtrip(tmp_path):
    storage = ConfigVersionStorage(
        ConfigVersionStorageConfig(base_dir=tmp_path / "config_store", history_file="history.json")
    )
    info = ConfigVersionInfo(
        version=Version("2.1.0"),
        timestamp=datetime.now(),
        creator="ops",
        description="deploy",
        config_hash="deadbeef",
        config_size=42,
        changes_summary={"added": ["feature"], "removed": [], "modified": []},
        parent_version=Version("2.0.5"),
        tags=["blue", "green"],
    )
    versions = {"service": [info]}
    current_versions = {"service": Version("2.1.0")}

    assert storage.save_history(versions, current_versions)

    loaded_versions, loaded_current = storage.load_history()
    assert "service" in loaded_versions
    loaded_info = loaded_versions["service"][0]
    assert loaded_info.description == "deploy"
    assert loaded_info.tags == ["blue", "green"]
    assert loaded_current["service"] == Version("2.1.0")


def test_config_version_comparator_and_validator_behaviour():
    comparator = ConfigVersionComparator()
    config_a = {"a": 1, "b": 2}
    config_b = {"a": 1, "b": 3, "c": 4}

    diff_only = comparator.compare_versions(
        ConfigVersionComparisonRequest(
            config_name="service",
            version1="0.0.1",
            version2="0.0.2",
            comparison_type="diff_only",
        ),
        config_a,
        config_b,
    )
    assert set(diff_only["added"]) == {"c"}
    assert diff_only["removed"] == []
    assert set(diff_only["modified"]) == {"b"}

    summary = comparator.compare_versions(
        ConfigVersionComparisonRequest(
            config_name="service",
            version1="0.0.1",
            version2="0.0.2",
            comparison_type="summary",
        ),
        config_a,
        config_b,
    )
    assert summary["total_changes"] == 2
    assert summary["modified_count"] == 1

    full = comparator.compare_versions(
        ConfigVersionComparisonRequest(
            config_name="service",
            version1="0.0.1",
            version2="0.0.2",
            comparison_type="full",
            include_values=True,
        ),
        config_a,
        config_b,
    )
    assert full["summary"]["added_count"] == 1
    assert full["values"]["config2"]["c"] == 4

    assert comparator.calculate_changes(None, config_b)["type"] == "initial"
    assert set(comparator.calculate_changes(config_a, config_b)["modified"]) == {"b"}

    validator = ConfigVersionValidator()
    schema = {"required": ["a", "b"], "types": {"a": int, "b": int}}

    errors = validator.validate_config({"a": "invalid"}, schema)
    assert any("缺少必需字段: b" in err for err in errors)
    assert any("字段 a 类型错误" in err for err in errors)

    assert validator.validate({"a": 1, "b": 2}, schema)
    type_errors = validator.validate_config(["not-dict"])
    assert "配置必须是字典类型" in type_errors[0]

