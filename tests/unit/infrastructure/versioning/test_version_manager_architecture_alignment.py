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

from src.infrastructure.versioning.manager.manager import VersionManager
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


def test_version_manager_basic_registration_and_history():
    manager = VersionManager()

    manager.create_version("1.0.0", name="core", description="initial")
    assert manager.get_version("core") == Version("1.0.0")
    assert manager.get_current_version_name() == "core"

    manager.update_version("core", "1.1.0")
    history = manager.get_version_history("core")
    assert history[-1] == Version("1.0.0")
    assert manager.get_version("core") == Version("1.1.0")

    assert manager.version_exists("core")
    assert manager.list_version_names() == ["core"]
    listing = manager.list_versions(as_dict=False)
    assert listing[0][0] == "core" and listing[0][1] == Version("1.1.0")


def test_version_manager_current_version_and_latest_detection():
    manager = VersionManager()
    manager.register_version("service", "0.9.0")
    manager.register_version("service", "1.0.0")
    manager.register_version("gateway", "2.5.1")

    assert manager.set_current_version("service", None)
    assert manager.get_current_version() == Version("1.0.0")

    assert manager.find_latest_version() == Version("2.5.1")
    assert manager.validate_version_compatibility("service", "service")
    assert not manager.validate_version_compatibility("service", "gateway")


def test_version_manager_import_export_roundtrip():
    manager = VersionManager()
    manager.register_version("alpha", "1.0.0")
    manager.update_version("alpha", "1.1.0")
    manager.register_version("beta", "0.5.0")
    manager.set_current_version("beta", None)

    exported = manager.export_to_dict()
    assert exported["versions"]["alpha"] == "1.1.0"
    assert exported["current_version"] == "beta"
    assert "alpha" in exported["version_history"]

    restored = VersionManager()
    restored.import_from_dict(exported)
    assert restored.get_version("alpha") == Version("1.1.0")
    assert restored.get_current_version_name() == "beta"
    assert restored.get_version_history("alpha")[0] == Version("1.0.0")

    simple_import = VersionManager()
    simple_import.import_from_dict({"gamma": "3.0.0", "delta": "4.0.0"})
    assert simple_import.get_version("gamma") == Version("3.0.0")
    assert simple_import.list_version_names() == ["gamma", "delta"]


def test_version_manager_removal_and_clear():
    manager = VersionManager()
    manager.register_version("legacy", "0.1.0")
    manager.register_version("active", "2.0.0")

    assert manager.remove_version("legacy")
    assert not manager.version_exists("legacy")

    manager.clear_versions()
    assert manager.list_version_names() == []
    assert manager.get_current_version() is None

