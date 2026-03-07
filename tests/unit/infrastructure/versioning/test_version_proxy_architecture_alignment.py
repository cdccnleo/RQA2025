from pathlib import Path
import sys
import types
import time

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

from src.infrastructure.versioning.core.version import Version
from src.infrastructure.versioning.proxy.proxy import VersionProxy

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


def test_version_proxy_registration_and_cache():
    proxy = VersionProxy(max_cache_size=2, cache_ttl=60)
    proxy.register_version("service", "1.0.0")

    first_fetch = proxy.get_version("service")
    second_fetch = proxy.get_version("service")
    assert first_fetch == second_fetch == Version("1.0.0")

    proxy.update_version("service", "1.1.0")
    updated = proxy.get_version("service")
    assert updated == Version("1.1.0")


def test_version_proxy_cache_eviction_and_ttl(monkeypatch):
    proxy = VersionProxy(max_cache_size=1, cache_ttl=1)
    proxy.register_version("svc1", "1.0.0")
    proxy.register_version("svc2", "2.0.0")

    assert proxy.get_version("svc1") == Version("1.0.0")
    assert proxy.get_version("svc2") == Version("2.0.0")

    # svc1 应该被挤出缓存，再次获取时仍可得到数据
    assert proxy.get_version("svc1") == Version("1.0.0")

    # 模拟缓存过期
    time.sleep(1.1)
    assert proxy.get_version("svc2") == Version("2.0.0")


def test_version_proxy_history_lookup():
    proxy = VersionProxy()
    proxy.register_version("data", "1.0.0")
    proxy.update_version("data", "1.1.0")

    assert proxy.get_version("data", "1.0.0") == Version("1.0.0")
    assert proxy.get_version("data", "1.1.0") == Version("1.1.0")
    assert proxy.get_version("data", "2.0.0") is None


def test_version_proxy_execute_interface():
    proxy = VersionProxy()
    proxy.execute("register_version", name="feature", version="0.1.0")
    assert proxy.execute("get_version", name="feature") == Version("0.1.0")

    with pytest.raises(ValueError):
        proxy.execute("unsupported_action")

