import pytest

from src.infrastructure.config.core import unified_manager_enhanced


def test_get_status_enhanced(monkeypatch):
    base_class = unified_manager_enhanced._UnifiedConfigManager

    def fake_get_status(self):
        return {}

    monkeypatch.setattr(base_class, "get_status", fake_get_status, raising=False)

    def fake_init(self, *a, **kw):
        self._data = {"default": {"a": 1}}
        self._initialized = True

    monkeypatch.setattr(base_class, "__init__", fake_init, raising=False)
    manager = unified_manager_enhanced.UnifiedConfigManager()

    status = manager.get_status()
    assert status["initialized"] == getattr(manager, "_initialized", False)
    assert status["sections_count"] >= 0
    assert status["total_keys"] >= 0
    assert status.get("status", "active") == "active"


def test_get_config_with_source_info_enhanced(monkeypatch):
    base_class = unified_manager_enhanced._UnifiedConfigManager

    def fake_get_config_with_source_info(self, key, default_section="default"):
        if key == "exists":
            return {"value": 42, "source": "file", "available": True, "type": "int"}
        return {"value": None}

    monkeypatch.setattr(base_class, "get_config_with_source_info", fake_get_config_with_source_info, raising=False)

    def fake_init(self, *a, **kw):
        self._data = {}
        self._initialized = True

    monkeypatch.setattr(base_class, "__init__", fake_init, raising=False)
    manager = unified_manager_enhanced.UnifiedConfigManager()

    info_existing = manager.get_config_with_source_info("exists")
    assert info_existing["source"] == "file"
    assert info_existing["available"] is True
    assert info_existing["type"] == "int"

    info_missing = manager.get_config_with_source_info("missing")
    assert info_missing["source"] == "memory"
    assert info_missing["available"] is False
    assert info_missing["type"] == "NoneType"


@pytest.mark.parametrize("value,expected_type", [(None, "NoneType"), ([1, 2], "list")])
def test_type_detection(monkeypatch, value, expected_type):
    base_class = unified_manager_enhanced._UnifiedConfigManager

    def fake_get_config_with_source_info(self, key, default_section="default"):
        return {"value": value}

    monkeypatch.setattr(base_class, "get_config_with_source_info", fake_get_config_with_source_info, raising=False)

    def fake_init(self, *a, **kw):
        self._data = {}
        self._initialized = True

    monkeypatch.setattr(base_class, "__init__", fake_init, raising=False)
    manager = unified_manager_enhanced.UnifiedConfigManager()

    info = manager.get_config_with_source_info("any")
    assert info["type"] == expected_type
    assert info["available"] is (value is not None)


