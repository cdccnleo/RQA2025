from types import SimpleNamespace

import pytest

from src.infrastructure.config.core.config_manager_operations import (
    UnifiedConfigManagerWithOperations,
)
from src.infrastructure.config.core.exceptions import (
    ConfigValidationError,
    ConfigKeyError,
    ConfigTypeError,
)


@pytest.fixture
def manager():
    mgr = UnifiedConfigManagerWithOperations({"default": {"existing": 1}})
    mgr._validator = SimpleNamespace(_validation_rules={})
    return mgr


def test_validation_rules_property_sync(manager):
    manager._validation_rules = {"rule": {"type": "string"}}
    assert manager._validator._validation_rules == {"rule": {"type": "string"}}

    del manager._validation_rules
    assert manager._validator._validation_rules == {}


def test_get_validation_rules_sources(manager):
    manager.config = {"validation_rules": {"cfg": {"type": "string"}}}
    assert manager._get_validation_rules() == {"cfg": {"type": "string"}}

    manager.config = {}
    manager._config_settings = {"validation_rules": {"settings": {"required": True}}}
    assert manager._get_validation_rules() == {"settings": {"required": True}}

    manager._config_settings = {}
    manager._data["validation_rules"] = {"data": {"type": "number"}}
    assert manager._get_validation_rules() == {"data": {"type": "number"}}


def test_validate_config_paths(manager):
    manager._data = {}
    with pytest.raises(ConfigValidationError):
        manager.validate_config()

    del manager._validation_rules
    assert manager.validate_config({"ok": 1}) is True

    rules = {"section": {"field": {"type": "string", "required": True}}}
    assert manager._validate_with_rules({"section": {"field": "value"}}, rules) is True
    assert manager._validate_with_rules({"section": {}}, rules) is False


def test_validate_config_key_and_basic(manager):
    with pytest.raises(ConfigTypeError):
        manager._validate_basic([])

    with pytest.raises(ConfigKeyError):
        manager._validate_config_key("")
    with pytest.raises(ConfigKeyError):
        manager._validate_config_key("<invalid>")
    with pytest.raises(ConfigKeyError):
        manager._validate_config_key(123)  # type: ignore[arg-type]


def test_rule_validators_raise(manager):
    with pytest.raises(ConfigValidationError):
        manager._validate_string_constraints("name", "sh", {"min_length": 3})
    with pytest.raises(ConfigValidationError):
        manager._validate_numeric_constraints("count", 1, {"min": 2})
    with pytest.raises(ConfigValidationError):
        manager._validate_enum_constraint("mode", "b", {"enum": ["a"]})
    with pytest.raises(ConfigValidationError):
        manager._validate_pattern_constraint("token", "abc", {"pattern": r"^xyz"})


def test_watchers_and_notifications(manager):
    events = []

    def callback(key, value):
        events.append((key, value))

    manager.watch("default.value", callback)
    assert manager.set("default.value", 10) is True
    assert events == [("default.value", 10)]

    manager.unwatch("default.value", callback)
    manager.set("default.value", 20)
    assert events == [("default.value", 10)]


def test_get_with_fallback_and_set_with_validation(manager):
    manager._data.setdefault("fallback", {})["val"] = "data"
    assert manager.get_with_fallback("primary.val", ["fallback.val"]) == "data"

    manager._validation_rules = {"env.port": {"type": "number", "min": 1000}}
    assert manager.set_with_validation("env.port", 1500) is True
    assert manager.set_with_validation("env.port", 100) is False


def test_batch_update_validation(manager):
    manager._validation_rules = {"env.port": {"type": "number", "min": 1000}}
    results = manager.batch_update({"env.port": 500, "env.host": "localhost"})
    assert results == {"env.port": False, "env.host": True}


def test_update_merge_and_failure(manager):
    manager._data = {"section": {"inner": {"value": 1}, "other": 2}}
    manager.update({"section": {"inner": {"extra": 3}}})
    assert manager._data["section"]["inner"]["extra"] == 3

    with pytest.raises(ValueError):
        manager.update(None)  # type: ignore[arg-type]


def test_parse_and_set_nested(manager):
    structure = manager._parse_key_structure("section.key.value")
    assert structure["is_nested"] is True
    assert structure["section"] == "section"

    manager._data["section"] = {"child": "leaf"}
    with pytest.raises(ValueError):
        manager._set_nested_config_value("section", ["child", "grand"], "value")


def test_get_config_snapshot_and_summary(manager):
    snapshot = manager.get_config_snapshot()
    assert "data" in snapshot and "metadata" in snapshot

    summary = manager.get_config_summary()
    assert summary["total_sections"] >= 1


def test_delete_and_clear_all(manager):
    manager.set("default.to_remove", 1)
    assert manager.delete("default", "to_remove") is True
    assert manager.delete("default", "to_remove") is False
    assert manager.clear_all() is True
    assert manager.get_sections() == []


def test_merge_config(manager):
    manager._data = {"section": {"value": 1}}
    assert manager.merge_config({"section": {"extra": 2}}) is True
    assert manager._data["section"]["extra"] == 2


