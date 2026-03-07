import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.infrastructure.monitoring.core.component_registrar import ComponentRegistrar


class DummyComponent:
    """示例组件"""

    VERSION = "1.2.3"
    CAPABILITIES = ["metrics", "alerts"]
    DEPENDENCIES = ["database"]


def test_register_and_get_component():
    registrar = ComponentRegistrar()
    assert registrar.register_component("dummy", DummyComponent) is True
    assert registrar.is_registered("dummy") is True
    assert registrar.get_component("dummy") is DummyComponent

    metadata = registrar.get_component_metadata("dummy")
    assert metadata["component_type"] == "DummyComponent"
    assert metadata["version"] == "1.2.3"
    assert "metrics" in metadata["capabilities"]


def test_register_component_with_metadata_and_duplicate():
    registrar = ComponentRegistrar()
    metadata = {"component_type": "custom", "capabilities": ["x"]}
    assert registrar.register_component("custom", DummyComponent, metadata=metadata) is True
    assert registrar.get_component_metadata("custom")["component_type"] == "custom"

    # Duplicate registration should return False and not overwrite metadata
    assert registrar.register_component("custom", DummyComponent) is False
    assert registrar.get_registered_count() == 1


def test_unregister_component_and_clear_all():
    registrar = ComponentRegistrar()
    registrar.register_component("dummy", DummyComponent)
    assert registrar.unregister_component("dummy") is True
    assert registrar.unregister_component("missing") is False

    registrar.register_component("a", DummyComponent)
    registrar.register_component("b", DummyComponent)
    cleared = registrar.clear_all_registrations()
    assert cleared == 2
    assert registrar.get_registered_count() == 0


def test_find_components_by_filters():
    registrar = ComponentRegistrar()
    registrar.register_component("dummy", DummyComponent)

    assert registrar.find_components_by_capability("metrics") == ["dummy"]
    assert registrar.find_components_by_capability("missing") == []

    assert registrar.find_components_by_type("DummyComponent") == ["dummy"]
    assert registrar.find_components_by_type("Other") == []

    assert registrar.list_components()  # 触发元数据列表


def test_update_metadata_and_summary():
    registrar = ComponentRegistrar()
    registrar.register_component("dummy", DummyComponent)

    assert registrar.update_component_metadata("dummy", {"description": "updated"}) is True
    assert registrar.get_component_metadata("dummy")["description"] == "updated"

    assert registrar.update_component_metadata("missing", {"description": "x"}) is False

    summary = registrar.get_registration_summary()
    assert summary["total_registered"] == 1
    assert summary["by_type"]["DummyComponent"] == 1
    assert summary["by_capability"]["metrics"] == 1
    assert summary["component_names"] == ["dummy"]


def test_validate_registration_errors():
    registrar = ComponentRegistrar()
    registrar.register_component("dummy", DummyComponent)

    errors = registrar.validate_registration("", DummyComponent)
    assert "组件名称必须是非空字符串" in errors

    errors = registrar.validate_registration("dummy", DummyComponent)
    assert "组件 'dummy' 已被注册" in errors

    errors = registrar.validate_registration("new", SimpleNamespace())
    assert "必须提供有效的类对象" in errors


def test_health_status_with_issues(monkeypatch):
    registrar = ComponentRegistrar()

    # No components registered -> should report warning
    status = registrar.get_health_status()
    assert status["status"] == "warning"
    assert "没有已注册的组件" in status["issues"]

    registrar.register_component("dummy", DummyComponent)
    registrar.update_component_metadata(
        "dummy",
        {"capabilities": ["cap"] * 5},  # Force duplicate capability warning
    )

    status = registrar.get_health_status()
    assert status["status"] == "warning"
    assert "能力重复声明过多" in status["issues"][0]


def test_health_status_error_path(monkeypatch, caplog):
    registrar = ComponentRegistrar()
    monkeypatch.setattr(
        registrar,
        "get_registration_summary",
        MagicMock(side_effect=RuntimeError("boom")),
    )

    with caplog.at_level(logging.ERROR):
        status = registrar.get_health_status()

    assert status["status"] == "error"
    assert status["error"] == "boom"
    assert "获取健康状态失败" in caplog.text


def test_register_component_failure(monkeypatch, caplog):
    registrar = ComponentRegistrar()

    monkeypatch.setattr(registrar, "_components", None)

    with caplog.at_level(logging.ERROR):
        assert registrar.register_component("broken", DummyComponent) is False
    assert "注册组件 broken 失败" in caplog.text


def test_unregister_component_failure(monkeypatch, caplog):
    registrar = ComponentRegistrar()

    class FaultyDict(dict):
        def __contains__(self, item):
            raise RuntimeError("boom")

    monkeypatch.setattr(registrar, "_components", FaultyDict())

    with caplog.at_level(logging.ERROR):
        assert registrar.unregister_component("x") is False
    assert "注销组件 x 失败" in caplog.text


def test_update_metadata_failure(monkeypatch, caplog):
    registrar = ComponentRegistrar()
    registrar.register_component("dummy", DummyComponent)

    class FaultyEntry(dict):
        def update(self, *args, **kwargs):
            raise RuntimeError("boom")

    registrar._metadata["dummy"] = FaultyEntry()

    with caplog.at_level(logging.ERROR):
        assert registrar.update_component_metadata("dummy", {"description": "x"}) is False
    assert "更新组件 dummy 元数据失败" in caplog.text

