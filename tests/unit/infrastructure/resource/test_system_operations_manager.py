from unittest.mock import MagicMock

import pytest

from src.infrastructure.resource.monitoring.alerts.system_operations_manager import SystemOperationsManager


class DummyRegistry:
    def __init__(self, components):
        self._components = components

    def list_components(self):
        return list(self._components.keys())

    def create_component(self, name):
        factory = self._components.get(name)
        if callable(factory):
            return factory()
        return factory


def constant(value):
    return lambda value=value: value


def test_start_system_counts_success_and_errors():
    logger = MagicMock()
    success_component = MagicMock()
    failing_component = MagicMock()
    failing_component.start.side_effect = RuntimeError("boom")
    registry = DummyRegistry(
        {
            "success": constant(success_component),
            "failure": constant(failing_component),
            "no_start": constant(object()),
        }
    )

    manager = SystemOperationsManager(registry, logger)
    result = manager.start_system()

    assert result is True
    success_component.start.assert_called_once()
    failing_component.start.assert_called_once()
    logger.error.assert_called_once()
    assert "系统启动完成" in logger.info.call_args_list[-1][0][0]


def test_start_system_without_success_returns_false():
    logger = MagicMock()
    registry = DummyRegistry({"no_method": constant(object())})
    manager = SystemOperationsManager(registry, logger)

    assert manager.start_system() is False
    logger.error.assert_not_called()


def test_stop_system_logs_and_returns_true():
    logger = MagicMock()
    first = MagicMock()
    second = MagicMock()
    second.stop.side_effect = ValueError("fail")
    registry = DummyRegistry({"first": constant(first), "second": constant(second)})

    manager = SystemOperationsManager(registry, logger)
    assert manager.stop_system() is True
    first.stop.assert_called_once()
    second.stop.assert_called_once()
    logger.error.assert_called_once()


def test_get_system_status_includes_component_states():
    class Healthy:
        def is_healthy(self):
            return True

    registry = DummyRegistry(
        {
            "healthy": Healthy,
            "unknown": constant(object()),
            "missing": constant(None),
        }
    )

    manager = SystemOperationsManager(registry, MagicMock())
    status = manager.get_system_status()

    assert status["system"] == "monitoring_alert_system"
    assert status["components"]["healthy"] == "healthy"
    assert status["components"]["unknown"] == "unknown"
    assert status["components"]["missing"] == "not_loaded"


def test_get_system_health_report(monkeypatch):
    class Component:
        def __init__(self):
            self._calls = 0

        def is_healthy(self):
            return True

        def get_health_info(self):
            return {"extra": "info"}

    registry = DummyRegistry({"comp": Component})

    monkeypatch.setattr(
        "src.infrastructure.resource.monitoring.alerts.system_operations_manager.time.time",
        lambda: 123.456,
    )

    manager = SystemOperationsManager(registry, MagicMock())
    report = manager.get_system_health_report()

    assert report["timestamp"] == 123.456
    assert report["components"][0]["name"] == "comp"
    assert report["components"][0]["extra"] == "info"


def test_update_configuration_notifies_components():
    logger = MagicMock()
    reloader = MagicMock()
    failing = MagicMock()
    failing.reload_config.side_effect = RuntimeError("error")
    registry = DummyRegistry({"ok": constant(reloader), "bad": constant(failing), "no_reload": constant(object())})

    manager = SystemOperationsManager(registry, logger)
    result = manager.update_configuration({"key": "value"})

    assert result is True
    reloader.reload_config.assert_called_once_with({"key": "value"})
    failing.reload_config.assert_called_once()
    logger.error.assert_called_once()
