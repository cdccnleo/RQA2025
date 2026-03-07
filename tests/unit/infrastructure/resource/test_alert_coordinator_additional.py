import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


class DummyAlert:
    def __init__(self, type, level, message, timestamp):
        self.type = type
        self.level = level
        self.message = message
        self.timestamp = timestamp
        self.id = f"alert-{int(timestamp)}"


class DummyPerformanceMetrics:
    def __init__(self, cpu_usage=0.0, memory_usage=0.0, disk_usage=0.0):
        self.cpu_usage = cpu_usage
        self.memory_usage = memory_usage
        self.disk_usage = disk_usage


def _create_stub_modules():
    alert_dataclasses = ModuleType("src.infrastructure.resource.monitoring.alert_dataclasses")
    alert_dataclasses.Alert = DummyAlert
    alert_dataclasses.PerformanceMetrics = DummyPerformanceMetrics

    alert_enums = ModuleType("src.infrastructure.resource.monitoring.alert_enums")
    alert_enums.AlertType = SimpleNamespace(PERFORMANCE=SimpleNamespace(value="performance"))
    alert_enums.AlertLevel = SimpleNamespace(
        WARNING=SimpleNamespace(value="warning"),
        CRITICAL=SimpleNamespace(value="critical"),
    )

    shared_interfaces = ModuleType("src.infrastructure.resource.monitoring.shared_interfaces")

    class DummyStandardLogger:
        def __init__(self, *args, **kwargs):
            pass

        def log_info(self, *args, **kwargs):
            pass

        def log_warning(self, *args, **kwargs):
            pass

        def log_error(self, *args, **kwargs):
            pass

    shared_interfaces.ILogger = object
    shared_interfaces.StandardLogger = DummyStandardLogger
    return alert_dataclasses, alert_enums, shared_interfaces


@pytest.fixture
def coordinator(monkeypatch):
    alert_dataclasses, alert_enums, shared_interfaces = _create_stub_modules()
    monkeypatch.setitem(sys.modules, "src.infrastructure.resource.monitoring.alert_dataclasses", alert_dataclasses)
    monkeypatch.setitem(sys.modules, "src.infrastructure.resource.monitoring.alert_enums", alert_enums)
    monkeypatch.setitem(sys.modules, "src.infrastructure.resource.monitoring.shared_interfaces", shared_interfaces)

    module = importlib.reload(importlib.import_module("src.infrastructure.resource.monitoring.alerts.alert_coordinator"))
    logger = MagicMock()
    alert_manager = MagicMock()
    coord = module.AlertCoordinator(alert_manager=alert_manager, logger=logger)
    return coord, logger, module


def test_check_alerts_generates_active_list(coordinator):
    coord, logger, module = coordinator
    metrics = module.PerformanceMetrics(cpu_usage=85.0, memory_usage=90.0, disk_usage=92.0)
    alerts = coord.check_alerts(metrics)
    assert len(alerts) == 3
    active = coord.get_active_alerts()
    assert len(active) == 3
    stats = coord.get_alert_statistics()
    assert stats["total_active"] == 3
    assert stats["by_level"]["warning"] == 2
    assert stats["by_level"]["critical"] == 1


def test_resolve_alert_removes_entry(coordinator):
    coord, logger, module = coordinator
    metrics = module.PerformanceMetrics(cpu_usage=81.0)
    coord.check_alerts(metrics)
    alert_key = next(iter(coord._active_alerts.keys()))
    assert coord.resolve_alert(alert_key) is True
    assert coord.get_active_alerts() == []
    assert coord.resolve_alert("missing") is False
    logger.log_warning.assert_called_with("未找到告警: missing")


def test_check_alerts_error_path(monkeypatch, coordinator):
    coord, logger, module = coordinator
    monkeypatch.setattr(module, "Alert", MagicMock(side_effect=RuntimeError("boom")))
    alerts = coord.check_alerts(module.PerformanceMetrics(cpu_usage=100))
    assert alerts == []
    logger.log_error.assert_called_once()
