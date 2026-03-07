import importlib
import sys
import types

import pytest

MODULE_PATH = "src.infrastructure.resource.monitoring_alert_system"


@pytest.fixture(autouse=True)
def cleanup_module_cache(monkeypatch):
    monkeypatch.delitem(sys.modules, MODULE_PATH, raising=False)
    yield
    monkeypatch.delitem(sys.modules, MODULE_PATH, raising=False)


def _ensure_stub_module(monkeypatch, name):
    module = types.ModuleType(name)
    module.__path__ = []
    monkeypatch.setitem(sys.modules, name, module)
    return module


def test_primary_module_import(monkeypatch):
    monkeypatch.delitem(sys.modules, "src.infrastructure.monitoring.alert_system", raising=False)
    infrastructure_pkg = importlib.import_module("src.infrastructure")

    monitoring_pkg = _ensure_stub_module(monkeypatch, "src.infrastructure.monitoring")
    monkeypatch.setattr(infrastructure_pkg, "monitoring", monitoring_pkg, raising=False)

    alert_system_mod = types.ModuleType("src.infrastructure.monitoring.alert_system")

    class PrimaryAlertSystem:  # pragma: no cover - test target
        pass

    alert_system_mod.AlertSystem = PrimaryAlertSystem
    monkeypatch.setitem(sys.modules, "src.infrastructure.monitoring.alert_system", alert_system_mod)

    imported = importlib.import_module(MODULE_PATH)
    assert imported.MonitoringAlertSystem is PrimaryAlertSystem


def test_fallback_to_risk_module(monkeypatch):
    infrastructure_pkg = importlib.import_module("src.infrastructure")
    for name in ["src.infrastructure.monitoring.alert_system", "src.infrastructure.monitoring"]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    monitoring_pkg = _ensure_stub_module(monkeypatch, "src.infrastructure.monitoring")
    monkeypatch.setattr(infrastructure_pkg, "monitoring", monitoring_pkg, raising=False)

    broken_module = types.ModuleType("src.infrastructure.monitoring.alert_system")

    def _broken_getattr(_name):  # pragma: no cover - helper
        raise AttributeError("missing")

    broken_module.__getattr__ = _broken_getattr
    monkeypatch.setitem(sys.modules, "src.infrastructure.monitoring.alert_system", broken_module)

    src_pkg = importlib.import_module("src")
    risk_pkg = _ensure_stub_module(monkeypatch, "src.risk")
    monkeypatch.setattr(src_pkg, "risk", risk_pkg, raising=False)

    alert_system_mod = types.ModuleType("src.risk.alert_system")

    class RiskAlertSystem:  # pragma: no cover - test target
        pass

    alert_system_mod.AlertSystem = RiskAlertSystem
    monkeypatch.setitem(sys.modules, "src.risk.alert_system", alert_system_mod)

    imported = importlib.import_module(MODULE_PATH)
    assert imported.MonitoringAlertSystem is RiskAlertSystem


def test_default_placeholder_when_all_missing(monkeypatch):
    infrastructure_pkg = importlib.import_module("src.infrastructure")
    src_pkg = importlib.import_module("src")

    for name in [
        "src.infrastructure.monitoring.alert_system",
        "src.infrastructure.monitoring",
        "src.risk.alert_system",
        "src.risk",
    ]:
        monkeypatch.delitem(sys.modules, name, raising=False)

    stub_monitoring = types.ModuleType("src.infrastructure.monitoring")
    stub_monitoring.__path__ = []
    monkeypatch.setitem(sys.modules, "src.infrastructure.monitoring", stub_monitoring)
    monkeypatch.setattr(infrastructure_pkg, "monitoring", stub_monitoring, raising=False)
    monkeypatch.setitem(sys.modules, "src.infrastructure.monitoring.alert_system", types.ModuleType("src.infrastructure.monitoring.alert_system"))

    stub_risk = types.ModuleType("src.risk")
    stub_risk.__path__ = []
    monkeypatch.setitem(sys.modules, "src.risk", stub_risk)
    monkeypatch.setattr(src_pkg, "risk", stub_risk, raising=False)
    monkeypatch.setitem(sys.modules, "src.risk.alert_system", types.ModuleType("src.risk.alert_system"))

    imported = importlib.import_module(MODULE_PATH)

    placeholder = imported.MonitoringAlertSystem
    assert placeholder.__module__ == MODULE_PATH
    assert placeholder.__name__ == "MonitoringAlertSystem"
    assert placeholder is not object
