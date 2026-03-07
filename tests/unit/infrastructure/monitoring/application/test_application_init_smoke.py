import builtins
import importlib
import sys
import types


def _reload_application_module():
    module_name = "src.infrastructure.monitoring.application"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)


def test_application_exports_with_available_dependencies(monkeypatch):
    module_name = "src.infrastructure.monitoring.application"

    stubs = {
        "src.infrastructure.monitoring.application.application_monitor": {
            "ApplicationMonitor": type("ApplicationMonitor", (), {}),
        },
        "src.infrastructure.monitoring.application.logger_pool_monitor": {
            "LoggerPoolMonitor": type("LoggerPoolMonitor", (), {}),
            "get_logger_pool_monitor": lambda: "pool",
            "get_logger_pool_metrics": lambda: {"total": 1},
        },
        "src.infrastructure.monitoring.application.logger_pool_monitor_refactored": {
            "LoggerPoolMonitorRefactored": type("LoggerPoolMonitorRefactored", (), {}),
        },
        "src.infrastructure.monitoring.application.production_monitor": {
            "ProductionMonitor": type("ProductionMonitor", (), {}),
        },
    }

    for module_path, attributes in stubs.items():
        stub = types.ModuleType(module_path)
        for name, value in attributes.items():
            setattr(stub, name, value)
        monkeypatch.setitem(sys.modules, module_path, stub)

    module = _reload_application_module()

    assert module.ApplicationMonitor is stubs[
        "src.infrastructure.monitoring.application.application_monitor"
    ]["ApplicationMonitor"]
    assert module.LoggerPoolMonitor is stubs[
        "src.infrastructure.monitoring.application.logger_pool_monitor"
    ]["LoggerPoolMonitor"]
    assert module.get_logger_pool_monitor() == "pool"
    assert module.get_logger_pool_metrics() == {"total": 1}
    assert module.LoggerPoolMonitorRefactored is stubs[
        "src.infrastructure.monitoring.application.logger_pool_monitor_refactored"
    ]["LoggerPoolMonitorRefactored"]
    assert module.ProductionMonitor is stubs[
        "src.infrastructure.monitoring.application.production_monitor"
    ]["ProductionMonitor"]
    assert set(module.__all__) == {
        "ApplicationMonitor",
        "LoggerPoolMonitor",
        "LoggerPoolMonitorRefactored",
        "ProductionMonitor",
        "get_logger_pool_monitor",
        "get_logger_pool_metrics",
    }


def test_application_handles_import_errors(monkeypatch, capsys):
    module_name = "src.infrastructure.monitoring.application"
    for key in list(sys.modules):
        if key == module_name or key.startswith(f"{module_name}."):
            sys.modules.pop(key, None)

    class BlockImport:
        def find_spec(self, fullname, path, target=None):
            if fullname.startswith(f"{module_name}."):
                raise ModuleNotFoundError(f"blocked import {fullname}")
            return None

    blocker = BlockImport()
    sys.meta_path.insert(0, blocker)

    try:
        module = _reload_application_module()
    finally:
        sys.meta_path.remove(blocker)
    capsys.readouterr()

    assert module.ApplicationMonitor is None
    assert module.LoggerPoolMonitor is None
    assert module.LoggerPoolMonitorRefactored is None
    assert module.ProductionMonitor is None
    assert module.get_logger_pool_monitor is None
    assert module.get_logger_pool_metrics is None

