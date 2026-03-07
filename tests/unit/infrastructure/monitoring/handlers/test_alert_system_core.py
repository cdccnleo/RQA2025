import builtins
import importlib
import sys
import types


def _reload_alert_module():
    module_name = "src.infrastructure.monitoring.alert_system"
    sys.modules.pop(module_name, None)
    return importlib.import_module(module_name)




def test_alert_system_fallback_on_import_error(monkeypatch):
    original = sys.modules.pop("src.risk.alert_system", None)

    original_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "src.risk.alert_system":
            raise ImportError("forced failure")
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    try:
        module = _reload_alert_module()

        assert module.AlertSystem.__module__ == "src.infrastructure.monitoring.alert_system"
        assert module.AlertLevel.__module__ == "src.infrastructure.monitoring.alert_system"
        assert module.Alert.__module__ == "src.infrastructure.monitoring.alert_system"
    finally:
        if original is not None:
            sys.modules["src.risk.alert_system"] = original
