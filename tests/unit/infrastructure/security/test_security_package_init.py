import importlib
import pytest


def test_security_lazy_import_user_role() -> None:
    security_module = importlib.import_module("src.infrastructure.security")
    user_role_cls = getattr(security_module, "UserRole")
    performance_monitor_cls = getattr(security_module, "PerformanceMonitor")

    assert user_role_cls.__name__ == "UserRole"
    assert hasattr(performance_monitor_cls, "record_operation")


def test_security_lazy_import_missing_attribute() -> None:
    security_module = importlib.import_module("src.infrastructure.security")
    with pytest.raises(AttributeError):
        getattr(security_module, "NonExistingSecurityThing")


def test_security_all_exports_contains_core_components() -> None:
    security_module = importlib.import_module("src.infrastructure.security")
    exported = set(security_module.__all__)
    assert {"SecurityConfigManager", "PolicyComponent", "AccessDecision"}.issubset(exported)
