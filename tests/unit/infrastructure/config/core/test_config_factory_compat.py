import types

import pytest

from src.infrastructure.config.core import config_factory_compat as compat


@pytest.fixture(autouse=True)
def reset_factory_state():
    compat.ConfigFactory._managers.clear()
    compat.ConfigFactory.providers.clear()
    compat._default_manager = None
    compat._factory_instance = None
    yield
    compat.ConfigFactory._managers.clear()
    compat.ConfigFactory.providers.clear()
    compat._default_manager = None
    compat._factory_instance = None


def test_create_config_manager_success(monkeypatch):
    class DummyFactory:
        def __init__(self):
            self.created = []
            self.manager = {"name": None, "config": None}

        def create_config_manager(self, name, **kwargs):
            self.created.append((name, kwargs))
            self.manager = {"name": name, "config": kwargs.get("config")}
            return self.manager

    dummy_factory = DummyFactory()
    monkeypatch.setattr(compat, "get_config_factory", lambda: dummy_factory)

    config_payload = {"foo": "bar"}
    manager = compat.ConfigFactory.create_config_manager("test", config_payload)

    assert manager == {"name": "test", "config": config_payload}
    assert compat.ConfigFactory._managers["test"] is manager
    assert dummy_factory.created == [("test", {"config": config_payload})]


def test_create_config_manager_fallback(monkeypatch):
    def failing_factory():
        raise RuntimeError("factory unavailable")

    captured_config = {}

    class DummyManager:
        def __init__(self, config):
            captured_config.update(config)
            self.config = config

    monkeypatch.setattr(compat, "get_config_factory", failing_factory)
    monkeypatch.setattr(compat, "UnifiedConfigManager", DummyManager)

    manager = compat.ConfigFactory.create_config_manager("legacy", {"auto_reload": False})

    assert isinstance(manager, DummyManager)
    assert captured_config["auto_reload"] is False
    assert captured_config["validation_enabled"] is True  # default maintained
    assert compat.ConfigFactory._managers["legacy"] is manager


def test_destroy_config_manager_cleans_caches(monkeypatch):
    destroyed_keys = []

    class DummyFactory:
        def __init__(self):
            self._managers = {"unified|name=test": object()}

        def get_all_managers(self):
            return self._managers

        def destroy_manager(self, key):
            destroyed_keys.append(key)
            return True

    compat.ConfigFactory._managers["test"] = object()
    monkeypatch.setattr(compat, "get_config_factory", lambda: DummyFactory())

    result = compat.ConfigFactory.destroy_config_manager("test")

    assert result is True
    assert "test" not in compat.ConfigFactory._managers
    assert destroyed_keys == ["unified|name=test"]


def test_get_all_managers_merges_sources(monkeypatch):
    compat.ConfigFactory._managers["compat"] = "legacy-manager"

    class DummyFactory:
        def get_all_managers(self):
            return {"factory": "new-manager"}

    monkeypatch.setattr(compat, "get_config_factory", lambda: DummyFactory())

    all_managers = compat.ConfigFactory.get_all_managers()

    assert all_managers["compat"] == "legacy-manager"
    assert all_managers["factory"] == "new-manager"


def test_create_config_provider_uses_factory(monkeypatch):
    recorded = {}

    class DummyFactory:
        def create_manager(self, manager_type, **kwargs):
            recorded["type"] = manager_type
            recorded["kwargs"] = kwargs
            return "provider-instance"

    monkeypatch.setattr(compat, "get_config_factory", lambda: DummyFactory())

    provider = compat.ConfigFactory.create_config_provider("env", path="config.yml")

    assert provider == "provider-instance"
    assert recorded["type"] == "unified"
    assert recorded["kwargs"] == {"path": "config.yml"}


def test_create_config_provider_fallback(monkeypatch):
    def failing_factory():
        raise RuntimeError("no factory")

    monkeypatch.setattr(compat, "get_config_factory", failing_factory)

    provider = compat.ConfigFactory.create_config_provider("env")

    assert provider is None


def test_get_default_config_manager_fallback(monkeypatch):
    class FailingFactory:
        def create_config_manager(self, name, **kwargs):
            raise RuntimeError("boom")

    sentinel = object()

    def fake_create_config_manager(cls, name="default", config=None, manager_class=None):
        return sentinel

    monkeypatch.setattr(compat, "get_config_factory", lambda: FailingFactory())
    monkeypatch.setattr(
        compat.ConfigFactory,
        "create_config_manager",
        classmethod(fake_create_config_manager),
    )

    manager = compat.get_default_config_manager()

    assert manager is sentinel
    assert compat._default_manager is sentinel


def test_register_provider_records_entry():
    compat.ConfigFactory.register_provider("custom", object)
    assert compat.ConfigFactory.providers["custom"] is object

