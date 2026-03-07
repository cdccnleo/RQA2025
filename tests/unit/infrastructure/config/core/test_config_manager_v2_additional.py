from unittest.mock import MagicMock

import pytest

import src.infrastructure.config.core.config_manager_v2 as manager_module


@pytest.fixture(autouse=True)
def patch_services(monkeypatch):
    class FakeStorageService:
        def __init__(self, cache_enabled=True):
            self.cache_enabled = cache_enabled
            self.backend = None
            self.loaded_sources = []
            self.saved = []
            self.reloaded = False
            self.cleaned = False
            self._initialized = True

        def set_storage_backend(self, backend):
            self.backend = backend

        def load(self, source):
            self.loaded_sources.append(source)
            return {"source": source, "value": len(source)}

        def save(self, config, target):
            self.saved.append((config, target))
            return True

        def reload(self):
            self.reloaded = True
            return True

        def get_storage_stats(self):
            return {"loaded": len(self.loaded_sources), "saved": len(self.saved)}

        def cleanup(self):
            self.cleaned = True

    class FakeOperationsService:
        def __init__(self, storage_service):
            self.storage_service = storage_service
            self.validators = []
            self.listeners = []
            self.cleaned = False
            self._initialized = True

        def add_validator(self, validator):
            self.validators.append(validator)

        def add_listener(self, listener):
            self.listeners.append(listener)

        def get_operation_stats(self):
            return {
                "validators": len(self.validators),
                "listeners": len(self.listeners),
            }

        def cleanup(self):
            self.cleaned = True

    monkeypatch.setattr(
        manager_module, "ConfigStorageService", FakeStorageService, raising=False
    )
    monkeypatch.setattr(
        manager_module, "ConfigOperationsService", FakeOperationsService, raising=False
    )

    return FakeStorageService, FakeOperationsService


@pytest.fixture
def manager():
    mgr = manager_module.ConfigManagerV2()
    return mgr


def test_basic_set_get_delete_flow(manager):
    assert manager.get("missing", default="fallback") == "fallback"
    assert manager._stats["operations"] == 1

    assert manager.set("key", "value") is True
    assert manager.get("key") == "value"
    assert manager.exists("key") is True

    assert sorted(manager.keys()) == ["key"]
    assert manager.keys("y") == ["key"]

    assert manager.delete("key") is True
    assert manager.delete("key") is False
    assert manager.exists("key") is False

    assert manager.clear() is True


def test_error_handling_for_get_manager(manager):
    class FailingLock:
        def __enter__(self):
            raise RuntimeError("boom")

        def __exit__(self, exc_type, exc, tb):
            return False

    manager._config_lock = FailingLock()
    assert manager.get("x", default="safe") == "safe"
    assert manager._stats["errors"] == 1


def test_load_save_reload_and_stats(monkeypatch, manager):
    config = manager.load_config("source.yaml")
    assert config["source"] == "source.yaml"
    assert manager.get("source") == "source.yaml"
    assert manager.get("value") == len("source.yaml")

    assert manager.save_config({"a": 1}, "target.json") is True
    assert manager.reload_config() is True

    stats = manager.get_stats()
    assert stats["basic"]["operations"] >= 1
    assert stats["storage"]["loaded"] == 1
    assert stats["operations"]["validators"] == 0
    assert stats["config_count"] >= 2


def test_load_config_raises(monkeypatch, manager):
    def fail_load(_):
        raise ValueError("load error")

    manager._storage_service.load = fail_load

    with pytest.raises(ValueError):
        manager.load_config("broken")


def test_save_and_reload_raise(monkeypatch, manager):
    manager._storage_service.save = MagicMock(side_effect=RuntimeError("save boom"))
    manager._storage_service.reload = MagicMock(side_effect=RuntimeError("reload boom"))

    with pytest.raises(RuntimeError):
        manager.save_config({}, "t")

    with pytest.raises(RuntimeError):
        manager.reload_config()


def test_health_status_success_and_failure(manager):
    status = manager.get_health_status()
    assert status["healthy"] is True
    assert status["services"]["storage"]["healthy"] is True

    manager._storage_service = object()
    failure = manager.get_health_status()
    assert failure["healthy"] is False
    assert "error" in failure


def test_batch_operations_success_and_failure(manager):
    manager.set("a", 1)
    result = manager.batch_get(["a", "b"])
    assert result["a"] == 1 and result["b"] is None

    assert manager.batch_set({"b": 2, "c": 3}) is True
    assert manager.get("b") == 2

    original_set = manager.set

    def failing_set(*args, **kwargs):
        raise RuntimeError("fail")

    manager.set = failing_set
    manager._logger = MagicMock()
    assert manager.batch_set({"d": 4}) is False
    manager.set = original_set


def test_cleanup_and_service_info(manager):
    manager.add_validator(lambda _: None)
    manager.add_listener(lambda *_: None)
    manager.cleanup()

    assert manager._operations_service.cleaned is True
    assert manager._storage_service.cleaned is True
    assert manager.get_all_config() == {}

    info = manager._get_service_info()
    assert info["service_name"].startswith("config")
    assert "ConfigManagerV2" in repr(manager)
    assert "ConfigManagerV2" in str(manager)


def test_create_with_file_storage_success(monkeypatch):
    created_configs = {}

    class DummyFileStorage:
        def __init__(self, config):
            self.config = config
            created_configs["instance"] = self

    monkeypatch.setattr(manager_module, "FileConfigStorage", DummyFileStorage, raising=False)

    mgr = manager_module.ConfigManagerV2.create_with_file_storage("config.yaml")
    assert isinstance(mgr, manager_module.ConfigManagerV2)
    backend = mgr._storage_service.backend
    assert isinstance(backend, DummyFileStorage)
    assert backend.config.path == "config.yaml"


def test_create_with_file_storage_failure(monkeypatch):
    def exploding_storage(_):
        raise RuntimeError("boom")

    monkeypatch.setattr(manager_module, "FileConfigStorage", exploding_storage, raising=False)

    mgr = manager_module.ConfigManagerV2.create_with_file_storage("conf.yaml")
    assert isinstance(mgr, manager_module.ConfigManagerV2)
    assert mgr._storage_service.backend is None


