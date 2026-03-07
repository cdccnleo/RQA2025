import time

import pytest

from src.infrastructure.config.services.config_storage_service import ConfigStorageService


class StubStorageBackend:
    def __init__(self):
        self.source_data = {}
        self.kv = {}
        self.load_calls = []
        self.save_calls = []
        self.reload_calls = 0
        self.get_calls = []
        self.set_calls = []
        self.delete_calls = []
        self.exists_calls = []
        self.keys_calls = []
        self.clear_calls = 0

    def load(self, source: str):
        self.load_calls.append(source)
        return self.source_data.get(source, {}).copy()

    def save(self, config, target: str) -> bool:
        self.save_calls.append((target, config.copy()))
        self.source_data[target] = config.copy()
        return True

    def reload(self) -> bool:
        self.reload_calls += 1
        return True

    def get(self, key: str, default=None):
        self.get_calls.append((key, default))
        return self.kv.get(key, default)

    def set(self, key: str, value) -> bool:
        self.set_calls.append((key, value))
        self.kv[key] = value
        return True

    def delete(self, key: str) -> bool:
        self.delete_calls.append(key)
        return self.kv.pop(key, None) is not None

    def exists(self, key: str) -> bool:
        self.exists_calls.append(key)
        return key in self.kv

    def keys(self, pattern: str):
        self.keys_calls.append(pattern)
        return list(self.kv.keys())

    def clear(self) -> bool:
        self.clear_calls += 1
        self.kv.clear()
        self.source_data.clear()
        return True


@pytest.fixture
def backend():
    return StubStorageBackend()


@pytest.fixture
def service(backend):
    return ConfigStorageService(storage_backend=backend, cache_enabled=True, cache_size=2)


def test_load_uses_cache_after_first_call(service, backend):
    source = "postgresql://config_table/service"
    backend.source_data[source] = {"feature": True}

    first = service.load(source)
    assert backend.load_calls == [source]
    assert first["feature"] is True

    # 修改底层数据，新加载仍应返回缓存值
    backend.source_data[source] = {"feature": False}
    second = service.load(source)

    assert backend.load_calls == [source]  # 未再次访问后端
    assert second["feature"] is True
    assert service._stats["cache_hits"] == 1


def test_load_without_backend_raises():
    svc = ConfigStorageService(storage_backend=None)
    with pytest.raises(ValueError):
        svc.load("postgresql://config_table/app")


def test_save_updates_backend_and_cache(service, backend):
    target = "postgresql://config_table/save_target"
    payload = {"threshold": 10}
    assert service.save(payload, target) is True
    assert backend.save_calls == [(target, payload)]

    cached = service.get(target)
    assert cached == payload
    assert backend.get_calls == []  # 缓存命中


def test_get_fetches_from_backend_when_not_cached(service, backend):
    backend.kv["timeout"] = 30
    value = service.get("timeout")

    assert value == 30
    assert backend.get_calls == [("timeout", None)]
    assert service._stats["cache_misses"] == 1


def test_set_persists_and_caches(service, backend):
    assert service.set("mode", "debug") is True
    assert backend.set_calls == [("mode", "debug")]
    assert service.get("mode") == "debug"
    assert backend.get_calls == []  # 来源于缓存


def test_delete_removes_cache(service, backend):
    service.set("obsolete", 1)
    backend.kv["obsolete"] = 1
    assert service.delete("obsolete") is True
    assert backend.delete_calls == ["obsolete"]
    assert "obsolete" not in service._cache


def test_reload_clears_cache_and_calls_backend(service, backend):
    service.set("cache_key", "value")
    assert service.reload() is True
    assert backend.reload_calls == 1
    assert service._cache == {}


def test_keys_without_backend_returns_empty():
    svc = ConfigStorageService(storage_backend=None)
    assert svc.keys() == []


def test_clear_invokes_backend_and_empties_cache(service, backend):
    service.set("entry", 1)
    assert service.clear() is True
    assert backend.clear_calls == 1
    assert service._cache == {}


def test_cache_eviction_uses_lru_policy(backend):
    svc = ConfigStorageService(storage_backend=backend, cache_enabled=True, cache_size=1)
    first = "postgresql://config_table/first"
    second = "postgresql://config_table/second"
    backend.source_data[first] = {"value": 1}
    backend.source_data[second] = {"value": 2}

    svc.load(first)
    time.sleep(0.01)  # 确保访问时间不同
    svc.load(second)

    assert first not in svc._cache
    assert second in svc._cache

