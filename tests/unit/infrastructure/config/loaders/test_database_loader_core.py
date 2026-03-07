import types

import pytest

from src.infrastructure.config.config_exceptions import ConfigLoadError
from src.infrastructure.config.loaders import database_loader as module
from src.infrastructure.config.loaders.database_loader import DatabaseLoader


def test_init_with_unknown_db_type():
    with pytest.raises(ValueError):
        DatabaseLoader("unknown_db")


def test_can_load_accepts_supported_protocols():
    loader = DatabaseLoader("postgresql")
    assert loader.can_load("postgresql://configs/app")
    assert loader.can_load("redis://cache/key")
    assert not loader.can_load("ftp://invalid/path")
    assert not loader.can_load("")


def test_parse_database_path_roundtrip():
    loader = DatabaseLoader("postgresql")
    db_type, table, key = loader._parse_database_path("postgresql://config_table/myapp")
    assert db_type == "postgresql"
    assert table == "config_table"
    assert key == "myapp"

    with pytest.raises(ValueError):
        loader._parse_database_path("postgresql://missing-key")

    with pytest.raises(ValueError):
        loader._parse_database_path("invalid-protocol://foo/bar")


def test_load_success_returns_metadata(monkeypatch):
    loader = DatabaseLoader("postgresql")
    disconnect_called = {"value": False}

    loader.can_load = lambda source: True
    loader._connect = lambda: setattr(loader, "_connection", object())
    loader._load_config_data = lambda source: {"feature": True}
    loader._disconnect = lambda: disconnect_called.update(value=True)

    result = loader.load("postgresql://config_table/test_key")

    assert result["feature"] is True
    metadata = loader.get_last_metadata()
    assert metadata["database_type"] == "postgresql"
    assert metadata["config_count"] == 1
    assert disconnect_called["value"] is True


def test_load_raises_config_load_error(monkeypatch):
    loader = DatabaseLoader("postgresql")
    loader.can_load = lambda source: True
    loader._connect = lambda: None
    loader._disconnect = lambda: None

    def _fail(source):
        raise ValueError("boom")

    loader._load_config_data = _fail

    with pytest.raises(ConfigLoadError) as excinfo:
        loader.load("postgresql://config_table/test_key")
    assert "Database config loading failed" in str(excinfo.value)


def test_batch_load_partial_failure(monkeypatch):
    loader = DatabaseLoader("postgresql")

    def fake_load(source):
        if "bad" in source:
            raise ConfigLoadError("bad source")
        return {"host": "localhost"}

    loader.load = fake_load  # type: ignore[assignment]

    result = loader.batch_load(
        ["postgresql://ok/key", "postgresql://bad/key", "postgresql://ok2/key"]
    )

    assert "postgresql://ok/key" in result
    assert "postgresql://ok2/key" in result


def test_batch_load_all_failures_raise(monkeypatch):
    loader = DatabaseLoader("postgresql")

    def always_fail(source):
        raise ConfigLoadError("fail")

    loader.load = always_fail  # type: ignore[assignment]

    with pytest.raises(ConfigLoadError):
        loader.batch_load(["postgresql://bad/key"])


def test_sqlite_connect_and_disconnect():
    loader = DatabaseLoader("sqlite", {"database": ":memory:"})
    loader._connect()
    try:
        assert loader._connection is not None
    finally:
        loader._disconnect()
    assert loader._connection is None


def test_redis_connect_uses_dummy_when_driver_missing(monkeypatch):
    monkeypatch.setattr(module, "redis", None)
    loader = DatabaseLoader("redis", {"host": "localhost"})
    loader._connect()
    try:
        assert isinstance(loader._connection, module._DummyRedis)
    finally:
        loader._disconnect()


def test_get_supported_formats_and_extensions():
    loader = DatabaseLoader("postgresql")
    assert loader.get_supported_extensions() == []
    assert loader.get_supported_formats() == [module.ConfigFormat.DATABASE]

