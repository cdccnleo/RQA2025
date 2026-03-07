import json
from pathlib import Path

import pytest

from src.features.core.config_integration import (
    ConfigScope,
    FeatureConfigIntegrationManager,
)


class StubConfigManager:
    def __init__(self):
        self._storage = {}

    def get(self, key, default=None):
        return self._storage.get(key, default)

    def set(self, key, value):
        self._storage[key] = value


@pytest.fixture
def local_config_dir(tmp_path):
    config_dir = tmp_path / "config" / "features"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "technical.json").write_text(
        json.dumps({"macd_fast": 15, "macd_slow": 30}), encoding="utf-8"
    )
    (config_dir / "sentiment.json").write_text(
        json.dumps({"model_type": "stub", "enable_topic_analysis": True}), encoding="utf-8"
    )
    (config_dir / "processing.json").write_text(
        json.dumps({"batch_size": 32, "timeout": 60}), encoding="utf-8"
    )
    return config_dir


def test_local_configuration_loads_from_files(local_config_dir, monkeypatch):
    manager = FeatureConfigIntegrationManager(config_manager=None)
    monkeypatch.setattr(manager._local_config, "config_dir", str(local_config_dir))
    manager._load_from_local()

    tech_config = manager.get_config(ConfigScope.TECHNICAL)
    assert tech_config["macd_fast"] == 15

    processing_config = manager.get_config(ConfigScope.PROCESSING)
    assert processing_config["batch_size"] == 32


def test_infrastructure_configuration_priority():
    stub = StubConfigManager()
    stub.set("technical", {"macd_fast": 21})
    stub.set("processing", {"batch_size": 64})

    manager = FeatureConfigIntegrationManager(config_manager=stub)
    assert manager.get_config(ConfigScope.TECHNICAL)["macd_fast"] == 21
    manager._load_configuration()
    assert manager.get_config(ConfigScope.PROCESSING)["batch_size"] == 64


def test_get_and_set_global_values(monkeypatch, tmp_path):
    manager = FeatureConfigIntegrationManager(config_manager=None)

    assert manager.set_config(ConfigScope.GLOBAL, "environment", "testing") is True
    assert manager.get_config(ConfigScope.GLOBAL, "environment") == "testing"

    save_dir = tmp_path / "out_config"
    monkeypatch.setattr(manager._local_config, "config_dir", str(save_dir))
    assert manager.save_config(ConfigScope.GLOBAL) is True


def test_register_config_watcher_and_notify():
    manager = FeatureConfigIntegrationManager(config_manager=None)
    events = []

    def watcher(scope, key, old, new):
        events.append((scope, key, old, new))

    manager.register_config_watcher(ConfigScope.PROCESSING, watcher)
    manager.notify_config_change(ConfigScope.PROCESSING, "batch_size", 10, 20)
    assert events == [(ConfigScope.PROCESSING, "batch_size", 10, 20)]


def test_get_config_returns_none_on_error(monkeypatch):
    manager = FeatureConfigIntegrationManager(config_manager=None)

    assert manager.get_config(ConfigScope.FEATURE) is None


def test_load_configuration_falls_back_to_defaults():
    class FailingManager:
        def get(self, key, default=None):
            raise RuntimeError("boom")

    manager = FeatureConfigIntegrationManager(config_manager=FailingManager())
    assert manager.get_config(ConfigScope.GLOBAL, "environment") == "development"


def test_save_to_infrastructure_success():
    class RecordingManager(StubConfigManager):
        def __init__(self):
            super().__init__()
            self.set_calls = []

        def set(self, key, value):
            self.set_calls.append((key, value))
            super().set(key, value)

    stub = RecordingManager()
    manager = FeatureConfigIntegrationManager(config_manager=stub)
    assert manager.save_config(ConfigScope.TECHNICAL) is True
    assert stub.set_calls and stub.set_calls[0][0] == "technical"


def test_save_to_infrastructure_failure():
    class FailingManager(StubConfigManager):
        def set(self, key, value):
            raise RuntimeError("fail")

    manager = FeatureConfigIntegrationManager(config_manager=FailingManager())
    assert manager.save_config(ConfigScope.TECHNICAL) is False


def test_save_to_local_failure(monkeypatch, tmp_path):
    manager = FeatureConfigIntegrationManager(config_manager=None)
    monkeypatch.setattr(manager._local_config, "config_dir", str(tmp_path))

    def bad_dump(*args, **kwargs):
        raise RuntimeError("dump error")

    monkeypatch.setattr("json.dump", bad_dump)
    assert manager.save_config(ConfigScope.TECHNICAL) is False


def test_save_to_local_global_writes_all(tmp_path):
    manager = FeatureConfigIntegrationManager(config_manager=None)
    manager._local_config.config_dir = str(tmp_path / "config" / "features")
    assert manager.save_config(ConfigScope.GLOBAL) is True
    path = Path(manager._local_config.config_dir)
    assert (path / "technical.json").exists()
    assert (path / "sentiment.json").exists()
    assert (path / "processing.json").exists()


def test_set_config_invalid_key_returns_false():
    manager = FeatureConfigIntegrationManager(config_manager=None)
    assert manager.set_config(ConfigScope.TECHNICAL, "unknown_field", 1) is False


def test_set_config_invalid_scope_returns_false():
    manager = FeatureConfigIntegrationManager(config_manager=None)
    assert manager.set_config(ConfigScope.FEATURE, "environment", "x") is False


def test_notify_config_change_handles_exceptions(caplog):
    manager = FeatureConfigIntegrationManager(config_manager=None)
    events = []

    def good(scope, key, old, new):
        events.append((scope, key, new))

    def bad(*args, **kwargs):
        raise RuntimeError("callback boom")

    manager.register_config_watcher(ConfigScope.SENTIMENT, bad)
    manager.register_config_watcher(ConfigScope.SENTIMENT, good)
    manager.notify_config_change(ConfigScope.SENTIMENT, "model_type", "a", "b")
    assert events == [(ConfigScope.SENTIMENT, "model_type", "b")]


def test_get_config_summary_returns_expected_structure():
    manager = FeatureConfigIntegrationManager(config_manager=None)
    summary = manager.get_config_summary()
    assert summary["environment"] == "development"
    assert "technical_config" in summary


def test_load_from_local_missing_dir(monkeypatch, tmp_path):
    manager = FeatureConfigIntegrationManager(config_manager=None)
    manager._local_config.config_dir = str(tmp_path / "missing_dir")
    manager._load_from_local()  # should not raise
    assert manager.get_config(ConfigScope.GLOBAL, "environment") == "development"


def test_save_to_infrastructure_global(manager_with_stub=None):
    stub = StubConfigManager()
    manager = FeatureConfigIntegrationManager(config_manager=stub)
    assert manager.save_config(ConfigScope.GLOBAL) is True
    assert "technical" in stub._storage
    assert "sentiment" in stub._storage
    assert "processing" in stub._storage

