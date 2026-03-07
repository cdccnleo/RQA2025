import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.features.core.config import FeatureRegistrationConfig, FeatureType
from src.features.core.config_integration import ConfigScope
from src.features.core.feature_store import FeatureStore, StoreConfig


class DummyConfigManager:
    """简化的配置管理器，避免依赖基础设施实现。"""

    def __init__(self):
        self._values = {}
        self._watchers = []

    def set_processing_config(self, **kwargs):
        self._values[ConfigScope.PROCESSING] = kwargs

    def get_config(self, scope: ConfigScope):
        return self._values.get(scope)

    def register_config_watcher(self, scope: ConfigScope, callback):
        self._watchers.append((scope, callback))

    def notify(self, scope: ConfigScope, key: str, old, new):
        for registered_scope, callback in self._watchers:
            if registered_scope == scope:
                callback(scope, key, old, new)


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "price": [100.0, 101.5, 102.2],
            "volume": [200, 210, 220],
        }
    )


@pytest.fixture
def registration_config():
    return FeatureRegistrationConfig(
        name="alpha_factor",
        feature_type=FeatureType.TECHNICAL,
        params={"window": 5},
        dependencies=["price", "volume"],
        description="alpha factor",
        tags=["core"],
    )


@pytest.fixture
def store_setup(tmp_path, monkeypatch):
    manager = DummyConfigManager()
    manager.set_processing_config()  # 默认返回 None，沿用传入 config

    monkeypatch.setattr(
        "src.features.core.feature_store.get_config_integration_manager",
        lambda: manager,
    )

    config = StoreConfig(
        base_path=str(tmp_path / "feature_store"),
        compression=False,
        ttl_hours=1,
    )
    store = FeatureStore(config=config)
    return store, manager


def test_store_and_load_feature_roundtrip(store_setup, sample_data, registration_config):
    store, _ = store_setup

    assert store.store_feature(
        registration_config.name, sample_data, registration_config, description="v1"
    )

    loaded = store.load_feature(registration_config.name, registration_config.params)
    assert loaded is not None
    data, metadata = loaded
    assert data.equals(sample_data)
    assert metadata.feature_name == registration_config.name
    assert store.stats["total_stored"] == 1
    assert store.stats["total_loaded"] == 1


def test_store_feature_updates_existing_metadata(store_setup, sample_data, registration_config):
    store, _ = store_setup
    store.store_feature(registration_config.name, sample_data, registration_config, description="initial")

    updated_data = sample_data.copy()
    updated_data["price"] += 10
    assert store.store_feature(
        registration_config.name, updated_data, registration_config, description="updated", tags=["new"]
    )

    loaded = store.load_feature(registration_config.name, registration_config.params)
    assert loaded is not None
    data, metadata = loaded
    assert data.equals(updated_data)
    assert metadata.description == "updated"
    assert metadata.tags == ["new"]


def test_load_feature_removes_expired_entries(store_setup, sample_data, registration_config):
    store, _ = store_setup
    store.store_feature(registration_config.name, sample_data, registration_config)

    feature_id = store._generate_feature_id(registration_config.name, registration_config.params)
    metadata_file = store.metadata_path / f"{feature_id}.json"
    metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    metadata["updated_at"] = (
        datetime.now() - timedelta(hours=store.config.ttl_hours + 1)
    ).isoformat()
    metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    assert store.load_feature(registration_config.name, registration_config.params) is None
    assert not metadata_file.exists()
    assert store.stats["cache_misses"] >= 1


def test_get_store_stats_reports_usage(store_setup, sample_data, registration_config):
    store, _ = store_setup

    store.store_feature(registration_config.name, sample_data, registration_config)
    extra_config = FeatureRegistrationConfig(
        name="beta_factor",
        feature_type=FeatureType.TECHNICAL,
        params={"window": 10},
    )
    store.store_feature(extra_config.name, sample_data * 2, extra_config)

    stats = store.get_store_stats()
    assert "hit_rate" in stats
    assert stats["feature_count"] == 2
    assert stats["total_size_mb"] > 0


def test_config_change_updates_base_path(store_setup, tmp_path):
    store, manager = store_setup
    new_path = tmp_path / "new_store_base"
    manager.notify(
        ConfigScope.PROCESSING,
        "base_path",
        str(store.base_path),
        str(new_path),
    )

    assert store.base_path == new_path
    assert Path(store.metadata_path).exists()
    assert Path(store.data_path).exists()


def test_feature_store_applies_processing_config_values(tmp_path, monkeypatch):
    manager = DummyConfigManager()
    manager.set_processing_config(
        base_path=str(tmp_path / "managed_base"),
        max_size_mb=2048,
        ttl_hours=6,
        compression=True,
        use_filesystem=True,
        max_workers=8,
    )
    monkeypatch.setattr(
        "src.features.core.feature_store.get_config_integration_manager",
        lambda: manager,
    )

    store = FeatureStore(config=StoreConfig(base_path=str(tmp_path / "ignored")))

    assert store.config.max_size_mb == 2048
    assert store.config.ttl_hours == 6
    assert store.config.compression is True
    assert store.base_path == Path(tmp_path / "managed_base")


def test_store_feature_returns_false_when_data_save_fails(
    store_setup, sample_data, registration_config, monkeypatch
):
    store, _ = store_setup

    monkeypatch.setattr(
        FeatureStore,
        "_save_feature_data",
        lambda self, feature_id, data: False,
    )

    assert (
        store.store_feature(registration_config.name, sample_data, registration_config)
        is False
    )
    assert store.stats["total_stored"] == 0


def test_store_feature_rolls_back_on_metadata_failure(
    store_setup, sample_data, registration_config, monkeypatch
):
    store, _ = store_setup

    def fake_save_metadata(self, feature_id, metadata):
        return False

    monkeypatch.setattr(FeatureStore, "_save_metadata", fake_save_metadata)

    assert (
        store.store_feature(
            registration_config.name, sample_data, registration_config
        )
        is False
    )

    feature_id = store._generate_feature_id(
        registration_config.name, registration_config.params
    )
    data_file = store.data_path / f"{feature_id}.pkl"
    assert not data_file.exists()


def test_cleanup_expired_removes_multiple_features(
    store_setup, sample_data, registration_config
):
    store, _ = store_setup
    store.store_feature(registration_config.name, sample_data, registration_config)

    another_config = FeatureRegistrationConfig(
        name="gamma_factor",
        feature_type=FeatureType.TECHNICAL,
        params={"window": 8},
    )
    store.store_feature(another_config.name, sample_data, another_config)

    for cfg in (registration_config, another_config):
        feature_id = store._generate_feature_id(cfg.name, cfg.params)
        metadata_file = store.metadata_path / f"{feature_id}.json"
        metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
        metadata["updated_at"] = (
            datetime.now() - timedelta(hours=store.config.ttl_hours + 2)
        ).isoformat()
        metadata_file.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    removed = store.cleanup_expired()
    assert removed == 2
    assert store.stats["total_deleted"] >= 2


def test_list_features_filters_by_type_and_tags(
    store_setup, sample_data, registration_config
):
    store, _ = store_setup
    store.store_feature(registration_config.name, sample_data, registration_config)

    sentiment_config = FeatureRegistrationConfig(
        name="sentiment_alpha",
        feature_type=FeatureType.SENTIMENT,
        params={"mode": "nlp"},
        tags=["nlp"],
    )
    store.store_feature(
        sentiment_config.name, sample_data, sentiment_config, tags=["nlp"]
    )

    tech_features = store.list_features(feature_type=FeatureType.TECHNICAL.value)
    assert len(tech_features) == 1
    assert tech_features[0].feature_name == registration_config.name

    tagged = store.list_features(tags=["nlp"])
    assert len(tagged) == 1
    assert tagged[0].feature_name == sentiment_config.name


def test_load_feature_handles_metadata_error(
    store_setup, sample_data, registration_config, monkeypatch
):
    store, _ = store_setup
    store.store_feature(registration_config.name, sample_data, registration_config)

    def boom(*_args, **_kwargs):
        raise RuntimeError("metadata error")

    monkeypatch.setattr(FeatureStore, "_load_metadata", boom)
    assert store.load_feature(registration_config.name, registration_config.params) is None


def test_load_feature_missing_data_file_returns_none(
    store_setup, sample_data, registration_config
):
    store, _ = store_setup
    store.store_feature(registration_config.name, sample_data, registration_config)
    feature_id = store._generate_feature_id(
        registration_config.name, registration_config.params
    )
    data_file = store.data_path / f"{feature_id}.pkl"
    data_file.unlink()

    assert store.load_feature(registration_config.name, registration_config.params) is None


def test_delete_feature_removes_files_and_updates_stats(
    store_setup, sample_data, registration_config
):
    store, _ = store_setup
    store.store_feature(registration_config.name, sample_data, registration_config)
    feature_id = store._generate_feature_id(
        registration_config.name, registration_config.params
    )

    assert store.delete_feature(feature_id) is True
    assert store.stats["total_deleted"] == 1
    assert not (store.metadata_path / f"{feature_id}.json").exists()
    assert not (store.data_path / f"{feature_id}.pkl").exists()


def test_store_with_compression_roundtrip(tmp_path, sample_data, registration_config, monkeypatch):
    manager = DummyConfigManager()
    manager.set_processing_config(base_path=str(tmp_path / "compressed"))
    monkeypatch.setattr(
        "src.features.core.feature_store.get_config_integration_manager",
        lambda: manager,
    )
    config = StoreConfig(base_path=str(tmp_path / "compressed"), compression=True)
    store = FeatureStore(config=config)

    assert store.store_feature(
        registration_config.name, sample_data, registration_config
    )
    loaded = store.load_feature(registration_config.name, registration_config.params)
    assert loaded is not None
    data, _ = loaded
    assert data.equals(sample_data)


def test_list_features_ignores_invalid_metadata(store_setup, tmp_path):
    store, _ = store_setup
    bad_metadata = store.metadata_path / "invalid.json"
    bad_metadata.write_text("{ not: valid json", encoding="utf-8")

    features = store.list_features()
    assert isinstance(features, list)


