import json
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import pytest

from src.features.core.config import FeatureRegistrationConfig, FeatureType
from src.features.core.config_integration import ConfigScope
from src.features.core.feature_store import FeatureStore, StoreConfig


class _StubConfigManager:
    def __init__(self):
        self.watchers = []

    def get_config(self, scope, key=None):
        return {}

    def register_config_watcher(self, scope, callback):
        self.watchers.append((scope, callback))
        return True


@pytest.fixture
def feature_store(tmp_path, monkeypatch):
    stub_manager = _StubConfigManager()
    monkeypatch.setattr(
        "src.features.core.feature_store.get_config_integration_manager",
        lambda: stub_manager,
    )

    base_path = tmp_path / "store"
    config = StoreConfig(
        base_path=str(base_path),
        ttl_hours=1,
        compression=False,
        max_size_mb=128,
    )
    store = FeatureStore(config=config)
    return store, stub_manager, base_path


@pytest.fixture
def sample_config():
    return FeatureRegistrationConfig(
        name="sma_feature",
        feature_type=FeatureType.TECHNICAL,
        params={"window": 5},
        dependencies=["close"],
        version="1.0",
    )


@pytest.fixture
def sample_frame():
    return pd.DataFrame({"close": [1.0, 1.5, 2.0, 2.5, 3.0]})


def test_store_and_load_feature_roundtrip(feature_store, sample_config, sample_frame):
    store, stub_manager, base_path = feature_store

    assert len(stub_manager.watchers) == 1
    assert store.store_feature("sma_feature", sample_frame, sample_config, tags=["alpha"])

    loaded = store.load_feature("sma_feature", sample_config.params)
    assert loaded is not None
    loaded_frame, metadata = loaded
    pd.testing.assert_frame_equal(loaded_frame, sample_frame)
    assert metadata.feature_name == "sma_feature"
    assert metadata.tags == ["alpha"]
    assert store.stats["total_stored"] == 1
    assert store.stats["cache_hits"] == 1


def test_store_feature_updates_existing_metadata(feature_store, sample_config, sample_frame):
    store, _, _ = feature_store
    store.store_feature("sma_feature", sample_frame, sample_config, description="base")

    feature_id = store._generate_feature_id("sma_feature", sample_config.params)
    first_meta = store._load_metadata(feature_id)
    assert first_meta is not None

    updated_frame = sample_frame.assign(close=[10, 11, 12, 13, 14])
    store.store_feature("sma_feature", updated_frame, sample_config, description="updated")

    second_meta = store._load_metadata(feature_id)
    assert second_meta is not None
    assert second_meta.description == "updated"
    assert second_meta.updated_at >= first_meta.updated_at
    assert tuple(second_meta.data_shape) == updated_frame.shape
    assert second_meta.checksum != first_meta.checksum


def test_cleanup_expired_removes_old_entries(feature_store, sample_config, sample_frame, monkeypatch):
    store, _, _ = feature_store
    store.store_feature("sma_feature", sample_frame, sample_config)

    feature_id = store._generate_feature_id("sma_feature", sample_config.params)
    metadata = store._load_metadata(feature_id)
    assert metadata is not None

    metadata.updated_at = datetime.now() - timedelta(hours=5)
    store._save_metadata(feature_id, metadata)

    monkeypatch.setattr(
        store,
        "list_features",
        lambda feature_type=None, tags=None: [metadata],
    )

    removed = store.cleanup_expired()
    assert removed == 1
    assert not store._feature_exists(feature_id)


def test_get_store_stats_reports_hit_rate(feature_store, sample_config, sample_frame):
    store, _, _ = feature_store

    store.store_feature("sma_feature", sample_frame, sample_config)
    assert store.load_feature("sma_feature", sample_config.params) is not None
    assert store.load_feature("unknown_feature", {}) is None

    stats = store.get_store_stats()
    assert pytest.approx(stats["hit_rate"], rel=1e-6) == 0.5
    assert stats["feature_count"] == 1


def test_on_config_change_updates_paths(feature_store):
    store, _, base_path = feature_store
    new_path = base_path / "updated"
    store._on_config_change(ConfigScope.PROCESSING, "base_path", str(store.base_path), str(new_path))

    assert store.base_path == new_path
    assert store.metadata_path.exists()
    assert store.data_path.exists()


def test_store_feature_handles_save_failure(feature_store, sample_config, sample_frame, monkeypatch):
    store, _, _ = feature_store

    monkeypatch.setattr(store, "_save_feature_data", lambda *_args, **_kwargs: False)
    assert store.store_feature("broken_feature", sample_frame, sample_config) is False


def test_store_feature_cleans_up_on_metadata_failure(feature_store, sample_config, sample_frame, monkeypatch):
    store, _, _ = feature_store
    delete_calls = []

    def fake_save(feature_id, data):
        return True

    def fake_save_meta(feature_id, metadata):
        return False

    def fake_delete(feature_id):
        delete_calls.append(feature_id)

    monkeypatch.setattr(store, "_save_feature_data", fake_save)
    monkeypatch.setattr(store, "_save_metadata", fake_save_meta)
    monkeypatch.setattr(store, "_delete_feature_data", fake_delete)

    assert store.store_feature("cleanup_feature", sample_frame, sample_config) is False
    assert delete_calls


def test_load_feature_handles_missing_data(feature_store, sample_config, sample_frame, monkeypatch):
    store, _, _ = feature_store
    store.store_feature("sma_feature", sample_frame, sample_config)

    monkeypatch.setattr(store, "_load_feature_data", lambda *_args, **_kwargs: None)
    assert store.load_feature("sma_feature", sample_config.params) is None


def test_store_feature_exception_path(feature_store, sample_config, sample_frame, monkeypatch):
    store, _, _ = feature_store
    monkeypatch.setattr(store, "_generate_feature_id", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    assert store.store_feature("boom_feature", sample_frame, sample_config) is False


def test_get_store_stats_handles_corrupted_metadata(feature_store, sample_config, sample_frame):
    store, _, base_path = feature_store
    store.store_feature("sma_feature", sample_frame, sample_config)

    corrupt_file = base_path / "metadata" / "corrupt.json"
    corrupt_file.write_text("{ invalid json", encoding="utf-8")

    stats = store.get_store_stats()
    assert "feature_count" in stats


def test_load_feature_expired_triggers_deletion(feature_store, sample_config, sample_frame, monkeypatch):
    store, _, _ = feature_store
    store.store_feature("sma_feature", sample_frame, sample_config)

    feature_id = store._generate_feature_id("sma_feature", sample_config.params)
    metadata = store._load_metadata(feature_id)
    assert metadata is not None
    metadata.updated_at = datetime.now() - timedelta(hours=store.config.ttl_hours + 2)
    store._save_metadata(feature_id, metadata)

    deleted = {}

    def fake_delete(fid):
        deleted["id"] = fid
        return True

    monkeypatch.setattr(store, "delete_feature", fake_delete)

    result = store.load_feature("sma_feature", sample_config.params)
    assert result is None
    assert deleted.get("id") == feature_id
    assert store.stats["cache_misses"] >= 1


def test_delete_feature_handles_failure(feature_store, monkeypatch):
    store, _, _ = feature_store

    def boom(*_args, **_kwargs):
        raise RuntimeError("delete failed")

    monkeypatch.setattr(store, "_delete_feature_data", boom)
    assert store.delete_feature("any") is False


def test_on_config_change_updates_simple_fields(feature_store):
    store, _, _ = feature_store
    store._on_config_change(ConfigScope.PROCESSING, "ttl_hours", store.config.ttl_hours, 10)
    assert store.config.ttl_hours == 10


def test_is_expired_respects_non_positive_ttl(feature_store, sample_config, sample_frame):
    store, _, _ = feature_store
    store.config.ttl_hours = 0
    store.store_feature("sma_feature", sample_frame, sample_config)
    feature_id = store._generate_feature_id("sma_feature", sample_config.params)
    metadata = store._load_metadata(feature_id)
    assert metadata is not None
    metadata.updated_at = datetime.now() - timedelta(days=10)
    assert store._is_expired(metadata) is False


def test_calculate_checksum_handles_exception(feature_store):
    store, _, _ = feature_store

    class BoomFrame(pd.DataFrame):
        def to_string(self, *args, **kwargs):
            raise RuntimeError("boom")

    checksum = store._calculate_checksum(BoomFrame({"a": [1, 2]}))
    assert checksum == ""


def test_list_features_filters_by_type_and_tags(feature_store, sample_config, sample_frame):
    store, _, _ = feature_store
    store.store_feature("sma_feature", sample_frame, sample_config, tags=["alpha"])
    results = store.list_features(feature_type=sample_config.feature_type.value, tags=["alpha"])
    assert len(results) == 1
    assert results[0].feature_name == "sma_feature"
    assert store.list_features(feature_type="unknown") == []
    assert store.list_features(tags=["beta"]) == []


def test_list_features_handles_exception(feature_store, monkeypatch):
    store, _, _ = feature_store

    class StubPath:
        def glob(self, *_args, **_kwargs):
            raise RuntimeError("glob failed")

    store.metadata_path = StubPath()
    assert store.list_features() == []


def test_update_feature_without_metadata_returns_false(feature_store, sample_config, sample_frame, monkeypatch):
    store, _, _ = feature_store
    feature_id = store._generate_feature_id("sma_feature", sample_config.params)

    monkeypatch.setattr(store, "_save_feature_data", lambda *_: True)
    monkeypatch.setattr(store, "_load_metadata", lambda *_: None)
    assert store._update_feature(feature_id, sample_frame, sample_config, "", None) is False


def test_load_metadata_invalid_json_returns_none(feature_store, sample_config, sample_frame):
    store, _, base_path = feature_store
    feature_id = "bad"
    metadata_file = base_path / "metadata" / f"{feature_id}.json"
    metadata_file.write_text("{ invalid json", encoding="utf-8")
    assert store._load_metadata(feature_id) is None

