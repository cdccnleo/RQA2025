#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FeatureStore 行为级单元测试

聚焦核心缓存/持久化路径，确保元数据、TTL 与统计逻辑可控，
以支撑特征层投产要求的 80% 覆盖目标。
"""

from __future__ import annotations

import pandas as pd
import pytest
from datetime import datetime, timedelta
from pathlib import Path

from src.features.core.config import FeatureRegistrationConfig, FeatureType
from src.features.core.feature_store import FeatureStore, StoreConfig

pytestmark = pytest.mark.features


class DummyConfigManager:
    """可控的配置管理桩件，避免依赖基础设施层"""

    def __init__(self, overrides: dict | None = None):
        self._overrides = overrides or {}
        self.watchers = []

    def get_config(self, scope):
        return self._overrides if scope.name == "PROCESSING" else None

    def register_config_watcher(self, scope, callback):
        self.watchers.append((scope, callback))


@pytest.fixture()
def feature_store(monkeypatch, tmp_path: Path) -> FeatureStore:
    overrides = {
        "base_path": str(tmp_path / "store"),
        "compression": False,
        "ttl_hours": 1,
    }
    dummy_manager = DummyConfigManager(overrides)
    monkeypatch.setattr(
        "src.features.core.feature_store.get_config_integration_manager",
        lambda: dummy_manager,
    )
    config = StoreConfig(
        base_path=overrides["base_path"],
        compression=False,
        ttl_hours=overrides["ttl_hours"],
    )
    return FeatureStore(config)


def _make_dataframe():
    return pd.DataFrame(
        {
            "close": [1.0, 1.1, 1.2],
            "volume": [100, 110, 120],
        }
    )


def _make_config(name: str, params: dict | None = None) -> FeatureRegistrationConfig:
    return FeatureRegistrationConfig(
        name=name,
        feature_type=FeatureType.TECHNICAL,
        params=params or {"window": 5},
        dependencies=["price"],
    )


def test_store_and_load_roundtrip(feature_store: FeatureStore):
    config = _make_config("trend_feature", {"window": 3})
    assert feature_store.store_feature("trend_feature", _make_dataframe(), config)

    loaded = feature_store.load_feature("trend_feature", config.params)
    assert loaded is not None
    data, metadata = loaded
    assert data.equals(_make_dataframe())
    assert metadata.feature_name == "trend_feature"
    assert metadata.params == config.params


def test_store_feature_updates_existing_metadata(feature_store: FeatureStore):
    config = _make_config("trend_feature", {"window": 5})
    first_df = _make_dataframe()
    second_df = pd.DataFrame({"close": [2.0, 2.1], "volume": [200, 210]})

    assert feature_store.store_feature("trend_feature", first_df, config)
    assert feature_store.store_feature("trend_feature", second_df, config)

    loaded = feature_store.load_feature("trend_feature", config.params)
    assert loaded is not None
    data, metadata = loaded
    assert data.equals(second_df)
    assert tuple(metadata.data_shape) == second_df.shape
    assert metadata.updated_at >= metadata.created_at


def test_cleanup_expired_removes_old_entries(feature_store: FeatureStore):
    config = _make_config("expiring_feature", {"window": 7})
    assert feature_store.store_feature("expiring_feature", _make_dataframe(), config)

    feature_id = feature_store._generate_feature_id("expiring_feature", config.params)
    metadata = feature_store._load_metadata(feature_id)
    metadata.updated_at = datetime.now() - timedelta(hours=5)
    feature_store._save_metadata(feature_id, metadata)

    removed = feature_store.cleanup_expired()
    assert removed == 1
    assert feature_store.load_feature("expiring_feature", config.params) is None


def test_get_store_stats_reports_usage(feature_store: FeatureStore):
    config = _make_config("stats_feature", {"window": 9})
    feature_store.store_feature("stats_feature", _make_dataframe(), config)
    stats = feature_store.get_store_stats()

    assert stats["total_stored"] >= 1
    assert "hit_rate" in stats
    assert stats["feature_count"] >= 1

