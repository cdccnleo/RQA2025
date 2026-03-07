#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Config integration unit tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.features.core.config_integration import (
    ConfigScope,
    FeatureConfigIntegrationManager,
)

pytestmark = pytest.mark.features


@pytest.fixture()
def manager(tmp_path):
    mgr = FeatureConfigIntegrationManager(config_manager=None)
    mgr._local_config.config_dir = str(tmp_path / "feature_cfg")
    return mgr


def test_set_and_get_processing_config(manager: FeatureConfigIntegrationManager):
    assert manager.set_config(ConfigScope.PROCESSING, "batch_size", 2048)
    assert manager.get_config(ConfigScope.PROCESSING, "batch_size") == 2048

    summary = manager.get_config_summary()
    assert summary["processing_config"]["batch_size"] == 2048


def test_global_and_monitoring_config(manager: FeatureConfigIntegrationManager):
    assert manager.set_config(ConfigScope.GLOBAL, "environment", "production")
    assert manager.set_config(ConfigScope.MONITORING, "enable_monitoring", False)

    assert manager.get_config(ConfigScope.GLOBAL, "environment") == "production"
    monitoring = manager.get_config(ConfigScope.MONITORING)
    assert monitoring["enable_monitoring"] is False


def test_register_and_notify_watchers(manager: FeatureConfigIntegrationManager):
    events = []

    def watcher(scope, key, old, new):
        events.append((scope, key, old, new))

    assert manager.register_config_watcher(ConfigScope.PROCESSING, watcher)
    manager.notify_config_change(ConfigScope.PROCESSING, "batch_size", 1000, 2048)
    assert events == [(ConfigScope.PROCESSING, "batch_size", 1000, 2048)]


def test_save_config_persists_local_files(manager: FeatureConfigIntegrationManager, tmp_path):
    manager.set_config(ConfigScope.SENTIMENT, "enable_caching", False)
    assert manager.save_config(ConfigScope.GLOBAL)

    config_dir = Path(manager._local_config.config_dir)
    assert (config_dir / "technical.json").exists()
    assert (config_dir / "processing.json").exists()

    with open(config_dir / "sentiment.json", "r", encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["enable_caching"] is False


def test_load_from_local_overrides_defaults(tmp_path):
    config_dir = tmp_path / "feature_cfg_local"
    config_dir.mkdir()
    processing_payload = {
        "batch_size": 32,
        "timeout": 10,
        "retry_count": 1,
        "retry_delay": 0.5,
        "max_memory_usage": 256.0,
        "enable_memory_monitoring": False,
        "continue_on_error": True,
        "log_errors": False,
        "enable_parallel_processing": False,
        "max_workers": 2,
        "chunk_size": 128,
    }
    with open(config_dir / "processing.json", "w", encoding="utf-8") as fh:
        json.dump(processing_payload, fh)

    manager = FeatureConfigIntegrationManager(config_manager=None)
    manager._local_config.config_dir = str(config_dir)
    manager._load_from_local()
    assert manager.get_config(ConfigScope.PROCESSING, "batch_size") == 32


def test_infrastructure_load_and_save():
    class FakeConfigManager:
        def __init__(self):
            self.data = {
                "technical": {"macd_fast": 8, "macd_slow": 20, "macd_signal": 5},
                "sentiment": {"enable_sentiment_analysis": True},
                "processing": {"batch_size": 128, "timeout": 20.0, "retry_count": 1, "retry_delay": 0.5,
                               "max_memory_usage": 512.0, "enable_memory_monitoring": True,
                               "continue_on_error": False, "log_errors": True,
                               "enable_parallel_processing": True, "max_workers": 2, "chunk_size": 64}
            }
            self.saved = {}

        def get(self, key, default=None):
            return self.data.get(key, default)

        def set(self, key, value):
            self.saved[key] = value

    fake_manager = FakeConfigManager()
    manager = FeatureConfigIntegrationManager(config_manager=fake_manager)
    assert manager._use_infrastructure_config
    assert manager.get_config(ConfigScope.TECHNICAL, "macd_fast") == 8

    manager.set_config(ConfigScope.TECHNICAL, "macd_fast", 13)
    manager.set_config(ConfigScope.SENTIMENT, "enable_sentiment_analysis", False)
    assert manager.save_config(ConfigScope.GLOBAL)
    assert "technical" in fake_manager.saved
    assert fake_manager.saved["technical"]["macd_fast"] == 13

