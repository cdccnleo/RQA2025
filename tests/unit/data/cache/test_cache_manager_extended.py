import asyncio
import pandas as pd
from unittest.mock import Mock

# Mock数据管理器模块以绕过复杂的导入问题
mock_data_manager = Mock()
mock_data_manager.DataManager = Mock()
mock_data_manager.DataLoaderError = Exception

# 配置DataManager实例方法
mock_instance = Mock()
mock_instance.validate_all_configs.return_value = True
mock_instance.health_check.return_value = {"status": "healthy"}
mock_instance.store_data.return_value = True
mock_instance.has_data.return_value = True
mock_instance.get_metadata.return_value = {"data_type": "test", "symbol": "X"}
mock_instance.retrieve_data.return_value = pd.DataFrame({"col": [1, 2, 3]})
mock_instance.get_stats.return_value = {"total_items": 1}
mock_instance.validate_data.return_value = {"valid": True}
mock_instance.shutdown.return_value = None

mock_data_manager.DataManager.return_value = mock_instance

# Mock整个模块
import sys
sys.modules["src.data.data_manager"] = mock_data_manager


from pathlib import Path

from src.data.cache.cache_manager import CacheManager, CacheConfig


class DummyStrategy:
    def __init__(self):
        self.on_evict_called = False

    def on_get(self, cache, key, entry, config):
        return None

    def on_set(self, cache, key, entry, config):
        return None

    def on_evict(self, cache, config):
        self.on_evict_called = True
        return next(iter(cache)) if cache else None


def _build_manager(tmp_path, **overrides):
    config = CacheConfig(
        enable_disk_cache=overrides.get("enable_disk_cache", True),
        disk_cache_dir=str(tmp_path),
        ttl=overrides.get("ttl", 1),
        max_size=overrides.get("max_size", 10),
        enable_stats=True,
    )
    return CacheManager(config=config, strategy=overrides.get("strategy"))


def test_cache_manager_cleanup_expired(tmp_path):
    manager = _build_manager(tmp_path, enable_disk_cache=False, ttl=1)
    try:
        assert manager.set("expired", "value")
        manager.memory_cache["expired"].created_at -= 10
        cleaned = manager.cleanup_expired()
        assert cleaned == 1
        assert "expired" not in manager.memory_cache
    finally:
        manager.stop()


def test_cache_manager_health_check_failure(monkeypatch, tmp_path):
    manager = _build_manager(tmp_path, enable_disk_cache=False)
    try:
        monkeypatch.setattr(manager, "set", lambda *args, **kwargs: False)
        result = manager.health_check()
        assert result["status"] == "error"
        assert "Set operation failed" in result["message"]
    finally:
        manager.stop()


def test_cache_manager_disk_roundtrip(tmp_path):
    manager = _build_manager(tmp_path)
    try:
        assert manager.set("disk_key", {"foo": "bar"})
        assert manager.get("disk_key") == {"foo": "bar"}
        stats = manager.get_stats()
        assert stats["disk_cache"]["file_count"] >= 1
        assert stats["disk_cache"]["cache_dir"] == str(Path(tmp_path))
    finally:
        manager.clear()
        manager.stop()


def test_cache_manager_eviction_strategy(tmp_path):
    strategy = DummyStrategy()
    manager = _build_manager(tmp_path, max_size=1, strategy=strategy)
    try:
        manager.set("first", 1)
        manager.set("second", 2)
        assert strategy.on_evict_called
        assert len(manager.memory_cache) <= 1
    finally:
        manager.stop()


def test_cache_manager_get_stats_includes_disk(tmp_path):
    manager = _build_manager(tmp_path)
    try:
        manager.set("k1", "v1")
        stats = manager.get_stats()
        assert stats["disk_cache_enabled"] is True
        assert "disk_cache" in stats
    finally:
        manager.stop()


def test_cache_manager_exists_checks_disk(tmp_path):
    manager = _build_manager(tmp_path)
    try:
        manager.set("persisted", "value")
        assert manager.exists("persisted") is True
        manager.delete("persisted")
        assert manager.exists("persisted") is False
    finally:
        manager.stop()

