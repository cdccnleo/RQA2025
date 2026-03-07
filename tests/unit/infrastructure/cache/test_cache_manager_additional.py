import pytest

from src.infrastructure.cache.core.cache_manager import (
    UnifiedCacheManager,
    ValidationError,
    create_memory_cache,
)
from src.infrastructure.cache.core.cache_configs import CacheConfig, BasicCacheConfig


def _make_strict_manager() -> UnifiedCacheManager:
    config = CacheConfig(
        basic=BasicCacheConfig(max_size=5, ttl=1),
        strict_validation=True,
    )
    return UnifiedCacheManager(config)


def test_shutdown_clears_state_and_stats():
    manager = create_memory_cache(max_size=4, ttl=10)
    manager.set("alpha", "value")
    assert manager.get("alpha") == "value"

    manager.shutdown()

    assert manager._is_shutdown is True
    assert manager._monitoring_enabled is False
    assert manager.size() == 0
    assert manager.get_stats() == {"hits": 0, "misses": 0}


def test_context_manager_shutdown_invoked():
    manager = create_memory_cache(max_size=2, ttl=5)
    with manager as cache:
        cache.set("key", "value")
        assert cache.get("key") == "value"

    assert manager._is_shutdown is True
    assert manager.get_stats() == {"hits": 0, "misses": 0}


def test_strict_validation_rejects_invalid_keys():
    manager = _make_strict_manager()

    with pytest.raises(ValidationError):
        manager.get("")

    with pytest.raises(ValidationError):
        manager.set("", "value")

    with pytest.raises(ValidationError):
        manager.delete(None)  # type: ignore[arg-type]


def test_unhashable_key_raises_type_error():
    manager = create_memory_cache()
    with pytest.raises(TypeError):
        manager.set({"a": 1}, "value")  # type: ignore[arg-type]


