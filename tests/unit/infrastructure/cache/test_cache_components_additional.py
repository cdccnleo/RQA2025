from datetime import datetime, timedelta

import pytest

from src.infrastructure.cache.core.cache_components import CacheComponent


@pytest.fixture
def component():
    return CacheComponent(component_id=101, component_type="memory", config={"region": "us-east"})


def test_component_initialization(component):
    assert component.component_name == "CacheComponent_101"
    assert component.component_type == "memory"
    assert component.health_check() is True
    status = component.get_component_status()
    assert status["status"] == "healthy"
    assert status["initialized"] is True
    assert status["cache_size"] == 0


def test_component_crud_operations(component):
    assert component.set_cache_item("key", "value") is True
    assert component.has_cache_item("key") is True
    assert component.get_cache_item("key") == "value"
    stats = component.get_cache_stats()
    assert stats["size"] == 1
    assert component.delete_cache_item("key") is True
    assert component.has_cache_item("key") is False
    assert component.clear_all_cache() is True


def test_component_info_and_shutdown(component):
    info = component.get_info()
    assert info["component_id"] == 101
    assert info["component_type"] == "memory"
    assert info["status"] == "healthy"
    assert info["config"]["region"] == "us-east"

    component.shutdown_component()
    assert component.get_component_status_string() == "stopped"
    assert component.get_cache_size() == 0

