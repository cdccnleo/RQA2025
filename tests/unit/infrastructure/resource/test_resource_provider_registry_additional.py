import pytest
from unittest.mock import MagicMock

from src.infrastructure.resource.core.resource_provider_registry import ResourceProviderRegistry
from src.infrastructure.resource.core.unified_resource_interfaces import IResourceProvider, ResourceInfo


class DummyProvider(IResourceProvider):
    def __init__(self, resource_type="cpu", resources=None):
        self._resource_type = resource_type
        self._resources = resources or []

    @property
    def resource_type(self) -> str:
        return self._resource_type

    def get_available_resources(self):
        return list(self._resources)

    def allocate_resource(self, request):
        raise NotImplementedError()

    def release_resource(self, allocation_id: str) -> bool:
        return False

    def get_resource_status(self, resource_id: str):
        return None

    def optimize_resources(self):
        return {}


@pytest.fixture
def event_bus():
    return MagicMock()


@pytest.fixture
def logger():
    return MagicMock()


@pytest.fixture
def error_handler():
    return MagicMock()


@pytest.fixture
def registry(event_bus, logger, error_handler):
    return ResourceProviderRegistry(event_bus=event_bus, logger=logger, error_handler=error_handler)


def test_unregister_provider_emits_event(registry, event_bus):
    provider = DummyProvider("memory")
    registry.register_provider(provider)

    assert registry.unregister_provider("memory") is True
    event = event_bus.publish.call_args[0][0]
    assert event.action == "provider_unregistered"
    assert event.resource_type == "memory"


def test_get_provider_status_with_capacity(registry):
    resources = [
        ResourceInfo(resource_id="res1", resource_type="cpu", name="res1", capacity={"total": 10}),
        ResourceInfo(resource_id="res2", resource_type="cpu", name="res2", capacity={"total": 5}),
    ]
    provider = DummyProvider("cpu", resources=resources)
    registry.register_provider(provider)

    status = registry.get_provider_status("cpu")
    assert status["available_count"] == 2
    assert status["total_capacity"] == 15


def test_get_provider_status_error_path(registry, error_handler):
    class BrokenProvider(DummyProvider):
        def get_available_resources(self):
            raise RuntimeError("boom")

    provider = BrokenProvider("gpu")
    registry.register_provider(provider)

    status = registry.get_provider_status("gpu")
    error_handler.handle_error.assert_called_once()
    assert status["status"] == "error"
    assert status["error"] == "boom"


def test_get_all_provider_status_collects_all(registry):
    registry.register_provider(DummyProvider("cpu"))
    registry.register_provider(DummyProvider("disk"))

    status = registry.get_all_provider_status()
    assert set(status.keys()) == {"cpu", "disk"}


def test_get_provider_info(registry):
    registry.register_provider(DummyProvider("network"))
    info = registry.get_provider_info("network")
    assert info["provider_type"] == DummyProvider.__name__


def test_get_provider_info_missing(registry):
    assert registry.get_provider_info("missing") is None


def test_update_provider_health_success(registry, logger):
    registry.register_provider(DummyProvider("cpu"))
    assert registry.update_provider_health("cpu", "healthy") is True
    assert any(
        args[0].startswith("提供者 cpu 健康状态更新为")
        for args, _ in logger.log_info.call_args_list
    )


def test_update_provider_health_missing(registry):
    assert registry.update_provider_health("ghost", "warning") is False


def test_clear_emits_bulk_events(registry, event_bus, logger):
    registry.register_provider(DummyProvider("a"))
    registry.register_provider(DummyProvider("b"))

    registry.clear()
    assert any(
        args[0].startswith("已清空所有资源提供者")
        for args, _ in logger.log_info.call_args_list
    )
    bulk_events = [
        args[0]
        for args, _ in event_bus.publish.call_args_list
        if args[0].action == "provider_bulk_unregistered"
    ]
    assert len(bulk_events) == 2

