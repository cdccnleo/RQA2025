from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
import threading

import pytest

import src.infrastructure.resource.core.unified_resource_manager as manager_module
from src.infrastructure.resource.core.unified_resource_manager import UnifiedResourceManager


class DummyLogger:
    def __init__(self):
        self.infos: List[str] = []
        self.warnings: List[str] = []

    def log_info(self, message: str, **kwargs):
        self.infos.append(message)

    def log_warning(self, message: str, **kwargs):
        self.warnings.append(message)

    def warning(self, message: str, **kwargs):
        self.log_warning(message, **kwargs)


class DummyErrorHandler:
    def __init__(self):
        self.calls: List[Dict[str, Any]] = []

    def handle_error(self, error: Exception, context: Dict[str, Any] | None = None):
        self.calls.append({"error": error, "context": context})


class DummyEventBus:
    def __init__(self, logger=None):
        self.logger = logger
        self.started = False
        self.stopped = False
        self.published: List[Any] = []

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def publish(self, event):
        self.published.append(event)


class DummyDependencyContainer:
    def __init__(self, logger):
        self.logger = logger


@dataclass
class DummyProvider:
    resource_type: str


class DummyProviderRegistry:
    _shared_providers: Dict[str, DummyProvider] = {}

    def __init__(self, event_bus, logger, error_handler):
        self.event_bus = event_bus
        self.logger = logger
        self.error_handler = error_handler
        self._providers = self.__class__._shared_providers

    def register_provider(self, provider: DummyProvider) -> bool:
        self._providers[provider.resource_type] = provider
        return True

    def unregister_provider(self, resource_type: str) -> bool:
        return self._providers.pop(resource_type, None) is not None

    def get_providers(self) -> List[DummyProvider]:
        return list(self._providers.values())


class DummyConsumer:
    def __init__(self, consumer_id: str):
        self.consumer_id = consumer_id


class DummyConsumerRegistry:
    _shared_consumers: Dict[str, DummyConsumer] = {}

    def __init__(self, logger, error_handler):
        self.logger = logger
        self.error_handler = error_handler
        self._consumers = self.__class__._shared_consumers

    def register_consumer(self, consumer: DummyConsumer) -> bool:
        self._consumers[consumer.consumer_id] = consumer
        return True

    def unregister_consumer(self, consumer_id: str) -> bool:
        return self._consumers.pop(consumer_id, None) is not None

    def get_consumers(self) -> List[DummyConsumer]:
        return list(self._consumers.values())


class DummyAllocationManager:
    def __init__(self, provider_registry, event_bus, logger, error_handler):
        self.provider_registry = provider_registry
        self.event_bus = event_bus
        self.logger = logger
        self.error_handler = error_handler
        self._allocations: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    def request_resource(self, consumer_id: str, resource_type: str, requirements: Dict[str, Any], priority: int = 1) -> str:
        allocation_id = f"{consumer_id}:{resource_type}:{len(self._allocations)}"
        self._allocations[allocation_id] = {
            "consumer": consumer_id,
            "resource_type": resource_type,
            "requirements": requirements,
            "priority": priority,
        }
        return allocation_id

    def release_resource(self, allocation_id: str) -> bool:
        return self._allocations.pop(allocation_id, None) is not None


class DummyStatusReporter:
    def __init__(self, provider_registry, consumer_registry, allocation_manager, logger, error_handler):
        self.provider_registry = provider_registry
        self.consumer_registry = consumer_registry
        self.allocation_manager = allocation_manager
        self.logger = logger
        self.error_handler = error_handler
        self.raise_on_status = False

    def get_resource_status(self) -> Dict[str, Any]:
        if self.raise_on_status:
            raise RuntimeError("status error")
        return {
            "summary": {
                "providers_count": len(self.provider_registry.get_providers()),
                "consumers_count": len(self.consumer_registry.get_consumers()),
                "active_allocations": len(self.allocation_manager._allocations),
            },
            "health": "optimal",
        }

    def get_detailed_report(self) -> Dict[str, Any]:
        return {"health": "optimal", "details": {}}


@pytest.fixture
def manager(monkeypatch):
    DummyProviderRegistry._shared_providers = {}
    DummyConsumerRegistry._shared_consumers = {}

    logger = DummyLogger()
    error_handler = DummyErrorHandler()

    monkeypatch.setattr(manager_module, "StandardLogger", lambda name=None: logger)
    monkeypatch.setattr(manager_module, "BaseErrorHandler", lambda: error_handler)
    monkeypatch.setattr(manager_module, "EventBus", DummyEventBus)
    monkeypatch.setattr(manager_module, "create_system_event", lambda **kwargs: {**kwargs})
    monkeypatch.setattr(manager_module, "DependencyContainer", DummyDependencyContainer)
    monkeypatch.setattr(manager_module, "ResourceProviderRegistry", DummyProviderRegistry)
    monkeypatch.setattr(manager_module, "ResourceConsumerRegistry", DummyConsumerRegistry)
    monkeypatch.setattr(manager_module, "ResourceAllocationManager", DummyAllocationManager)
    monkeypatch.setattr(manager_module, "ResourceStatusReporter", DummyStatusReporter)

    mgr = UnifiedResourceManager()
    return mgr, logger, error_handler


def test_lazy_components_and_registry_flow(manager):
    mgr, logger, _ = manager

    provider_registry = mgr.provider_registry
    consumer_registry = mgr.consumer_registry
    allocation_manager = mgr.allocation_manager

    mgr.register_provider(DummyProvider("cpu"))
    mgr.register_consumer(DummyConsumer("c1"))

    status_reporter = mgr.status_reporter
    assert status_reporter is mgr.status_reporter

    assert provider_registry is mgr.provider_registry
    assert consumer_registry is mgr.consumer_registry
    assert allocation_manager is mgr.allocation_manager
    assert status_reporter is mgr.status_reporter

    allocation_id = mgr.request_resource("c1", "cpu", {"cores": 4})
    assert allocation_id in allocation_manager._allocations
    assert mgr.release_resource(allocation_id) is True

    status = mgr.get_system_status()
    assert status == {
        "providers_count": 1,
        "consumers_count": 1,
        "allocations_count": 0,
        "system_health": "optimal",
    }
    assert "统一资源管理器已初始化" in logger.infos[0]


def test_get_system_status_failure_fallback(manager):
    mgr, logger, _ = manager
    mgr.status_reporter.raise_on_status = True

    status = mgr.get_system_status()

    assert status == {
        "providers_count": 0,
        "consumers_count": 0,
        "allocations_count": 0,
        "system_health": "error",
    }
    assert any("获取系统状态失败" in msg for msg in logger.warnings)


def test_cleanup_and_shutdown(manager):
    mgr, logger, _ = manager
    allocation_id = mgr.request_resource("c1", "cpu", {})
    assert allocation_id in mgr.allocation_manager._allocations

    mgr.shutdown()

    assert any("正在关闭统一资源管理器" in msg for msg in logger.infos)
    assert any("统一资源管理器已关闭" in msg for msg in logger.infos)

    mgr._cleanup_allocations()
    assert any("清理" in msg for msg in logger.infos)


def test_event_bus_creation_warnings(monkeypatch, manager):
    mgr, logger, _ = manager

    monkeypatch.setattr(manager_module, "EventBus", None)
    assert mgr._create_event_bus() is None
    assert any("EventBus组件不可用" in msg for msg in logger.warnings)

    class FailingEventBus:
        def __init__(self, *_args, **_kwargs):
            raise ValueError("broken")

    monkeypatch.setattr(manager_module, "EventBus", FailingEventBus)
    assert mgr._create_event_bus() is None
    assert any("组件创建失败" in msg for msg in logger.warnings)


def test_dependency_container_warning(monkeypatch, manager):
    mgr, logger, _ = manager
    monkeypatch.setattr(manager_module, "DependencyContainer", None)
    assert mgr._create_container() is None
    assert any("DependencyContainer组件不可用" in msg for msg in logger.warnings)


def test_optimize_resources_and_accessors(manager):
    mgr, _, _ = manager
    result = mgr.optimize_resources()
    assert result["success"] is True
    assert isinstance(mgr.get_event_bus(), DummyEventBus)
    assert isinstance(mgr.get_container(), DummyDependencyContainer)


def test_stop_invokes_shutdown(manager, monkeypatch):
    mgr, logger, _ = manager
    shutdown_called = False

    def fake_shutdown():
        nonlocal shutdown_called
        shutdown_called = True

    monkeypatch.setattr(mgr, "shutdown", fake_shutdown)
    mgr.stop()
    assert shutdown_called is True
    assert "统一资源管理器停止" in logger.infos[-1]
