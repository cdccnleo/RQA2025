import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock

from src.infrastructure.resource.core.resource_allocation_manager import ResourceAllocationManager
from src.infrastructure.resource.core.unified_resource_interfaces import ResourceAllocation


@pytest.fixture
def manager():
    provider_registry = MagicMock()
    provider_registry.has_provider.return_value = True
    provider_registry.get_provider.return_value = MagicMock()
    event_bus = MagicMock()
    logger = MagicMock()
    manager = ResourceAllocationManager(
        provider_registry=provider_registry,
        event_bus=event_bus,
        logger=logger,
        error_handler=MagicMock(),
    )
    return manager, provider_registry, event_bus, logger


def make_allocation(
    allocation_id="alloc-1",
    request_id="req_1_consumerA",
    resource_id="cpu_node1",
    include_resource_type=True,
    resource_type="cpu",
    allocated_at=None,
):
    allocation = ResourceAllocation(
        allocation_id=allocation_id,
        request_id=request_id,
        resource_id=resource_id,
        allocated_resources={},
    )
    if include_resource_type:
        setattr(allocation, "resource_type", resource_type)
    if allocated_at is not None:
        allocation.allocated_at = allocated_at
    return allocation


def test_get_resource_type_infers_from_resource_id(manager):
    manager_instance, *_ = manager
    allocation = make_allocation(include_resource_type=False, resource_id="cpu_primary")
    inferred = manager_instance._get_resource_type(allocation)
    assert inferred == "cpu"


def test_get_allocations_for_consumer_filters_by_suffix(manager):
    manager_instance, *_ = manager
    alloc_a = make_allocation(allocation_id="alloc-a", request_id="req_99_consumerA")
    alloc_b = make_allocation(allocation_id="alloc-b", request_id="req_77_consumerB")
    manager_instance._allocations["alloc-a"] = alloc_a
    manager_instance._allocations["alloc-b"] = alloc_b

    result = manager_instance.get_allocations_for_consumer("consumerA")
    assert result == [alloc_a]


def test_get_allocations_for_resource_type(manager):
    manager_instance, *_ = manager
    cpu_alloc = make_allocation(allocation_id="cpu-1", resource_type="cpu")
    mem_alloc = make_allocation(allocation_id="mem-1", resource_type="memory")
    manager_instance._allocations["cpu-1"] = cpu_alloc
    manager_instance._allocations["mem-1"] = mem_alloc

    result = manager_instance.get_allocations_for_resource_type("cpu")
    assert result == [cpu_alloc]


def test_get_active_allocations_returns_snapshot(manager):
    manager_instance, *_ = manager
    alloc = make_allocation()
    manager_instance._allocations["alloc-1"] = alloc

    active = manager_instance.get_active_allocations()
    assert active == [alloc]
    manager_instance._allocations.clear()
    # ensure returned snapshot not affected by later mutations
    assert active == [alloc]


def test_clear_expired_allocations_removes_old_entries(manager):
    manager_instance, _, _, logger = manager
    old_allocation = make_allocation(
        allocation_id="expired",
        allocated_at=datetime.now() - timedelta(hours=3),
    )
    fresh_allocation = make_allocation(
        allocation_id="fresh",
        allocated_at=datetime.now() - timedelta(minutes=10),
    )
    manager_instance._allocations["expired"] = old_allocation
    manager_instance._allocations["fresh"] = fresh_allocation

    removed = manager_instance.clear_expired_allocations(max_age_seconds=3600)
    assert removed == 1
    assert "expired" not in manager_instance._allocations
    logger.log_info.assert_called_once()


def test_force_release_allocation_emits_event(manager):
    manager_instance, _, event_bus, logger = manager
    allocation = make_allocation(allocation_id="force-1", resource_type="gpu", resource_id="gpu_01")
    manager_instance._allocations["force-1"] = allocation

    result = manager_instance.force_release_allocation("force-1")
    assert result is True
    assert "force-1" not in manager_instance._allocations
    event = event_bus.publish.call_args[0][0]
    assert event.action == "force_released"
    assert event.resource_type == "gpu"
    assert event.resource_id == "gpu_01"
    logger.log_warning.assert_called_once()


def test_clear_all_allocations_publishes_events(manager):
    manager_instance, _, event_bus, logger = manager
    manager_instance._allocations["a1"] = make_allocation(allocation_id="a1")
    manager_instance._allocations["a2"] = make_allocation(allocation_id="a2")

    manager_instance.clear_all_allocations()
    assert manager_instance._allocations == {}
    assert event_bus.publish.call_count == 2
    logger.log_info.assert_called_once()

