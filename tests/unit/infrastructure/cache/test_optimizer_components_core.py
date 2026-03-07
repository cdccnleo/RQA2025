from typing import Dict

import pytest

from src.infrastructure.cache.core.optimizer_components import (
    OptimizerComponent,
    OptimizerComponentFactory,
    create_optimizer_component_11,
    create_optimizer_component_17,
    create_optimizer_component_23,
)


SUPPORTED_IDS = OptimizerComponentFactory.SUPPORTED_COMPONENT_IDS


def test_optimizer_component_exposes_basic_metadata():
    component = OptimizerComponent(component_id=SUPPORTED_IDS[0])

    info = component.get_info()
    assert info["component_id"] == SUPPORTED_IDS[0]
    assert info["component_type"] == "Cache"
    assert info["type"] == "unified_optimizer_component"
    assert info["initialized"] is False
    assert component.get_processing_type() == "optimizer_processing"
    assert component.get_component_type_name() == "optimizer"


def test_optimizer_component_status_flow():
    component = OptimizerComponent(component_id=SUPPORTED_IDS[1])

    assert component.get_status()["status"] == "stopped"
    component.initialize({"threshold": 0.75})
    status = component.get_status()
    assert status["status"] == "running"
    assert status["config"]["threshold"] == 0.75

    component.shutdown()
    assert component.get_status()["status"] == "stopped"


def test_optimizer_component_factory_create_and_list():
    factory = OptimizerComponentFactory()

    available = factory.get_available_components()
    assert available == sorted(SUPPORTED_IDS)

    created = factory.create_all_components()
    assert set(created.keys()) == set(SUPPORTED_IDS)
    assert all(isinstance(instance, OptimizerComponent) for instance in created.values())

    with pytest.raises(ValueError):
        factory.create_component(999)


def test_optimizer_component_factory_info_payload():
    info = OptimizerComponentFactory.get_component_info()
    assert info["factory_name"] == "OptimizerComponentFactory"
    assert info["total_components"] == len(SUPPORTED_IDS)
    assert info["supported_ids"] == sorted(SUPPORTED_IDS)
    assert "description" in info and "created_at" in info


@pytest.mark.parametrize(
    "factory_func,expected_id",
    [
        (create_optimizer_component_11, 11),
        (create_optimizer_component_17, 17),
        (create_optimizer_component_23, 23),
    ],
)
def test_optimizer_component_helper_creators(factory_func, expected_id):
    component = factory_func()
    assert isinstance(component, OptimizerComponent)
    assert component.component_id == expected_id


