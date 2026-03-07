import pytest

from src.infrastructure.cache.core.optimizer_components import (
    OptimizerComponent,
    OptimizerComponentFactory,
    create_optimizer_component_11,
    create_optimizer_component_17,
    create_optimizer_component_23,
)


def test_optimizer_component_info():
    component = OptimizerComponent(component_id=11)
    info = component.get_info()
    assert info["component_id"] == 11
    assert info["component_type"] == "Cache"
    assert component.get_processing_type() == "optimizer_processing"
    assert component.get_component_type_name() == "optimizer"


@pytest.mark.parametrize("component_id", [11, 17, 23])
def test_optimizer_component_factory_create(component_id):
    component = OptimizerComponentFactory.create_component(component_id)
    assert isinstance(component, OptimizerComponent)
    assert component.component_id == component_id


def test_optimizer_component_factory_invalid():
    with pytest.raises(ValueError):
        OptimizerComponentFactory.create_component(999)


def test_optimizer_component_factory_helpers():
    available = OptimizerComponentFactory.get_available_components()
    assert available == sorted([11, 17, 23])

    all_components = OptimizerComponentFactory.create_all_components()
    assert set(all_components.keys()) == set(available)

    info = OptimizerComponentFactory.get_component_info()
    assert info["total_components"] == len(available)

    assert isinstance(create_optimizer_component_11(), OptimizerComponent)
    assert isinstance(create_optimizer_component_17(), OptimizerComponent)
    assert isinstance(create_optimizer_component_23(), OptimizerComponent)

