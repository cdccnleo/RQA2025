import pytest

from src.ml.ensemble.stacking_components import StackingComponent, StackingComponentFactory


def test_stacking_component_info_and_status():
    component = StackingComponent(stacking_id=6, component_type="Hybrid")
    info = component.get_info()
    status = component.get_status()

    assert info["component_type"] == "Hybrid"
    assert status["status"] == "ready"


def test_stacking_component_process_returns_payload():
    component = StackingComponent(stacking_id=2)
    payload = {"layers": ["base1", "base2"]}
    result = component.process(payload)

    assert result["input"] == payload
    assert result["stacking_id"] == 2


def test_factory_register_and_list():
    factory = StackingComponentFactory()
    component = StackingComponent(stacking_id=4)
    factory.register(component)

    assert factory.get_component(4) is component
    assert factory.list_components() == [4]


def test_factory_enforces_supported_ids():
    factory = StackingComponentFactory(supported_ids=[1, 2])
    factory.create_component(1)

    with pytest.raises(ValueError):
        factory.create_component(5)

