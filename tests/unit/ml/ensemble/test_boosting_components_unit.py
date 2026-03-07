import pytest

from src.ml.ensemble.boosting_components import BoostingComponent, BoostingComponentFactory


def test_boosting_component_info_and_status():
    component = BoostingComponent(boosting_id=7, component_type="Gradient")
    info = component.get_info()
    status = component.get_status()

    assert info["component_type"] == "Gradient"
    assert status["status"] == "ready"


def test_boosting_component_process_returns_payload():
    component = BoostingComponent(boosting_id=3)
    payload = {"rounds": 10}
    result = component.process(payload)

    assert result["input"] == payload
    assert result["boosting_id"] == 3


def test_factory_register_and_list():
    factory = BoostingComponentFactory()
    component = BoostingComponent(boosting_id=5)
    factory.register(component)

    assert factory.get_component(5) is component
    assert factory.list_components() == [5]


def test_factory_enforces_supported_ids():
    factory = BoostingComponentFactory(supported_ids=[1, 2])
    factory.create_component(1)

    with pytest.raises(ValueError):
        factory.create_component(4)

