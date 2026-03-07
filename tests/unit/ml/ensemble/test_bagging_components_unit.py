import pytest

from src.ml.ensemble.bagging_components import BaggingComponent, BaggingComponentFactory


def test_bagging_component_info_and_status():
    component = BaggingComponent(bagging_id=9, component_type="Bootstrap")
    info = component.get_info()
    status = component.get_status()

    assert info["component_type"] == "Bootstrap"
    assert status["status"] == "ready"


def test_bagging_component_process_returns_payload():
    component = BaggingComponent(bagging_id=2)
    payload = {"samples": [1, 2, 3]}
    result = component.process(payload)

    assert result["input"] == payload
    assert result["bagging_id"] == 2


def test_factory_register_and_list():
    factory = BaggingComponentFactory()
    component = BaggingComponent(bagging_id=4)
    factory.register(component)

    assert factory.get_component(4) is component
    assert factory.list_components() == [4]


def test_factory_enforces_supported_ids():
    factory = BaggingComponentFactory(supported_ids=[1, 2])
    factory.create_component(1)

    with pytest.raises(ValueError):
        factory.create_component(5)

