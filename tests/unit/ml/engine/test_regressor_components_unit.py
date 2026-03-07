import pytest

from src.ml.engine.regressor_components import (
    RegressorComponent,
    RegressorComponentFactory,
)


def test_regressor_component_info_and_status():
    component = RegressorComponent(regressor_id=6, component_type="Linear")
    info = component.get_info()
    status = component.get_status()

    assert info["component_type"] == "Linear"
    assert status["status"] == "ready"


def test_regressor_component_process_returns_payload():
    component = RegressorComponent(regressor_id=1)
    payload = {"target": 0.5}
    result = component.process(payload)

    assert result["input"] == payload
    assert result["regressor_id"] == 1


def test_factory_register_and_list():
    factory = RegressorComponentFactory()
    component = RegressorComponent(regressor_id=3)
    factory.register(component)

    assert factory.get_component(3) is component
    assert factory.list_components() == [3]


def test_factory_enforces_supported_ids():
    factory = RegressorComponentFactory(supported_ids=[1, 2])
    factory.create_component(1)

    with pytest.raises(ValueError):
        factory.create_component(4)

