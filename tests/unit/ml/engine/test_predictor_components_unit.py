import pytest

from src.ml.engine.predictor_components import (
    PredictorComponent,
    PredictorComponentFactory,
)


def test_predictor_component_info_and_status():
    component = PredictorComponent(predictor_id=7, component_type="Batch")
    info = component.get_info()
    status = component.get_status()

    assert info["component_type"] == "Batch"
    assert status["status"] == "ready"


def test_predictor_component_process_returns_payload():
    component = PredictorComponent(predictor_id=2)
    payload = {"features": [0.1, 0.2]}
    result = component.process(payload)

    assert result["input"] == payload
    assert result["predictor_id"] == 2


def test_factory_register_and_list():
    factory = PredictorComponentFactory()
    component = PredictorComponent(predictor_id=4)
    factory.register(component)

    assert factory.get_component(4) is component
    assert factory.list_components() == [4]


def test_factory_enforces_supported_ids():
    factory = PredictorComponentFactory(supported_ids=[1, 2])
    factory.create_component(1)

    with pytest.raises(ValueError):
        factory.create_component(5)

