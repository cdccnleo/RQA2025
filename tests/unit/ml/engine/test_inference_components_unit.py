import pytest

from src.ml.engine.inference_components import (
    InferenceComponent,
    InferenceComponentFactory,
)


def test_component_info_and_status():
    component = InferenceComponent(inference_id=9, component_type="Realtime")
    info = component.get_info()
    status = component.get_status()

    assert info["component_type"] == "Realtime"
    assert status["status"] == "idle"


def test_component_process_returns_payload():
    component = InferenceComponent(inference_id=3)
    payload = {"input": [1, 2, 3]}
    result = component.process(payload)

    assert result["input"] == payload
    assert result["inference_id"] == 3


def test_factory_register_and_list():
    factory = InferenceComponentFactory()
    component = InferenceComponent(inference_id=5)
    factory.register(component)

    assert factory.get_component(5) is component
    assert factory.list_components() == [5]


def test_factory_enforces_supported_ids():
    factory = InferenceComponentFactory(supported_ids=[1, 2])
    factory.create_component(1)

    with pytest.raises(ValueError):
        factory.create_component(5)

