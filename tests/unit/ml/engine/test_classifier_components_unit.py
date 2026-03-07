import pytest

from src.ml.engine.classifier_components import (
    ClassifierComponent,
    ClassifierComponentFactory,
)


def test_classifier_component_info_and_status():
    component = ClassifierComponent(classifier_id=4, component_type="Binary")
    info = component.get_info()
    status = component.get_status()

    assert info["component_type"] == "Binary"
    assert status["status"] == "ready"


def test_classifier_component_process_returns_payload():
    component = ClassifierComponent(classifier_id=1)
    payload = {"sample": [0.1, 0.9]}
    result = component.process(payload)

    assert result["input"] == payload
    assert result["classifier_id"] == 1


def test_factory_register_and_list():
    factory = ClassifierComponentFactory()
    component = ClassifierComponent(classifier_id=2)
    factory.register(component)

    assert factory.get_component(2) is component
    assert factory.list_components() == [2]


def test_factory_enforces_supported_ids():
    factory = ClassifierComponentFactory(supported_ids=[1, 2])
    factory.create_component(1)

    with pytest.raises(ValueError):
        factory.create_component(5)

