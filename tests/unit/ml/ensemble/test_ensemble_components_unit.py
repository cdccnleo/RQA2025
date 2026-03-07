import pytest

from src.ml.ensemble.ensemble_components import (
    EnsembleComponent,
    EnsembleComponentFactory,
)


def test_ensemble_component_info_and_status():
    component = EnsembleComponent(ensemble_id=11, component_type="Stacking")
    info = component.get_info()
    status = component.get_status()

    assert info["component_type"] == "Stacking"
    assert status["status"] == "ready"


def test_ensemble_component_process_returns_payload():
    component = EnsembleComponent(ensemble_id=3)
    payload = {"models": ["a", "b"]}
    result = component.process(payload)

    assert result["input"] == payload
    assert result["ensemble_id"] == 3


def test_factory_register_and_list():
    factory = EnsembleComponentFactory()
    component = EnsembleComponent(ensemble_id=5)
    factory.register(component)

    assert factory.get_component(5) is component
    assert factory.list_components() == [5]


def test_factory_enforces_supported_ids():
    factory = EnsembleComponentFactory(supported_ids=[1, 2])
    factory.create_component(1)

    with pytest.raises(ValueError):
        factory.create_component(4)

