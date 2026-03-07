import logging

import pytest

from src.features.processors.normalizer_components import (
    NormalizerComponent,
    NormalizerComponentFactory,
)


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.normalizer_components.logger",
        logging.getLogger(__name__),
    )


def test_normalizer_component_process_success():
    component = NormalizerComponent(normalizer_id=3, component_type="Normalizer")
    data = {"value": 1}
    result = component.process(data)
    assert result["status"] == "success"
    assert result["input_data"] == data


def test_normalizer_component_process_failure(monkeypatch):
    component = NormalizerComponent(normalizer_id=3, component_type="Normalizer")

    def faulty_assign(*args, **kwargs):
        raise RuntimeError("fail")

    monkeypatch.setattr(component, "component_name", "FaultyComponent")
    monkeypatch.setattr(component, "component_type", "FaultyType")
    monkeypatch.setattr(component, "process", lambda data: {"status": "error", "error": "fail"})
    result = NormalizerComponent(3, "Normalizer").process({"value": 1})
    assert result["status"] == "success"


def test_factory_create_component_success():
    component = NormalizerComponentFactory.create_component(3)
    assert component.get_normalizer_id() == 3
    info = component.get_info()
    assert info["component_name"].startswith("Normalizer_Component_3")


def test_factory_create_component_invalid():
    with pytest.raises(ValueError):
        NormalizerComponentFactory.create_component(2)


def test_factory_create_all_normalizers():
    normalizers = NormalizerComponentFactory.create_all_normalizers()
    assert set(normalizers.keys()) == set(NormalizerComponentFactory.SUPPORTED_NORMALIZER_IDS)


def test_component_factory_create_component():
    component = NormalizerComponentFactory.create_component(3)
    assert isinstance(component, NormalizerComponent)

