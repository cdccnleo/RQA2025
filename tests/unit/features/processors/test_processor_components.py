import logging

import pytest

from src.features.processors.processor_components import (
    FeatureProcessorComponentFactory,
    ProcessorComponent,
)


@pytest.fixture(autouse=True)
def silence_logger(monkeypatch):
    monkeypatch.setattr(
        "src.features.processors.processor_components.logger",
        logging.getLogger(__name__),
    )


def test_processor_component_process_success():
    component = ProcessorComponent(processor_id=1, component_type="FeatureProcessor")
    data = {"value": 42}
    result = component.process(data)
    assert result["status"] == "success"
    assert result["input_data"] == data


def test_processor_component_get_status():
    component = ProcessorComponent(processor_id=6, component_type="FeatureProcessor")
    status = component.get_status()
    assert status["status"] == "active"
    assert status["processor_id"] == 6


def test_feature_processor_factory_create_component_valid():
    component = FeatureProcessorComponentFactory.create_component(1)
    assert component.get_processor_id() == 1
    assert component.component_type == "FeatureProcessor"


def test_feature_processor_factory_create_component_invalid():
    with pytest.raises(ValueError):
        FeatureProcessorComponentFactory.create_component(2)


def test_feature_processor_factory_create_all_processors():
    processors = FeatureProcessorComponentFactory.create_all_processors()
    assert set(processors.keys()) == set(FeatureProcessorComponentFactory.SUPPORTED_PROCESSOR_IDS)

