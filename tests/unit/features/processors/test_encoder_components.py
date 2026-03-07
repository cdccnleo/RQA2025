import logging
from datetime import datetime

import pytest

from src.features.processors.encoder_components import (
    ComponentFactory,
    EncoderComponent,
    EncoderComponentFactory,
    create_encoder_encoder_component_5,
)


class DummyComponent:
    def __init__(self, should_init=True):
        self.should_init = should_init
        self.config = None

    def initialize(self, config):
        self.config = config
        return self.should_init


def test_component_factory_create_success(monkeypatch):
    factory = ComponentFactory()
    dummy = DummyComponent()
    monkeypatch.setattr(factory, "_create_component_instance", lambda comp, cfg: dummy)

    result = factory.create_component("dummy", {"alpha": 1})
    assert result is dummy
    assert dummy.config == {"alpha": 1}


def test_component_factory_create_failure(monkeypatch, caplog):
    factory = ComponentFactory()
    monkeypatch.setattr(factory, "_create_component_instance", lambda *args, **kwargs: DummyComponent(False))
    with caplog.at_level(logging.ERROR):
        result = factory.create_component("dummy", {})
    assert result is None


def test_component_factory_exception(monkeypatch):
    factory = ComponentFactory()

    def _raise(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(factory, "_create_component_instance", _raise)
    assert factory.create_component("dummy", {}) is None


def test_encoder_component_process_success():
    component = EncoderComponent(encoder_id=10, component_type="Test")
    data = {"payload": "value"}
    result = component.process(data)
    assert result["status"] == "success"
    assert result["component_name"] == component.component_name
    assert result["input_data"] == data


def test_encoder_component_process_failure(monkeypatch):
    component = EncoderComponent(encoder_id=20)

    class BoomDateTime:
        calls = 0

        @classmethod
        def now(cls):
            cls.calls += 1
            if cls.calls == 1:
                raise ValueError("no time")
            return datetime.now()

    monkeypatch.setattr("src.features.processors.encoder_components.datetime", BoomDateTime)
    result = component.process({"x": 1})
    assert result["status"] == "error"
    assert result["error"] == "no time"


def test_encoder_component_info_and_status():
    component = EncoderComponent(encoder_id=15)
    info = component.get_info()
    assert info["encoder_id"] == 15
    assert "component_name" in info
    # description 字符串保留模板占位符
    assert info["description"] == "统一{self.component_type}组件实现"

    status = component.get_status()
    assert status["status"] == "active"
    assert status["encoder_id"] == 15


@pytest.mark.parametrize("encoder_id", [5, 25, 75])
def test_encoder_factory_create_component(encoder_id):
    component = EncoderComponentFactory.create_component(encoder_id)
    assert isinstance(component, EncoderComponent)
    assert component.get_encoder_id() == encoder_id


def test_encoder_factory_invalid_id():
    with pytest.raises(ValueError):
        EncoderComponentFactory.create_component(3)


def test_encoder_factory_metadata():
    info = EncoderComponentFactory.get_factory_info()
    assert info["factory_name"] == "EncoderComponentFactory"
    assert len(info["supported_ids"]) == info["total_encoders"]


def test_encoder_factory_create_all():
    all_components = EncoderComponentFactory.create_all_encoders()
    assert len(all_components) == len(EncoderComponentFactory.SUPPORTED_ENCODER_IDS)
    assert 5 in all_components
    assert isinstance(all_components[5], EncoderComponent)


def test_legacy_creator_function():
    component = create_encoder_encoder_component_5()
    assert component.get_encoder_id() == 5

