import logging
from datetime import datetime

import pytest

from src.features.processors.scaler_components import (
    ComponentFactory,
    ScalerComponent,
    ScalerComponentFactory,
    create_scaler_scaler_component_4,
)


class DummyComponent:
    def __init__(self, should_init=True):
        self.should_init = should_init
        self.initialized_with = None

    def initialize(self, config):
        self.initialized_with = config
        return self.should_init


def test_component_factory_create_success(monkeypatch):
    factory = ComponentFactory()
    dummy = DummyComponent()
    monkeypatch.setattr(factory, "_create_component_instance", lambda comp, cfg: dummy)
    result = factory.create_component("scaler", {"scale": 2})
    assert result is dummy
    assert dummy.initialized_with == {"scale": 2}


def test_component_factory_create_failure(monkeypatch):
    factory = ComponentFactory()
    monkeypatch.setattr(factory, "_create_component_instance", lambda comp, cfg: DummyComponent(False))
    assert factory.create_component("scaler", {}) is None


def test_component_factory_exception(monkeypatch, caplog):
    factory = ComponentFactory()

    def _boom(*args, **kwargs):
        raise RuntimeError("explode")

    monkeypatch.setattr(factory, "_create_component_instance", _boom)
    with caplog.at_level(logging.ERROR):
        assert factory.create_component("scaler", {}) is None
    assert "创建组件失败" in caplog.text


def test_scaler_component_process_success():
    component = ScalerComponent(14, component_type="Scaler")
    result = component.process({"value": 10})
    assert result["status"] == "success"
    assert result["scaler_id"] == 14


def test_scaler_component_process_failure(monkeypatch):
    component = ScalerComponent(24)

    class BoomDatetime:
        calls = 0

        @classmethod
        def now(cls):
            cls.calls += 1
            if cls.calls == 1:
                raise ValueError("bad time")
            return datetime.now()

    monkeypatch.setattr("src.features.processors.scaler_components.datetime", BoomDatetime)
    result = component.process({"value": 10})
    assert result["status"] == "error"
    assert result["error"] == "bad time"


def test_scaler_component_info_and_status():
    component = ScalerComponent(34)
    info = component.get_info()
    assert info["scaler_id"] == 34
    assert info["description"] == "统一{self.component_type}组件实现"

    status = component.get_status()
    assert status["status"] == "active"
    assert status["scaler_id"] == 34


@pytest.mark.parametrize("scaler_id", [4, 44, 79])
def test_scaler_factory_create_component(scaler_id):
    component = ScalerComponentFactory.create_component(scaler_id)
    assert component.get_scaler_id() == scaler_id


def test_scaler_factory_invalid_id():
    with pytest.raises(ValueError):
        ScalerComponentFactory.create_component(5)


def test_scaler_factory_metadata():
    info = ScalerComponentFactory.get_factory_info()
    assert info["factory_name"] == "ScalerComponentFactory"
    assert len(info["supported_ids"]) == info["total_scalers"]


def test_scaler_factory_create_all():
    all_scalers = ScalerComponentFactory.create_all_scalers()
    assert len(all_scalers) == len(ScalerComponentFactory.SUPPORTED_SCALER_IDS)
    assert isinstance(all_scalers[4], ScalerComponent)


def test_legacy_creator_wrapper():
    component = create_scaler_scaler_component_4()
    assert component.get_scaler_id() == 4

