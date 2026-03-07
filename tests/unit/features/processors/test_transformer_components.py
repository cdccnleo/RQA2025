from datetime import datetime
from types import SimpleNamespace

import pytest

import src.features.processors.transformer_components as transformer_module


class FixedDatetime(datetime):
    """提供可控的 datetime.now() 返回值"""

    _now = datetime(2024, 1, 1, 9, 30, 0)

    @classmethod
    def now(cls):
        return cls._now


def test_component_factory_returns_initialized_component(monkeypatch):
    factory = transformer_module.ComponentFactory()
    component = SimpleNamespace(initialize=lambda cfg: True)
    monkeypatch.setattr(factory, "_create_component_instance", lambda ctype, cfg: component)
    result = factory.create_component("demo", {"foo": "bar"})
    assert result is component


def test_component_factory_handles_creation_failure(monkeypatch):
    factory = transformer_module.ComponentFactory()

    def raising(*_args, **_kwargs):
        raise RuntimeError("creation error")

    monkeypatch.setattr(factory, "_create_component_instance", raising)
    assert factory.create_component("broken", {}) is None


def test_transformer_component_info_and_status(monkeypatch):
    monkeypatch.setattr(transformer_module, "datetime", FixedDatetime)
    component = transformer_module.TransformerComponent(42, "Custom")
    info = component.get_info()
    assert info["transformer_id"] == 42
    assert info["component_type"] == "Custom"
    assert info["component_name"] == "Custom_Component_42"
    assert info["creation_time"] == FixedDatetime._now.isoformat()

    status = component.get_status()
    assert status["status"] == "active"
    assert status["transformer_id"] == 42
    assert status["creation_time"] == FixedDatetime._now.isoformat()


def test_transformer_component_process_success(monkeypatch):
    class ProcessDatetime(FixedDatetime):
        _now = datetime(2024, 1, 2, 15, 45, 0)

    monkeypatch.setattr(transformer_module, "datetime", ProcessDatetime)
    component = transformer_module.TransformerComponent(7, "Transformer")
    result = component.process({"value": 100})
    assert result["status"] == "success"
    assert result["transformer_id"] == 7
    assert result["input_data"] == {"value": 100}
    assert "processed_at" in result


def test_transformer_component_process_error(monkeypatch):
    monkeypatch.setattr(transformer_module, "datetime", FixedDatetime)
    component = transformer_module.TransformerComponent(12, "Transformer")

    class FaultyDatetime(FixedDatetime):
        calls = 0

        @classmethod
        def now(cls):
            cls.calls += 1
            if cls.calls == 1:
                raise RuntimeError("time error")
            return FixedDatetime._now

    monkeypatch.setattr(transformer_module, "datetime", FaultyDatetime)
    outcome = component.process({"bad": True})
    assert outcome["status"] == "error"
    assert outcome["error_type"] == "RuntimeError"


def test_transformer_factory_supported_ids_and_creation():
    factory_info = transformer_module.TransformerComponentFactory.get_factory_info()
    supported = transformer_module.TransformerComponentFactory.get_available_transformers()
    assert factory_info["total_transformers"] == len(supported)
    assert supported == sorted(supported)

    component = transformer_module.TransformerComponentFactory.create_component(supported[0])
    assert isinstance(component, transformer_module.TransformerComponent)


def test_transformer_factory_rejects_unknown_id():
    with pytest.raises(ValueError):
        transformer_module.TransformerComponentFactory.create_component(-1)


def test_transformer_factory_create_all_transformers():
    components = transformer_module.TransformerComponentFactory.create_all_transformers()
    supported = transformer_module.TransformerComponentFactory.SUPPORTED_TRANSFORMER_IDS
    assert set(components.keys()) == set(supported)
    assert all(isinstance(comp, transformer_module.TransformerComponent) for comp in components.values())


def test_backward_compatible_factories():
    component = transformer_module.create_transformer_transformer_component_2()
    assert component.get_transformer_id() == 2

