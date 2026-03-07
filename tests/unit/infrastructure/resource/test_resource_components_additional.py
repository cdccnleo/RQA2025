import pytest

from src.infrastructure.resource.core.resource_components import ResourceComponentFactory, ResourceComponent


@pytest.fixture
def factory():
    return ResourceComponentFactory()


def test_create_component_with_resource_prefix(factory):
    component = factory.create_component("resource_13")
    assert isinstance(component, ResourceComponent)
    assert component.resource_id == 13


def test_create_component_with_numeric_string(factory):
    component = factory.create_component("7")
    assert component.resource_id == 7


def test_create_component_invalid_resource(factory):
    with pytest.raises(ValueError, match="不支持的resource ID: 2"):
        factory.create_component("resource_2")


def test_create_component_delegates_to_super(factory):
    factory.register_factory("custom_component", lambda config: {"created": True})
    result = factory.create_component("custom_component", {"foo": "bar"})
    assert result == {"created": True}

