import inspect
import pytest
from unittest.mock import MagicMock

from src.infrastructure.resource.core.dependency_container import (
    DependencyContainer,
    ServiceLifetime,
    ServiceRegistrationError,
    ServiceNotFoundError,
    CircularDependencyError,
)


class SimpleService:
    def __init__(self):
        self.value = object()


class DependentService:
    def __init__(self, simple: SimpleService):
        self.simple = simple


class FirstCircularService:
    def __init__(self, second):
        self.second = second


class SecondCircularService:
    def __init__(self, first):
        self.first = first


FirstCircularService.__init__.__annotations__ = {
    "second": SecondCircularService,
    "return": None,
}
SecondCircularService.__init__.__annotations__ = {
    "first": FirstCircularService,
    "return": None,
}

first_sig = inspect.signature(FirstCircularService.__init__)
first_parameters = [
    param.replace(annotation=SecondCircularService) if name == "second" else param
    for name, param in first_sig.parameters.items()
]
FirstCircularService.__init__.__signature__ = first_sig.replace(parameters=first_parameters)

second_sig = inspect.signature(SecondCircularService.__init__)
second_parameters = [
    param.replace(annotation=FirstCircularService) if name == "first" else param
    for name, param in second_sig.parameters.items()
]
SecondCircularService.__init__.__signature__ = second_sig.replace(parameters=second_parameters)


def test_singleton_registration_and_resolution():
    container = DependencyContainer()
    container.register(SimpleService)

    first = container.resolve(SimpleService)
    second = container.resolve(SimpleService)

    assert first is second
    assert isinstance(first, SimpleService)


def test_transient_lifetime_creates_new_instances():
    container = DependencyContainer()
    container.register(SimpleService, lifetime=ServiceLifetime.TRANSIENT)

    first = container.resolve(SimpleService)
    second = container.resolve(SimpleService)

    assert first is not second


def test_scoped_lifetime_behaviour():
    container = DependencyContainer()
    container.register(SimpleService, lifetime=ServiceLifetime.SCOPED)

    with pytest.raises(Exception):
        container.resolve(SimpleService)

    with container.begin_scope() as first_scope:
        scoped_a = container.resolve(SimpleService)
        scoped_b = container.resolve(SimpleService)
        assert scoped_a is scoped_b
        assert first_scope

    with container.begin_scope():
        new_scoped = container.resolve(SimpleService)
        assert new_scoped is not scoped_a


def test_register_instance_returns_same_reference():
    service = SimpleService()
    container = DependencyContainer()

    container.register_instance(SimpleService, service)
    assert container.resolve(SimpleService) is service


def test_register_factory_supports_custom_constructor():
    container = DependencyContainer()
    factory = MagicMock(return_value=SimpleService())

    container.register_factory(SimpleService, factory)
    instance = container.resolve(SimpleService)

    assert isinstance(instance, SimpleService)
    factory.assert_called_once()


def test_constructor_injection_resolves_dependencies():
    container = DependencyContainer()
    container.register(SimpleService)
    container.register(DependentService)

    dependent = container.resolve(DependentService)

    assert isinstance(dependent.simple, SimpleService)
    assert dependent.simple is container.resolve(SimpleService)


def test_duplicate_registration_raises_error():
    container = DependencyContainer()
    container.register(SimpleService)

    with pytest.raises(ServiceRegistrationError):
        container.register(SimpleService)


def test_resolving_unknown_service_raises_error():
    container = DependencyContainer()

    with pytest.raises(ServiceNotFoundError):
        container.resolve(SimpleService)


def test_circular_dependency_detection():
    container = DependencyContainer()
    container.register(FirstCircularService)
    container.register(SecondCircularService)

    with pytest.raises(CircularDependencyError):
        container._resolve(FirstCircularService, {FirstCircularService})


def test_unregister_and_clear():
    container = DependencyContainer()
    container.register(SimpleService)

    assert container.unregister(SimpleService) is True
    assert container.is_registered(SimpleService) is False

    container.register(SimpleService)
    container.clear()

    assert len(container) == 0
    assert container.get_registered_services() == []


def test_get_service_info_includes_basic_metadata():
    container = DependencyContainer()
    container.register(SimpleService, lifetime=ServiceLifetime.TRANSIENT)

    info = container.get_service_info(SimpleService)

    assert info == {
        "service_type": "SimpleService",
        "implementation_type": "SimpleService",
        "lifetime": ServiceLifetime.TRANSIENT,
        "has_instance": False,
        "has_factory": False,
    }

    assert container.get_service_info(DependentService) is None


