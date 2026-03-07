import pytest

from src.infrastructure.error.core.container import (
    DependencyContainer,
    Lifecycle,
)


class ServiceA:
    def __init__(self):
        self.value = 42


class ServiceB:
    def __init__(self, service_a: ServiceA):
        self.service_a = service_a


class ServiceC:
    def __init__(self, optional: int = 7):
        self.optional = optional


class ServiceWithMissingDep:
    def __init__(self, missing):
        self.missing = missing


def test_dependency_container_singleton_and_transient():
    container = DependencyContainer()
    container.register_singleton(ServiceA)
    container.register_transient(ServiceB)

    a1 = container.resolve(ServiceA)
    a2 = container.resolve(ServiceA)
    assert a1 is a2

    b1 = container.resolve(ServiceB)
    b2 = container.resolve(ServiceB)
    assert b1 is not b2
    assert b1.service_a is a1


def test_dependency_container_scoped_instances():
    container = DependencyContainer()
    container.register_scoped(ServiceA)

    with pytest.raises(RuntimeError):
        container.resolve(ServiceA)

    with container.scope() as scope1:
        a1 = scope1.resolve(ServiceA)
        a2 = scope1.resolve(ServiceA)
        assert a1 is a2

    with container.scope() as scope2:
        a3 = scope2.resolve(ServiceA)
        assert a3 is not a1


def test_dependency_container_unannotated_parameter_error():
    container = DependencyContainer()
    container.register(ServiceWithMissingDep)

    with pytest.raises(Exception) as exc:
        container.resolve(ServiceWithMissingDep)
    assert "Parameter missing in ServiceWithMissingDep" in str(exc.value)


def test_dependency_container_defaults_and_clear():
    container = DependencyContainer()
    container.register(ServiceC)

    service = container.resolve(ServiceC)
    assert service.optional == 7

    assert container.has_service(ServiceC) is True
    assert ServiceC in container.get_registered_services()

    container.clear()
    assert container.has_service(ServiceC) is False

