import pytest

from src.ml.ensemble.voting_components import VotingComponent, VotingComponentFactory


def test_voting_component_info_and_status():
    component = VotingComponent(voting_id=12, component_type="Majority")
    info = component.get_info()
    status = component.get_status()

    assert info["component_type"] == "Majority"
    assert status["status"] == "ready"


def test_voting_component_process_returns_payload():
    component = VotingComponent(voting_id=4)
    payload = {"votes": [0.6, 0.4]}
    result = component.process(payload)

    assert result["input"] == payload
    assert result["voting_id"] == 4


def test_factory_register_and_list():
    factory = VotingComponentFactory()
    component = VotingComponent(voting_id=7)
    factory.register(component)

    assert factory.get_component(7) is component
    assert factory.list_components() == [7]


def test_factory_enforces_supported_ids():
    factory = VotingComponentFactory(supported_ids=[1, 2])
    factory.create_component(1)

    with pytest.raises(ValueError):
        factory.create_component(6)

