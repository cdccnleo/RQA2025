import logging
from unittest.mock import MagicMock

import pytest

import src.infrastructure.monitoring.core.subscription_manager as sub_module
from src.infrastructure.monitoring.core.subscription_manager import SubscriptionManager


@pytest.fixture
def manager():
    # Patch MessageType to avoid importing heavy dependencies
    class DummyMessageType:
        ALERT = "ALERT"
        METRIC = "METRIC"

    sub_module.MessageType = DummyMessageType  # type: ignore
    return SubscriptionManager()


def _make_message(message_type="ALERT", payload=None, topic=None):
    class DummyMessage:
        def __init__(self):
            self.type = message_type
            self.payload = payload or {"value": 1}
            self.topic = topic

    return DummyMessage()


def test_subscribe_and_publish_success(manager):
    received = []

    def handler(msg):
        received.append(msg.payload["value"])

    assert manager.subscribe("ALERT", handler) is True
    message = _make_message("ALERT", {"value": 42})
    result = manager.publish(message)

    assert received == [42]
    assert result["delivered"] == 1
    assert manager.stats["messages_delivered"] == 1


def test_pattern_subscription(manager):
    received = []

    def pattern_handler(msg):
        received.append("pattern")

    assert manager.subscribe("METRIC.*", pattern_handler, pattern="METRIC.*") is True
    message = _make_message("METRIC.cpu", {"value": 10})
    result = manager.publish(message)

    assert received == ["pattern"]
    assert result["total_subscribers"] == 1


def test_topic_subscription(manager):
    received = []

    def topic_handler(msg):
        received.append("topic")

    assert manager.subscribe_topic("performance", topic_handler) is True
    message = _make_message("ALERT", {"value": 1}, topic="performance")
    result = manager.publish(message)

    assert received == ["topic"]
    assert result["delivered"] == 1


def test_unsubscribe_exact_and_pattern(manager):
    handler = MagicMock()
    manager.subscribe("ALERT", handler)
    manager.subscribe("ALERT.*", handler, pattern="ALERT.*")

    assert manager.unsubscribe("ALERT", handler) is True
    assert manager.unsubscribe("ALERT.*", handler, pattern="ALERT.*") is True

    message = _make_message("ALERT", {"value": 1})
    result = manager.publish(message)
    assert result["delivered"] == 0


def test_unsubscribe_topic(manager):
    handler = MagicMock()
    manager.subscribe_topic("performance", handler)

    assert manager.unsubscribe_topic("performance", handler) is True
    message = _make_message(topic="performance")
    result = manager.publish(message)
    assert result["delivered"] == 0


def test_publish_handler_failure(manager, caplog):
    def failing_handler(msg):
        raise RuntimeError("boom")

    manager.subscribe("ALERT", failing_handler)
    with caplog.at_level(logging.ERROR):
        result = manager.publish(_make_message("ALERT"))

    assert result["failed"] == 1
    assert manager.stats["delivery_failures"] == 1
    assert any("消息分发失败" in record.msg for record in caplog.records)


def test_get_subscription_info(manager):
    handler = MagicMock()
    manager.subscribe("ALERT", handler)
    info_specific = manager.get_subscription_info("ALERT")
    info_all = manager.get_subscription_info()

    assert info_specific["total"] >= 1
    assert info_all["total_subscriptions"] >= 1


def test_list_subscribers(manager):
    def handler(msg):
        pass

    manager.subscribe("ALERT.status", handler)
    manager.subscribe_topic("topic", handler)
    manager.subscribe("ALERT.*", handler, pattern="ALERT.*")

    subscribers = manager.list_subscribers("ALERT.status")
    assert any("exact" in item for item in subscribers)
    assert any("pattern:ALERT.*" in item for item in subscribers)


def test_clear_subscriptions(manager):
    handler = MagicMock()
    manager.subscribe("ALERT", handler)
    manager.subscribe_topic("topic", handler)

    manager.clear_subscriptions("ALERT")
    assert manager.publish(_make_message("ALERT"))["delivered"] == 0

    manager.clear_subscriptions()
    assert manager.stats["total_subscriptions"] == 0


def test_subscription_stats(manager):
    handler = MagicMock()
    manager.subscribe("ALERT", handler)
    stats = manager.get_subscription_stats()
    assert stats["total_subscriptions"] >= 1
    assert "delivery_success_rate" in stats


def test_health_status(manager):
    health = manager.get_health_status()
    assert health["status"] in {"healthy", "warning"}


def test_publish_exception(manager, monkeypatch):
    monkeypatch.setattr(manager, "subscriptions", None)

    result = manager.publish(_make_message("ALERT"))
    assert result["failed"] == 0
    assert result["errors"]

