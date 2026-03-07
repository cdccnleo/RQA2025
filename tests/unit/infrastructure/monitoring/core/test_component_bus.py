import time
import uuid
import threading
from types import SimpleNamespace
from datetime import datetime, timedelta

import pytest

from src.infrastructure.monitoring.core.component_bus import (
    ComponentBus,
    Message,
    MessagePriority,
    MessageType,
    publish_event,
    send_notification,
)


@pytest.fixture
def sync_bus():
    bus = ComponentBus(enable_async=False, max_queue_size=2)

    # patch publish to process queue immediately for deterministic tests
    original_publish = ComponentBus.publish

    def publish_and_process(message: Message) -> bool:
        result = original_publish(bus, message)
        bus._process_messages_async(process_once=True)
        return result

    bus.publish = publish_and_process  # type: ignore[attr-defined]
    return bus


def _make_message(topic: str, message_type: MessageType = MessageType.EVENT, ttl: int = 10) -> Message:
    return Message(
        message_id=str(uuid.uuid4()),
        message_type=message_type,
        topic=topic,
        sender="tester",
        payload={},
        priority=MessagePriority.NORMAL,
        ttl=ttl,
    )


def test_message_expiration_and_publish(sync_bus):
    fresh = _make_message("topic.fresh", ttl=5)
    assert fresh.is_expired() is False

    expired = _make_message("topic.expired", ttl=0)
    expired.timestamp = datetime.now() - timedelta(seconds=5)
    assert expired.is_expired() is True
    assert sync_bus.publish(expired) is False


def test_queue_full_drops_message():
    bus = ComponentBus(enable_async=False, max_queue_size=1)
    message = _make_message("topic.one")
    assert ComponentBus.publish(bus, message) is True

    second = _make_message("topic.two")
    assert ComponentBus.publish(bus, second) is False
    stats = bus.get_stats()
    assert stats["messages_dropped"] == 1


def test_publish_handles_generic_exception(monkeypatch):
    bus = ComponentBus(enable_async=False)
    monkeypatch.setattr(bus.message_queue, "put", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("fail")))
    assert bus.publish(_make_message("topic.error")) is False


def test_unsubscribe_returns_false(sync_bus):
    assert sync_bus.unsubscribe("nonexistent") is False


def test_send_command_success(sync_bus):
    received = []

    def command_handler(msg: Message):
        received.append(msg.payload["value"])
        response_msg = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            topic=f"response.{msg.correlation_id}",
            sender="worker",
            payload={"status": "ok"},
            correlation_id=msg.correlation_id,
        )
        sync_bus.publish(response_msg)

    sync_bus.subscribe("worker", "command.worker.do", command_handler)

    result = sync_bus.send_command("worker", "do", {"value": 42}, timeout=1)
    assert received == [42]
    assert result == {"status": "ok"}


def test_send_command_publish_failure(sync_bus, monkeypatch):
    bus = ComponentBus(enable_async=False)
    bus.publish = lambda message: False  # type: ignore[assignment]
    assert bus.send_command("worker", "do", {}) is None


def test_send_command_timeout(sync_bus, monkeypatch):
    def fake_publish(message):
        sync_bus.message_queue.put((0, 0, message))
        return True

    sync_bus.publish = fake_publish  # type: ignore[assignment]
    monkeypatch.setattr(threading.Event, "wait", lambda self, timeout=None: False)
    result = sync_bus.send_command("worker", "noop", {}, timeout=0.01)
    assert result is None


def test_query_collects_results(sync_bus):
    responses = []

    def query_handler(msg: Message):
        responses.append(msg.payload["q"])
        reply = Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RESPONSE,
            topic=f"query.{msg.correlation_id}.response",
            sender="service",
            payload={"result": msg.payload["q"] * 2},
            correlation_id=msg.correlation_id,
        )
        sync_bus.publish(reply)

    sync_bus.subscribe("service", "metrics.compute", query_handler)

    results = sync_bus.query("metrics.compute", {"q": 21}, timeout=0.05)
    assert responses == [21]
    assert results == [{"result": 42}]


def test_query_publish_failure(sync_bus):
    bus = ComponentBus(enable_async=False)
    bus.publish = lambda message: False  # type: ignore[assignment]
    assert bus.query("topic", {}) == []


def test_process_messages_async_handles_errors():
    bus = ComponentBus(enable_async=False)

    def failing_handler(msg: Message):
        raise RuntimeError("boom")

    bus.subscribe("svc", "event.fail", failing_handler)
    ComponentBus.publish(bus, _make_message("event.fail"))
    bus._handle_message = failing_handler  # force exception path
    bus._process_messages_async(process_once=True)
    assert bus.processing_errors >= 1


def test_shutdown_clears_queue(sync_bus):
    sync_bus.publish(_make_message("topic.shutdown"))
    sync_bus.shutdown()
    assert sync_bus.running is False
    assert sync_bus.message_queue.qsize() == 0


def test_publish_event_uses_global_bus(monkeypatch, sync_bus):
    monkeypatch.setattr(
        "src.infrastructure.monitoring.core.component_bus.global_component_bus",
        sync_bus,
        raising=False,
    )

    events = []

    def handler(msg: Message):
        events.append(msg.payload)

    sync_bus.subscribe("listener", "alerts.raised", handler)
    publish_event("alerts.raised", {"k": "v"}, sender="tester")
    assert events == [{"k": "v"}]


def test_send_notification_uses_high_priority(monkeypatch, sync_bus):
    captures = []

    def fake_publish(message):
        captures.append(message)
        return True

    monkeypatch.setattr(
        "src.infrastructure.monitoring.core.component_bus.global_component_bus",
        SimpleNamespace(publish=fake_publish),
        raising=False,
    )

    assert send_notification("notify.topic", {"hello": "world"}, sender="tester") is True
    assert captures
    assert captures[0].priority == MessagePriority.HIGH
