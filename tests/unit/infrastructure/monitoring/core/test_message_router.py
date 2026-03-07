import queue
import threading
from datetime import datetime

import pytest

import src.infrastructure.monitoring.core.message_router as router_module
from src.infrastructure.monitoring.core.component_bus import MessageType


class DummyMessage:
    def __init__(
        self,
        message_type=None,
        *,
        type=None,
        target="component",
        source="tester",
        data=None,
        correlation_id=None,
    ):
        self.type = message_type if message_type is not None else type
        self.target = target
        self.source = source
        self.data = data or {}
        self.correlation_id = correlation_id or "corr-1"


@pytest.fixture
def message_router(monkeypatch):
    monkeypatch.setattr(router_module, "Message", DummyMessage)
    router = router_module.MessageRouter(max_workers=2, queue_size=5)
    original_shutdown = router.executor.shutdown

    def safe_shutdown(wait=True, timeout=None):
        return original_shutdown(wait=wait)

    router.executor.shutdown = safe_shutdown
    yield router
    router.shutdown()


def _shutdown_safe(router):
    original = router.executor.shutdown

    def safe_shutdown(wait=True, timeout=None):
        return original(wait=wait)

    router.executor.shutdown = safe_shutdown
    router.shutdown()
    router.executor.shutdown = original


def test_route_command_sync(message_router):
    responses = []

    def handler(message):
        responses.append(message.target)
        return "ok"

    message_router.register_response_handler(str(MessageType.COMMAND), handler)
    msg = DummyMessage(MessageType.COMMAND, target="service")

    result = message_router.route_message(msg)
    assert result == "ok"
    assert responses == ["service"]
    assert message_router.stats["messages_processed"] == 1


def test_route_unknown_type(message_router, caplog):
    msg = DummyMessage("UNKNOWN")
    result = message_router.route_message(msg)
    assert result is None
    assert any("未知消息类型" in record.message for record in caplog.records)


def test_route_message_failure(monkeypatch, message_router, caplog):
    def boom(*args, **kwargs):
        raise RuntimeError("route fail")

    monkeypatch.setattr(message_router, "_handle_sync_message", boom)
    msg = DummyMessage(MessageType.COMMAND)

    with caplog.at_level("ERROR"):
        result = message_router.route_message(msg)
    assert result is None
    assert message_router.stats["messages_failed"] == 1
    assert any("消息路由失败" in record.message for record in caplog.records)


def test_route_event_async_with_callback(message_router):
    callback_results = []
    done = threading.Event()

    def callback(message, error):
        callback_results.append((message.target, error))
        done.set()

    msg = DummyMessage(MessageType.EVENT, target="evt")
    message_router.route_message(msg, callback=callback)

    assert done.wait(timeout=1.0)
    assert callback_results == [("evt", None)]
    assert message_router.stats["messages_queued"] == 1
    assert message_router.stats["async_operations"] == 1


def test_handle_async_queue_full(message_router, monkeypatch):
    def raise_full(item, timeout=1):
        raise queue.Full

    monkeypatch.setattr(message_router.message_queue, "put", raise_full)

    msg = DummyMessage(MessageType.EVENT)
    message_router._handle_async_message(msg, None)
    assert message_router.stats["messages_failed"] == 1


def test_handle_async_message_internal_event_and_callback(message_router):
    msg = DummyMessage(MessageType.EVENT, target="event" )
    errors = []

    def callback(message, error):
        errors.append(error)

    message_router._handle_async_message_internal(msg, callback)
    assert errors == [None]


def test_handle_async_message_internal_error(message_router, caplog):
    msg = DummyMessage(MessageType.NOTIFICATION, target="notify")

    def bad_callback(message, error):
        raise RuntimeError("callback boom")

    with caplog.at_level("ERROR"):
        message_router._handle_async_message_internal(msg, bad_callback)
    assert any("异步回调执行失败" in record.message for record in caplog.records)


def test_broadcast_message_success(message_router):
    def handler(message):
        return f"ack:{message.target}"

    message_router.register_response_handler(str(MessageType.COMMAND), handler)
    msg = DummyMessage(MessageType.COMMAND, data={"value": 5})

    result = message_router.broadcast_message(msg, targets=["a", "b"])
    assert result["total_targets"] == 2
    assert result["successful"] == 2
    assert result["failed"] == 0
    assert result["results"]["a"]["result"] == "ack:a"


def test_broadcast_message_failure(message_router, monkeypatch):
    def bad_route(message):
        raise RuntimeError("route error")

    monkeypatch.setattr(message_router, "route_message", bad_route)
    msg = DummyMessage(MessageType.COMMAND)

    summary = message_router.broadcast_message(msg, targets=["x"])
    assert summary["failed"] == 1
    assert summary["results"]["x"]["success"] is False


def test_register_and_unregister_handlers(message_router):
    message_router.register_response_handler("my_type", lambda m: "ok")
    assert "my_type" in message_router.response_handlers
    message_router.unregister_response_handler("my_type")
    assert "my_type" not in message_router.response_handlers


def test_handle_sync_message_missing_handler(message_router, caplog):
    msg = DummyMessage(MessageType.COMMAND)
    with caplog.at_level("WARNING"):
        result = message_router._handle_sync_message(msg)
    assert result is None
    assert any("未找到同步消息处理器" in record.message for record in caplog.records)


def test_handle_sync_message_handler_error(message_router, caplog):
    def bad_handler(message):
        raise RuntimeError("sync fail")

    message_router.register_response_handler(str(MessageType.COMMAND), bad_handler)
    msg = DummyMessage(MessageType.COMMAND)

    with caplog.at_level("ERROR"):
        result = message_router._handle_sync_message(msg)
    assert result is None
    assert any("同步消息处理失败" in record.message for record in caplog.records)


def test_collect_results(message_router):
    future_ok = message_router.executor.submit(lambda: "done")

    def raise_error():
        raise RuntimeError("boom")

    future_fail = message_router.executor.submit(raise_error)
    summary = message_router.collect_results([future_ok, future_fail])

    assert summary["total"] == 2
    assert summary["completed"] == 1
    assert summary["failed"] == 1
    assert {"error": "boom"} in summary["results"]


def test_get_message_stats(message_router):
    msg = DummyMessage(MessageType.COMMAND)
    message_router.route_message(msg)
    stats = message_router.get_message_stats()

    assert "messages_processed" in stats
    assert stats["queue_size"] == 0
    assert stats["messages_processed"] == 1


def test_health_status_warning(message_router, monkeypatch):
    stats = {
        "messages_processed": 10,
        "messages_failed": 2,
        "messages_queued": 9,
        "async_operations": 5,
        "success_rate": 0.8,
        "queue_size": 5,
        "uptime_seconds": 1,
        "throughput_per_second": 10,
    }
    monkeypatch.setattr(message_router, "get_message_stats", lambda: stats)
    message_router.processing_thread = None

    health = message_router.get_health_status()
    assert health["status"] == "warning"
    assert any("异步处理线程未运行" in issue for issue in health["issues"])


def test_health_status_queue_backlog(message_router, monkeypatch):
    stats = message_router.get_message_stats()
    stats.update({"queue_size": message_router.message_queue.maxsize})
    monkeypatch.setattr(message_router, "get_message_stats", lambda: stats)
    health = message_router.get_health_status()
    assert any("消息队列积压严重" in issue for issue in health["issues"])


def test_health_status_error(message_router, monkeypatch):
    monkeypatch.setattr(
        message_router, "get_message_stats", lambda: (_ for _ in ()).throw(RuntimeError("fail"))
    )

    health = message_router.get_health_status()
    assert health["status"] == "error"
    assert health["error"] == "fail"


def test_shutdown_stops_processing_thread(message_router, monkeypatch):
    thread = threading.Thread(target=lambda: None)
    thread.start()
    message_router.processing_thread = thread
    message_router.shutdown(timeout=0.1)
    assert not thread.is_alive()


def test_global_message_router(monkeypatch):
    router = router_module.MessageRouter(max_workers=1)
    monkeypatch.setattr(router_module, "global_message_router", router)
    assert router_module.global_message_router is router
    _shutdown_safe(router)

