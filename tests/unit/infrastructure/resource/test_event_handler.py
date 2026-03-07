import asyncio
import types

import pytest
from unittest.mock import MagicMock

from src.infrastructure.resource.core.event_handler import EventHandler


class ImmediateThread:
    def __init__(self, target, args=(), kwargs=None, daemon=None):
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.daemon = daemon

    def start(self):
        self._target(*self._args, **self._kwargs)


def test_register_and_dispatch_sync_handler():
    logger = MagicMock()
    handler = EventHandler(logger=logger)
    received = []

    def on_event(data):
        received.append(data)

    handler.register_handler("sync", on_event)
    handler.dispatch_event("sync", {"value": 1})

    assert received == [{"value": 1}]
    logger.log_error.assert_not_called()


def test_register_async_handler_dispatch_triggers_executor(monkeypatch):
    logger = MagicMock()
    handler = EventHandler(logger=logger)
    async_handler = MagicMock()
    handler._execute_async_handler = MagicMock()

    monkeypatch.setattr("threading.Thread", ImmediateThread)

    handler.register_async_handler("async", async_handler)
    handler.dispatch_event("async", {"value": 2})

    handler._execute_async_handler.assert_called_once_with(async_handler, {"value": 2})


def test_unregister_handler_removes_sync_and_async():
    handler = EventHandler(logger=MagicMock())

    def sync_handler(data):
        pass

    async def async_handler(data):
        return None

    handler.register_handler("event", sync_handler)
    handler.register_async_handler("event", async_handler)

    assert handler.get_handler_count("event") == 2

    handler.unregister_handler("event", sync_handler)
    handler.unregister_handler("event", async_handler)

    assert handler.get_handler_count("event") == 0


def test_dispatch_sync_handler_logs_on_exception():
    logger = MagicMock()
    handler = EventHandler(logger=logger)

    def failing_handler(_):
        raise RuntimeError("Boom")

    handler.register_handler("error", failing_handler)
    handler.dispatch_event("error", {})

    logger.log_error.assert_called_once()
    assert "事件处理器执行失败" in logger.log_error.call_args[0][0]


def test_execute_async_handler_success():
    logger = MagicMock()
    handler = EventHandler(logger=logger)
    invoked = {}

    async def async_handler(event):
        invoked["value"] = event["value"]

    handler._execute_async_handler(async_handler, {"value": 42})

    assert invoked["value"] == 42
    logger.log_error.assert_not_called()


def test_execute_async_handler_logs_on_failure():
    logger = MagicMock()
    handler = EventHandler(logger=logger)

    async def failing_handler(_):
        raise ValueError("async boom")

    handler._execute_async_handler(failing_handler, {})

    logger.log_error.assert_called_once()
    assert "异步处理器执行异常" in logger.log_error.call_args[0][0]


def test_clear_handlers_specific_and_all():
    logger = MagicMock()
    handler = EventHandler(logger=logger)

    handler.register_handler("one", lambda _: None)
    handler.register_async_handler("one", lambda _: None)
    handler.register_handler("two", lambda _: None)

    handler.clear_handlers("one")
    assert handler.get_handler_count("one") == 0
    assert handler.get_handler_count("two") == 1

    handler.clear_handlers()
    assert handler.get_handler_count("two") == 0
    logger.log_info.assert_called()


def test_get_handler_count_combines_sync_and_async():
    handler = EventHandler(logger=MagicMock())

    handler.register_handler("combined", lambda _: None)
    handler.register_async_handler("combined", lambda _: None)

    assert handler.get_handler_count("combined") == 2


