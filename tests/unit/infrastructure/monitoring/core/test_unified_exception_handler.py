import time
from unittest.mock import MagicMock

import pytest

from src.infrastructure.monitoring.core.unified_exception_handler import (
    AlertProcessingError,
    ConfigurationError,
    ConnectionError as MonitoringConnectionError,
    ExceptionHandler,
    ExceptionHandlingContext,
    LogAndContinueStrategy,
    MonitoringException,
    NotificationError,
    RetryStrategy,
    ValidationError,
    handle_monitoring_exception,
    with_exception_handling,
)


def test_monitoring_exception_to_dict():
    try:
        raise ValueError("boom")
    except ValueError as cause:
        exc = MonitoringException("failed", error_code="TEST", context={"key": "v"}, cause=cause)
    data = exc.to_dict()
    assert data["error_code"] == "TEST"
    assert data["context"]["key"] == "v"
    assert "cause" in data


def test_log_and_continue_strategy_returns_default(caplog):
    strategy = LogAndContinueStrategy()
    result = strategy.handle(RuntimeError("boom"), {"operation": "op", "default_value": 123})
    assert result == 123
    assert any("操作 'op'" in record.message for record in caplog.records)


def test_retry_strategy_success(monkeypatch):
    attempts = []

    def func():
        attempts.append(True)
        if len(attempts) < 2:
            raise RuntimeError("try again")
        return "ok"

    monkeypatch.setattr(time, "sleep", lambda _: None)
    strategy = RetryStrategy(max_retries=2, delay_seconds=0.01)
    result = strategy.handle(RuntimeError("fail"), {"operation": "op", "func": func})
    assert result == "ok"
    assert len(attempts) == 2


def test_retry_strategy_exceeds(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda _: None)
    strategy = RetryStrategy(max_retries=1, delay_seconds=0.01)
    with pytest.raises(RuntimeError):
        strategy.handle(RuntimeError("fail"), {"operation": "op", "func": lambda: (_ for _ in ()).throw(RuntimeError("x"))})


def test_exception_handler_default_strategy_mapping():
    handler = ExceptionHandler()
    assert handler._get_default_strategy(NotificationError("msg")) == "log_and_continue"
    assert handler._get_default_strategy(ConfigurationError("msg")) == "raise"
    assert handler._get_default_strategy(RuntimeError("msg")) == "log_and_continue"


def test_handle_exception_unknown_strategy_raises():
    handler = ExceptionHandler()
    with pytest.raises(ValueError):
        handler.handle_exception(ValueError("boom"), strategy="unknown")


def test_handle_exception_custom_context(monkeypatch):
    handler = ExceptionHandler()

    class CustomStrategy:
        def __init__(self):
            self.last_context = None

        def handle(self, exception, context):
            self.last_context = context
            return "fallback"

    strategy = CustomStrategy()
    handler.add_strategy("custom", strategy)  # type: ignore[arg-type]
    result = handler.handle_exception(ValueError("boom"), operation="test", strategy="custom", context={"extra": 1})
    assert result == "fallback"
    assert strategy.last_context["extra"] == 1


def test_handle_monitoring_exception_decorator(monkeypatch):
    handler = ExceptionHandler()
    monkeypatch.setattr("src.infrastructure.monitoring.core.unified_exception_handler.ExceptionHandler", lambda: handler)
    spy = MagicMock(return_value="logged")
    handler.add_strategy("log_and_continue", MagicMock(handle=spy))

    @handle_monitoring_exception(operation="test", strategy="log_and_continue", context={"default_value": "logged"})
    def func():
        raise RuntimeError("boom")

    func()
    spy.assert_called_once()


def test_with_exception_handling_context_manager():
    data = {}

    class SpyHandler(ExceptionHandler):
        def handle_exception(self, exception, operation, strategy=None, context=None):
            data["handled"] = (operation, strategy)
            return "handled"

    ctx = ExceptionHandlingContext(operation="op", strategy="log_and_continue")
    ctx.handler = SpyHandler()

    with ctx:
        raise RuntimeError("boom")

    assert data["handled"] == ("op", "log_and_continue")


def test_handle_exception_retry_default(monkeypatch):
    handler = ExceptionHandler()
    monkeypatch.setattr(time, "sleep", lambda _: None)
    calls = []

    def func():
        calls.append(True)
        return "done"

    res = handler.handle_exception(MonitoringConnectionError("fail"), context={"func": func}, operation="connect")
    assert res == "done"
    assert calls
