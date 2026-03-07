#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UnifiedErrorHandler 测试，覆盖统计、日志与降级分支。"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import pytest

from src.infrastructure.utils.core import error as err


def _make_handler(monkeypatch: pytest.MonkeyPatch) -> Tuple[err.UnifiedErrorHandler, List[Tuple[str, str]]]:
    handler = err.UnifiedErrorHandler("tests.error")
    log_calls: List[Tuple[str, str]] = []

    def _log(level: str):
        def _inner(message: str, exc_info: bool = False) -> None:
            log_calls.append((level, message))
        return _inner

    fake_logger = SimpleNamespace(
        debug=_log("debug"),
        info=_log("info"),
        warning=_log("warning"),
        error=_log("error"),
        critical=_log("critical"),
    )
    monkeypatch.setattr(handler, "logger", fake_logger)
    return handler, log_calls


def test_handle_updates_stats_and_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, calls = _make_handler(monkeypatch)

    handler.handle(ValueError("bad data"), context="loader", level="warning")
    assert handler.error_stats == {"ValueError": 1}
    assert handler.last_errors[-1]["context"] == "loader"
    assert calls == [("warning", "[loader] ValueError: bad data")]

    handler.handle_connection_error(ConnectionError("conn fail"), host="db", port=5432)
    assert handler.error_stats["ConnectionError"] == 1
    assert calls[-1][1].startswith("[Connection failed to db:5432]")

    handler.clear_stats()
    assert handler.error_stats == {}
    assert handler.last_errors == []


def test_manage_error_history_limits_size(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, _ = _make_handler(monkeypatch)
    for i in range(105):
        handler.handle(ValueError(f"err{i}"))
    assert len(handler.last_errors) == 100
    assert handler.last_errors[0]["message"].endswith("err5")


def test_logging_failure_prints_fallback(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    handler = err.UnifiedErrorHandler("tests.error")

    def _boom(message: str, exc_info: bool = False) -> None:
        raise RuntimeError("logger down")

    fake_logger = SimpleNamespace(
        debug=_boom,
        info=_boom,
        warning=_boom,
        error=_boom,
        critical=_boom,
    )
    monkeypatch.setattr(handler, "logger", fake_logger)

    handler.handle(RuntimeError("broken"))
    out = capsys.readouterr().out
    assert "Logging failed" in out
    assert "Original error" in out


def test_record_error_failure_prints(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    handler, calls = _make_handler(monkeypatch)

    def _fail_create(*args: Any, **kwargs: Any) -> Dict[str, Any]:
        raise ValueError("no record")

    monkeypatch.setattr(handler, "_create_error_info", _fail_create)
    handler.handle(ValueError("oops"))
    out = capsys.readouterr().out
    assert "Failed to record error info" in out
    assert calls  # logging仍被调用


def test_final_failure_path(capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
    handler, _ = _make_handler(monkeypatch)

    def _boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("stats failed")

    monkeypatch.setattr(handler, "_update_error_stats", _boom)
    handler.handle(ValueError("fatal"))
    out = capsys.readouterr().out
    assert "Critical error in error handler" in out
    assert "Original error was" in out


def test_get_recent_errors_return_in_reverse(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, _ = _make_handler(monkeypatch)
    for i in range(3):
        handler.handle(ValueError(str(i)))
    recent = handler.get_recent_errors(limit=2)
    assert [item["message"] for item in recent] == ["2", "1"]


def test_handle_timeout_and_validation_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    handler, calls = _make_handler(monkeypatch)

    handler.handle_timeout_error(TimeoutError("slow"), operation="sync", timeout=1.5)
    handler.handle_validation_error(ValueError("bad field"), field="name", value="")

    assert any("Operation 'sync' timed out" in msg for _, msg in calls)
    assert any("Validation failed for field 'name'" in msg for _, msg in calls)

    stats = handler.get_error_stats()
    assert stats["total_errors"] == 2
    assert stats["recent_errors_count"] == 2
