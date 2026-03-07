#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
from unittest.mock import MagicMock, patch

import pytest

from src.infrastructure.logging.advanced.advanced_logger import AdvancedLogger, LogEntry
from src.infrastructure.logging.core import LogLevel


@pytest.fixture()
def logger_sync():
    """构建同步模式的 AdvancedLogger，避免线程副作用。"""
    return AdvancedLogger("sync", enable_async=False, enable_monitoring=True)


def test_log_structured_merges_payload(logger_sync):
    with patch.object(logger_sync, "log") as mock_log:
        payload = logger_sync.log_structured(
            LogLevel.INFO,
            {"message": "hello", "request_id": "req-1"},
            user="alice",
        )

    mock_log.assert_called_once_with(LogLevel.INFO.value, "hello")
    assert payload["request_id"] == "req-1"
    assert payload["user"] == "alice"


def test_do_async_log_updates_statistics(logger_sync):
    logger_sync.enable_monitoring = True
    with patch.object(logger_sync, "log_structured"):
        logger_sync._do_async_log(LogLevel.WARNING, "async-event")

    stats = logger_sync._performance_stats
    assert stats["log_count"] == 1
    assert stats["total_logs"] == 1
    assert stats["max_processing_time"] >= 0.0
    assert stats["min_processing_time"] <= stats["max_processing_time"]


def test_should_filter_blocks_entry(logger_sync):
    logger_sync.add_filter(lambda entry: entry.level != "FORBIDDEN")
    entry = LogEntry(timestamp=time.time(), level="FORBIDDEN", message="blocked")

    assert logger_sync._should_filter(entry) is True


def test_log_with_performance_tracking_sync_path(logger_sync):
    with patch.object(logger_sync, "log_structured") as mock_structured, patch.object(
        logger_sync, "_update_performance_stats"
    ) as mock_stats:
        logger_sync.log_with_performance_tracking(LogLevel.ERROR, "sync-op", "op-id", detail="x")

    mock_structured.assert_called_once()
    mock_stats.assert_called_once()


def test_log_async_falls_back_when_disabled(logger_sync):
    with patch.object(logger_sync, "log_structured") as mock_structured:
        logger_sync.log_async(LogLevel.DEBUG, "fallback", foo=1)

    mock_structured.assert_called_once_with(LogLevel.DEBUG, "fallback", foo=1)


def test_update_config_with_string_level(logger_sync):
    with patch("logging.getLevelName", return_value="WARNING"):
        logger_sync.update_config({"level": "DEBUG"})

    assert logger_sync.level is LogLevel.WARNING


def test_update_config_with_enum_level(logger_sync):
    logger_sync.update_config({"level": LogLevel.CRITICAL})
    assert logger_sync.level is LogLevel.CRITICAL


def test_log_batch_handles_dict_and_plain(logger_sync):
    with patch.object(logger_sync, "log") as mock_log, patch.object(
        logger_sync, "info"
    ) as mock_info:
        logger_sync.log_batch(
            [
                {"level": "WARNING", "msg": "dict-msg"},
                "plain-text",
            ]
        )

    mock_log.assert_called_once_with("WARNING", "dict-msg")
    mock_info.assert_called_once_with("plain-text")


def test_log_security_event_uses_warning(logger_sync):
    with patch.object(logger_sync, "log") as mock_log:
        logger_sync.log_security_event("unauthorized", {"ip": "127.0.0.1"})

    mock_log.assert_called_once()
    level_arg, message_arg = mock_log.call_args[0][:2]
    assert level_arg == LogLevel.WARNING.value
    assert "unauthorized" in message_arg


def test_shutdown_closes_async_resources():
    logger = AdvancedLogger("async", enable_async=False)
    fake_executor = MagicMock()
    fake_loop = MagicMock()
    logger.enable_async = True
    logger._async_executor = fake_executor
    logger._event_loop = fake_loop

    logger.shutdown()

    fake_executor.shutdown.assert_called_once_with(wait=True)
    fake_loop.stop.assert_called_once()


def test_log_performance_delegates(logger_sync):
    with patch.object(logger_sync, "_log_performance") as mock_lp:
        logger_sync.log_performance("load-data", 0.42, {"rows": 10})

    mock_lp.assert_called_once_with("load-data", 0.42, {"rows": 10})



