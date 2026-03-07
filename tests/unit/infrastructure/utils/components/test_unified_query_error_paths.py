#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
针对 unified_query 模块的错误和降级分支补充用例。
"""

from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.infrastructure.utils.components.unified_query import (
    QueryRequest,
    QueryResult,
    QueryType,
    StorageType,
    UnifiedQueryInterface,
)


def _make_request(query_type: QueryType, **extra: Any) -> QueryRequest:
    now = datetime.now()
    return QueryRequest(
        query_id=f"{query_type.value}-req",
        query_type=query_type,
        data_type=extra.get("data_type", "tick"),
        symbols=extra.get("symbols", ["AAA"]),
        start_time=extra.get("start_time", now - timedelta(minutes=1)),
        end_time=extra.get("end_time", now),
        storage_preference=extra.get("storage_preference"),
        aggregation=extra.get("aggregation"),
        filters=extra.get("filters"),
        limit=extra.get("limit"),
    )


def _make_result(success: bool = True, query_id: str = "req") -> QueryResult:
    df = pd.DataFrame({"symbol": ["AAA"], "value": [1]})
    return QueryResult(query_id=query_id, success=success, data=df)


@pytest.fixture
def interface() -> UnifiedQueryInterface:
    ui = UnifiedQueryInterface({"cache_enabled": True, "cache_ttl": 1})
    # 使用 fallback 缓存逻辑，避免依赖外部组件
    ui._cache_manager = None
    yield ui
    ui.query_executor.shutdown(wait=True)
    ui.async_executor.shutdown(wait=True)


def test_query_data_propagates_executor_error(interface: UnifiedQueryInterface) -> None:
    request = _make_request(QueryType.REALTIME)
    with patch.object(interface.async_executor, "submit", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            interface.query_data(request)


def test_execute_query_async_returns_error_result(interface: UnifiedQueryInterface) -> None:
    interface.cache_enabled = True
    interface._cache_manager = None
    request = _make_request(QueryType.REALTIME)
    with patch.object(interface, "_execute_realtime_query", side_effect=ValueError("bad type")):
        result = interface._execute_query_async(request)
        assert result.success is False
        assert result.error_message == "bad type"


def test_query_realtime_data_missing_validator(interface: UnifiedQueryInterface) -> None:
    interface._validator = None
    with patch.object(interface, "_execute_query_async", return_value=_make_result()) as mock_exec:
        result = interface.query_realtime_data(["AAA"])
    mock_exec.assert_called_once()
    assert result.success is True


def test_register_adapter_logs_warning(interface: UnifiedQueryInterface) -> None:
    adapter = object()
    with patch("src.infrastructure.utils.components.unified_query.logger") as mock_logger:
        interface.register_adapter("unknown", adapter)  # type: ignore[arg-type]
        mock_logger.warning.assert_called_once()


def test_unregister_adapter_missing(interface: UnifiedQueryInterface) -> None:
    with patch("src.infrastructure.utils.components.unified_query.logger") as mock_logger:
        interface.unregister_adapter(StorageType.PARQUET)
        mock_logger.warning.assert_called_once()


def test_handle_query_exception_with_none_requests(interface: UnifiedQueryInterface) -> None:
    results = interface._handle_query_exception(RuntimeError("oops"), [])
    assert results == []


def test_process_query_results_handles_non_iterable(interface: UnifiedQueryInterface) -> None:
    grouped = {"realtime": [_make_request(QueryType.REALTIME)]}
    processed = interface._process_query_results("not-iterable", grouped)  # type: ignore[arg-type]
    assert len(processed) == 1
    assert processed[0].success is False
    assert "not-iterable" in processed[0].error_message


def test_cache_result_fallback_skips_when_disabled(interface: UnifiedQueryInterface) -> None:
    interface.cache_enabled = False
    request = _make_request(QueryType.REALTIME)
    interface._cache_result(request, _make_result())
    assert interface.query_cache == {}


